import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load pre-trained Llama3-8B (user must have HF access and auth for gated model)
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
backbone = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

# Freeze backbone base params
for param in backbone.parameters():
    param.requires_grad = False

# Apply LoRA for parameter efficiency
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
backbone = get_peft_model(backbone, lora_config)

# EBT Adapter: Small MLP for energy E(h_x, h_y) - updated to handle sequence h_y [batch, seq_len, hidden]
class EBTAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_x, h_y):
        # Mean-pool h_y over seq_len for energy computation
        h_y_pooled = h_y.mean(dim=1)
        concat = torch.cat((h_x.to(torch.float32), h_y_pooled.to(torch.float32)), dim=-1)
        return self.mlp(concat).squeeze(-1)

adapter = EBTAdapter(backbone.config.hidden_size).to(backbone.device)

# Helper to get mean-pooled hidden states (with optional no_grad for memory efficiency)
def get_hidden_states(backbone, input_ids, attention_mask=None, no_grad=True):
    if no_grad:
        with torch.no_grad():
            outputs = backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    else:
        outputs = backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return outputs.hidden_states[-1].mean(dim=1)

# Load Alpaca dataset (adjust to 2000 samples as in your run)
dataset = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=42).select(range(2000))  # 2000 examples
train_size = 1800
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, 2000))

# Data collation function
def collate_fn(batch, tokenizer, max_len=128):
    prompts = [f"{item['instruction']} {item['input']}" for item in batch]
    outputs = [item['output'] for item in batch]
    x = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    y = tokenizer(outputs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return x.to(backbone.device), y.to(backbone.device)

# Create data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer))

def train_ebt_adapter(backbone, adapter, train_loader, epochs=3, gen_len=10):
    # Optimizer and device setup
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    device = next(backbone.parameters()).device
    backbone.to(device)
    adapter.to(device)

    # EBT hyperparameters (based on paper: small number of optimization steps)
    alpha = 0.01  # Step size for gradient descent on energy
    num_opt_steps = 3  # Number of refinement steps (keep small for memory)
    d = backbone.config.hidden_size  # e.g., 4096

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Unpack batch as tokenized dicts
            x_tokens, y_tokens = batch

            context_tokens = x_tokens
            target_tokens = y_tokens

            # Get context features x [b, d]
            with torch.no_grad():
                context_outputs = backbone(**context_tokens, output_hidden_states=True)
                x = context_outputs.hidden_states[-1].mean(dim=1)  # Mean pool over sequence

            # Get true hidden states (use embeddings of target for continuous space)
            with torch.no_grad():
                h_y_true = backbone.get_input_embeddings()(target_tokens['input_ids'])  # [b, true_len, d]

            # Sample initial hat_y [b, gen_len, d]
            b = x.shape[0]
            hat_y = torch.randn(b, gen_len, d, device=device, requires_grad=True)

            # EBT optimization loop: refine hat_y by minimizing energy
            energy = None  # To store last energy
            for _ in range(num_opt_steps):
                energy = adapter(x, hat_y)  # Assume adapter(x, hat_y) returns energy [b] or [b, 1]
                if energy.dim() > 1:
                    energy = energy.squeeze(1)
                grad = torch.autograd.grad(energy.sum(), hat_y, create_graph=True)[0]
                hat_y = hat_y - alpha * grad  # Creates new tensor, preserves graph

            # Fixed loss calculation with slicing for shape mismatch
            true_len = h_y_true.shape[1]
            gen_len_current = hat_y.shape[1]
            min_len = min(true_len, gen_len_current)

            hat_y_sliced = hat_y[:, :min_len, :]
            h_y_true_sliced = h_y_true[:, :min_len, :].to(hat_y.dtype)

            loss = F.mse_loss(hat_y_sliced, h_y_true_sliced) + energy.mean()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")

    print("Training completed.")

# Updated Inference: Sequence optimization for coherent longer generations
def infer_ebt_adapter(backbone, adapter, x_batch, num_samples=10, base_opt_steps=30, max_opt_steps=60, alpha=0.01, sigma=0.1, uncertainty_threshold=0.5, gen_len=20):
    h_x = get_hidden_states(backbone, x_batch.input_ids, x_batch.attention_mask, no_grad=True)
    embed_matrix = backbone.get_input_embeddings().weight  # For nearest token mapping
    candidates = []

    for _ in range(num_samples):
        hat_y = torch.randn(x_batch.input_ids.shape[0], gen_len, backbone.config.hidden_size, dtype=torch.float32, device=h_x.device).requires_grad_()  # Use float32 for precision
        opt_steps = base_opt_steps

        # Initial refinement with annealed noise
        for step in range(opt_steps):
            energy = adapter(h_x, hat_y)
            energy_grad = grad(energy.sum(), hat_y, create_graph=False)[0]
            current_sigma = sigma * (1 - step / opt_steps)  # Anneal sigma
            hat_y = (hat_y - alpha * energy_grad + torch.randn_like(hat_y) * current_sigma).detach().requires_grad_()
            energy_grad = torch.clamp(energy_grad, -1.0, 1.0)  # Clip gradients

        # Uncertainty modeling
        sub_samples = [hat_y + torch.randn_like(hat_y) * 0.5 for _ in range(10)]
        sub_energies = torch.stack([adapter(h_x, ss).mean() for ss in sub_samples])
        uncertainty = sub_energies.std().item()

        # Dynamic compute
        if uncertainty > uncertainty_threshold:
            extra_steps = min(max_opt_steps - opt_steps, 5)
            for step in range(extra_steps):
                energy = adapter(h_x, hat_y)
                energy_grad = grad(energy.sum(), hat_y, create_graph=False)[0]
                current_sigma = sigma * (1 - (opt_steps + step) / max_opt_steps)
                hat_y = (hat_y - alpha * energy_grad + torch.randn_like(hat_y) * current_sigma).detach().requires_grad_()
                energy_grad = torch.clamp(energy_grad, -1.0, 1.0)
            opt_steps += extra_steps

        # Verification
        prev_hat_y = hat_y.clone()
        energy = adapter(h_x, hat_y)
        energy_grad = grad(energy.sum(), hat_y, create_graph=False)[0]
        hat_y = (hat_y - alpha * energy_grad).detach()
        change_norm = torch.norm(hat_y - prev_hat_y).item()
        prev_norm = torch.norm(prev_hat_y).item()
        relative_change = change_norm / (prev_norm + 1e-6)
        is_verified = relative_change < 0.05 or change_norm < 1.0

        final_energy = adapter(h_x, hat_y).mean().item()
        candidates.append((hat_y, final_energy, uncertainty, is_verified, opt_steps))

    # Select best
    best_candidate = min(candidates, key=lambda c: (c[1], -int(c[3])))
    best_hat_y, best_energy, uncertainty, is_verified, steps_used = best_candidate
    print(f"Best: Energy={best_energy:.2f}, Uncertainty={uncertainty:.2f}, Verified={is_verified}, Steps={steps_used}")

    # Map to valid tokens: Nearest neighbor in embedding matrix with normalization for cosine dist
    with torch.no_grad():
        best_hat_y = F.normalize(best_hat_y.to(embed_matrix.dtype), dim=-1)
        embed_matrix_norm = F.normalize(embed_matrix, dim=-1)
        dists = torch.cdist(best_hat_y.view(-1, best_hat_y.shape[-1]), embed_matrix_norm)

        # Top-k for diversity: Select top-3 per position, then rerank by energy of sequence
        topk = 3
        top_token_ids = dists.topk(topk, dim=1, largest=False)[1]  # [flat_len, topk]
        # Simple: Take argmin for now; for better, generate combos and score
        token_ids = dists.argmin(dim=1).view(best_hat_y.shape[0], -1)

        # Generate longer text autoregressively from mapped tokens
        full_input_ids = torch.cat((x_batch.input_ids, token_ids), dim=1)
        generated = backbone.generate(
            full_input_ids,
            max_new_tokens=100,  # Longer for non-debug
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            attention_mask=torch.cat((x_batch.attention_mask, torch.ones_like(token_ids)), dim=1)
        )
        decoded_text = tokenizer.batch_decode(generated[:, x_batch.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Generated Outputs: {decoded_text}")

    return best_hat_y

# Test function: Updated to include inference metrics and consistent loss
def test_ebt_adapter(backbone, adapter, test_loader):
    backbone.eval()
    adapter.eval()
    total_loss = 0
    for x_batch, y_batch in test_loader:
        h_x = get_hidden_states(backbone, x_batch.input_ids, x_batch.attention_mask, no_grad=True)
        h_y_true = backbone.get_input_embeddings()(y_batch.input_ids)  # [b, seq, d]
        # Generate hat_y for consistency with training
        b, d = h_x.shape
        gen_len = h_y_true.shape[1]
        hat_y = torch.randn(b, gen_len, d, device=h_x.device, dtype=torch.float32).requires_grad_()
        for _ in range(5):  # Quick opt for test
            energy = adapter(h_x, hat_y)
            grad_ = grad(energy.sum(), hat_y, create_graph=False)[0]
            hat_y = (hat_y - 0.01 * grad_).detach().requires_grad_()
        energy = adapter(h_x, hat_y)
        min_len = min(gen_len, h_y_true.shape[1])
        loss = F.mse_loss(hat_y[:, :min_len, :], h_y_true[:, :min_len, :].to(hat_y.dtype)) + energy.mean()
        total_loss += loss.item()

        # Run inference for demo
        infer_ebt_adapter(backbone, adapter, x_batch)

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

# Train on Alpaca subset
train_ebt_adapter(backbone, adapter, train_loader, epochs=3, gen_len=10)

# Evaluate on test subset (now includes inference with features)
test_ebt_adapter(backbone, adapter, test_loader)

# Sample inference
x_sample, _ = collate_fn([test_dataset[0]], tokenizer)
best_pred = infer_ebt_adapter(backbone, adapter, x_sample)
print("Sample inference output shape:", best_pred.shape)
