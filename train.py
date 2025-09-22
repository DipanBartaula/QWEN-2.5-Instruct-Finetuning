import torch
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from dataloader import get_dataloaders
from utils import set_seed, save_checkpoint
import wandb

# -----------------------------
# Config
# -----------------------------
set_seed(42)

csv_path = "/content/NeurIPS11092/text2cad_v1.1.csv"
base_dir = "/content/CadQuery_data/CQ"
model_name = "Qwen/Qwen2.5-7B-Instruct"
batch_size = 2
lr = 3e-4
epochs = 3
max_length = 256
checkpoint_dir = "./checkpoints"
log_interval = 10
save_interval = 100

# -----------------------------
# W&B Init
# -----------------------------
wandb.init(project="text2cad-qwen-lora", name="qwen2.5-lora-ft")

# -----------------------------
# DataLoaders
# -----------------------------
print("Initializing DataLoaders...")
train_loader, val_loader = get_dataloaders(csv_path, base_dir, tokenizer_name=model_name, batch_size=batch_size, max_length=max_length, augment=True)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# -----------------------------
# Model + LoRA + Quantization
# -----------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
print("LoRA model ready.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# -----------------------------
# Training loop
# -----------------------------
global_step = 0
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % log_interval == 0:
            print(f"Step {global_step} | Train Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.2f}")

            # Validation
            model.eval()
            val_loss = 0
            val_count = 0
            for vbatch in val_loader:
                if vbatch is None:
                    continue
                v_input_ids = vbatch["input_ids"].to(device)
                v_attention_mask = vbatch["attention_mask"].to(device)
                v_labels = vbatch["labels"].to(device)
                with torch.no_grad():
                    v_outputs = model(input_ids=v_input_ids, attention_mask=v_attention_mask, labels=v_labels)
                    val_loss += v_outputs.loss.item()
                    val_count += 1
            val_loss /= max(val_count, 1)
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "grad_norm": grad_norm}, step=global_step)
            print(f"Validation loss at step {global_step}: {val_loss:.4f}")
            model.train()

        if global_step % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir)
            print(f"Checkpoint saved at step {global_step}")

print("Training completed!")
