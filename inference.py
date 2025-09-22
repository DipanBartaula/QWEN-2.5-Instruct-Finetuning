import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-7B-Instruct"
checkpoint_path = "./checkpoints/checkpoint-1000.pt"  # replace with your LoRA checkpoint

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Sample prompt
prompt = "Create a CAD model of a simple cube with a 10mm side length."

# Role-based formatting
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_text = "\n".join([f"[{m['role']}]: {m['content']}" for m in messages])

# Tokenize
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=256)
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Code:\n", output)
