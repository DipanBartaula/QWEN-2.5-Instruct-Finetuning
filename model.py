import torch
from transformers import AutoModelForCausalLM, AutoTokenizer




def load_model_and_tokenizer(model_name: str, device_map: str = 'auto', use_bfloat16: bool = False):
dtype = torch.bfloat16 if use_bfloat16 and torch.cuda.is_available() else None
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token_id is None:
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
model_name,
device_map=device_map,
torch_dtype=dtype,
trust_remote_code=True,
low_cpu_mem_usage=True,
)
return model, tokenizer