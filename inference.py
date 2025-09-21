import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




def generate(text, model_name_or_path, max_new_tokens=256, device='cuda'):
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', trust_remote_code=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=max_new_tokens)
return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True)
args = parser.parse_args()
print(generate(args.prompt, args.model_path))


--- FILE: test.py ---
# Test script: argparse + compute perplexity using evaluate
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import evaluate




def perplexity(model, tokenizer, texts, batch_size=1, device='cuda'):
model.to(device)
model.eval()
ppl = []
for t in texts:
enc = tokenizer(t, return_tensors='pt', truncation=True)
input_ids = enc['input_ids'].to(device)
with torch.no_grad():
outputs = model(input_ids, labels=input_ids)
neg_log_likelihood = outputs.loss * input_ids.size(1)
ppl.append(math.exp((neg_log_likelihood / input_ids.size(1)).item()))
return sum(ppl) / len(ppl)




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--test-file', type=str, required=True)
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')


with open(args.test_file, 'r', encoding='utf-8') as f:
texts = [l.strip() for l in f if l.strip()]


ppl = perplexity(model, tokenizer, texts[:50], device=model.device)
print(f'Perplexity on sample: {ppl:.4f}')


if __name__ == '__main__':
main()