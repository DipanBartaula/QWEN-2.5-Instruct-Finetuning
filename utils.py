import os
import json
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




def save_json(obj: Any, path: str):
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w', encoding='utf-8') as f:
json.dump(obj, f, ensure_ascii=False, indent=2)




def load_json(path: str):
with open(path, 'r', encoding='utf-8') as f:
return json.load(f)


--- FILE: dataloader.py ---
# A generic dataloader that expects a JSONL file of instruction-response pairs.
# For WHUCAD you'll likely need to adapt this to whatever text/metadata format you want to train on.
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple




def load_whucad_as_text(dataset_path: str, split: str = 'train') -> Dataset:
"""Attempt to load WHUCAD if it's structured as JSON/JSONL with fields 'instruction' and 'response'.
Adjust mapping if the dataset uses different field names (e.g., 'prompt', 'target').
"""
if os.path.isdir(dataset_path):
# assume user has pushed a local dataset in jsonl format per-split
ds = load_dataset('json', data_files={split: os.path.join(dataset_path, f'{split}.jsonl')})[split]
else:
# try remote repo id (huggingface or git) or single jsonl file
ds = load_dataset('json', data_files={split: dataset_path})[split]


return ds




def preprocess_for_causal(ds, tokenizer_name: str, max_length: int = 2048, text_key: str = 'text'):
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
if tokenizer.pad_token_id is None:
tokenizer.pad_token = tokenizer.eos_token


def _concat(example):
# If dataset has instruction/response split, adapt here
if 'instruction' in example and 'response' in example:
txt = example['instruction'].strip() + "\n" + example['response'].strip()
elif text_key in example:
txt = example[text_key]
else:
# fallback: join values
txt = ' '.join(str(v) for v in example.values())
return tokenizer(txt, truncation=True, max_length=max_length)


tokenized = ds.map(_concat, batched=False)
return tokenized, tokenizer