import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import random
import os

class Text2CadQueryDataset(Dataset):
    def __init__(self, csv_path, base_dir, tokenizer_name, max_length=256, split="train", val_ratio=0.1, augment=False):
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.augment = augment

        # Read CSV
        df = pd.read_csv(csv_path)
        train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)
        self.data = train_df if split=="train" else val_df
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Choose prompt level if augmentation is enabled
        prompt_fields = ["abstract", "beginner", "intermediate", "expert", "description", "keywords"]
        if self.augment:
            prompt_field = random.choice(prompt_fields)
        else:
            prompt_field = "abstract"

        prompt = row[prompt_field]
        uid = row['uid']
        code_path = os.path.join(self.base_dir, uid + ".py")

        if not os.path.exists(code_path):
            return None  # skip missing files

        with open(code_path, "r") as f:
            code = f.read()

        # Role-based formatting
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_text = "\n".join([f"[{m['role']}]: {m['content']}" for m in messages])

        # Tokenize input and labels
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.squeeze()
        }


def get_dataloaders(csv_path, base_dir, tokenizer_name, batch_size=2, max_length=256, augment=True):
    train_dataset = Text2CadQueryDataset(csv_path, base_dir, tokenizer_name, max_length=max_length, split="train", augment=augment)
    val_dataset = Text2CadQueryDataset(csv_path, base_dir, tokenizer_name, max_length=max_length, split="val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
