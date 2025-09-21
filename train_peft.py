import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--dataset-path', required=True)
parser.add_argument('--output-dir', default='./peft_outputs')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
args.model_name,
load_in_8bit=True,
device_map='auto',
trust_remote_code=True,
)


model = prepare_model_for_kbit_training(model)


peft_config = LoraConfig(
r=8,
lora_alpha=32,
target_modules=['q_proj','k_proj','v_proj','o_proj'],
lora_dropout=0.05,
bias='none',
task_type='CAUSAL_LM'
)


model = get_peft_model(model, peft_config)


ds = load_dataset('json', data_files={'train': args.dataset_path})['train']


def preprocess(example):
text = example.get('instruction','') + '
' + example.get('response','') if 'instruction' in example else example.get('text','')
return tokenizer(text, truncation=True, max_length=2048)


tokenized = ds.map(preprocess, batched=False)


training_args = TrainingArguments(
output_dir=args.output_dir,
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
max_steps=2000,
save_steps=500,
fp16=False,
logging_steps=50,
report_to='wandb'
)


trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized,
tokenizer=tokenizer,
)


trainer.train()
model.save_pretrained(args.output_dir)


if __name__ == '__main__':
main()