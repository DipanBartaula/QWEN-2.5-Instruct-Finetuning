import os
parser.add_argument('--learning-rate', type=float, default=2e-5)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--report-to', type=str, default='wandb')
args = parser.parse_args()


# Initialize wandb (Trainer also initializes through report_to)
wandb.init(project='qwen-whucad-finetune', config=vars(args))


model, tokenizer = load_model_and_tokenizer(args.model_name)


ds = load_whucad_as_text(args.dataset_path, split='train')
tokenized_ds, tokenizer = preprocess_for_causal(ds, tokenizer_name=args.model_name)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
output_dir=args.output_dir,
per_device_train_batch_size=args.per_device_batch_size,
gradient_accumulation_steps=args.gradient_accumulation_steps,
max_steps=args.max_steps,
evaluation_strategy='no',
save_steps=args.save_steps,
logging_steps=args.logging_steps,
fp16=args.fp16,
learning_rate=args.learning_rate,
weight_decay=args.weight_decay,
report_to=args.report_to,
push_to_hub=False,
save_total_limit=5,
remove_unused_columns=False,
)


trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized_ds,
data_collator=data_collator,
tokenizer=tokenizer,
)


trainer.train()
trainer.save_model(args.output_dir)
wandb.finish()


if __name__ == '__main__':
main()