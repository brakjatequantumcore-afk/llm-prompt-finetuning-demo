from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("imdb", split='train[:1%]')

def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

encoded = dataset.map(preprocess, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded,
    tokenizer=tokenizer
)

trainer.train()