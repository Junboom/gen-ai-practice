import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

df = pd.read_csv('../../code/ch04_LoRA/data/review_data.csv', encoding='cp949')
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=0)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = 'beomi/kcbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none',
    target_modules=['query', 'key']
)
model = get_peft_model(base_model, peft_config)

def preprocess(data):
    res = tokenizer(data['text'], truncation=True, padding='max_length', max_length=64)
    return res

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir='./saved_models/peft_lora_sentiment',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    label_names=['labels'],
    use_cpu=True
)

def compute_metrics(predict):
    preds = np.argmax(predict.predictions, axis=1)
    acc = (preds == predict.label_ids).mean()
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

test_texts = ['이 제품 너무 좋아요!', '별로에요. 추천 안함.']
inputs = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True, max_length=64)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
preds = torch.argmax(outputs.logits, dim=1)
print('예측결과: ', preds.tolist())