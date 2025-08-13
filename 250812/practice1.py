import numpy as np
from transformers.trainer_utils import EvalPrediction

hyper_parameter = {
    'a': 3,
    'b': 'birthday',
    'c': [1, 2, 3, 4, 5]
}

def explain1(a, b, c):
    return a, b, c

print(explain1(**hyper_parameter))

def explain2(param):
    return {**param, 'd': 30}

print(explain2(hyper_parameter))

def accuracy_score(predict, axis):
    preds = np.argmax(predict.predictions, axis=axis)
    acc = (preds == predict.label_ids).mean()
    return {'accuracy': acc}

y_hat = np.array(
    [
        [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9]
    ]
)
y_true = np.array([1, 0, 1])

pred = EvalPrediction(predictions=y_hat, label_ids=y_true)
print(pred.predictions)
print('--------------')
print(pred.label_ids)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = 'beomi/kcbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

text = '이 영화 너무 감동적이었어! 최고야!'
inputs = tokenizer(text, return_tensors='pt')

print('--------------')
print(inputs.input_ids)
print('--------------')
print(inputs.token_type_ids)
print('--------------')
print(inputs.attention_mask)

with torch.no_grad():
    outputs = model(**inputs)

print('--------------')
print(outputs.logits)

logits = outputs.logits
sentiment = torch.argmax(logits).item()

print('감성 분석 결과: ', '긍정' if sentiment == 1 else '부정')