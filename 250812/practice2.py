import numpy as np
import pandas as pd
import torch

from datasets import Value
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv("../../code/ch02_트랜스포머_기초_감성분석/data/review_data.csv", encoding='cp949')
print(df)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=0)

# ntrain = 150000
# df = pd.read_csv("ratings_test.tsv", sep='\t', quoting=3, encoding='utf-8-sig')
df = pd.read_csv("ratings_test.csv", encoding='utf-8-sig')
# df = pd.DataFrame(np.random.permutation(df))
# train_df, test_df = df[:ntrain], df[ntrain:]
# header = 'id document label'.split()
# train_df.to_csv('ratings_train.csv', sep='\t', index=False, header=header, encoding='utf-8-sig')
# train_df.to_csv('ratings_test.csv', sep='\t', index=False, header=header, encoding='utf-8-sig')
def safe_train_test_split_df(
    df: pd.DataFrame,
    label_col: str,
    text_col: str | None = None,
    test_size: float | int = 0.2,
    stratify: bool = True,
    random_state: int = 42,
    *,
    # === drop/정리 옵션들 ===
    drop_cols: list[str] | None = None,           # 지정 컬럼 드롭
    drop_cols_regex: str = r"^Unnamed:",          # 패턴 컬럼 드롭 (기본: Unnamed:*)
    coerce_label_numeric: bool = True,            # 라벨 숫자 변환(문자 있으면 NaN→드롭)
    allowed_labels: set | None = None,            # {0,1}처럼 허용 라벨만 남김
    dropna: bool = True,                          # label/text NaN 행 드롭
    drop_empty_text: bool = True,                 # 공백/빈 문자열 문서 드롭
    drop_duplicates_on: str | list[str] | None = None,  # 중복 드롭
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()

    # 1) 컬럼 드롭
    if drop_cols:
        df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore", inplace=True)
    if drop_cols_regex:
        df = df.loc[:, ~df.columns.str.contains(drop_cols_regex, regex=True)]

    # 2) 라벨 전처리
    if coerce_label_numeric:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    if allowed_labels is not None:
        df = df[df[label_col].isin(allowed_labels)]

    # 3) NaN/빈 텍스트 드롭
    subset = [label_col]
    if text_col:
        subset.append(text_col)
    if dropna:
        df = df.dropna(subset=subset)
    if drop_empty_text and text_col:
        df = df[df[text_col].astype(str).str.strip().ne("")]

    # 4) 중복 제거
    if drop_duplicates_on is not None:
        df = df.drop_duplicates(subset=drop_duplicates_on, keep="first")

    # 5) 크기/stratify 가능성 체크
    n = len(df)
    if n == 0:
        return None
    n_test = int(np.floor(n * test_size)) if isinstance(test_size, float) else int(test_size)
    if n_test <= 0 or n_test >= n:
        return None

    strat_series = None
    if stratify:
        y = df[label_col]
        # 각 클래스가 train/test에 최소 1개씩 들어갈 수 있는지
        cnt = y.value_counts()
        if isinstance(test_size, float):
            can = (cnt >= 2).all() and (cnt * test_size >= 1).all() and (cnt * (1 - test_size) >= 1).all()
        else:
            can = (cnt >= 2).all()
        strat_series = y if can else None

    try:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=strat_series, shuffle=True, random_state=random_state
        )
        return train_df, test_df
    except ValueError:
        return None

train_df, test_df = safe_train_test_split_df(
    df,
    label_col="labels",
    text_col="text",
    test_size=0.2,
    stratify=True,
    drop_cols_regex=r"^Unnamed:",     # Unnamed:* 자동 제거
    allowed_labels={0, 1},            # 라벨 0/1만 유지
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print('--------------')
print(train_dataset)
print(train_dataset[0])
print('--------------')

model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.problem_type = "single_label_classification"

train_dataset = train_dataset.rename_column("label", "labels") if "label" in train_dataset.column_names else train_dataset
test_dataset  = test_dataset.rename_column("label",  "labels") if "label"  in test_dataset.column_names  else test_dataset
train_dataset = train_dataset.cast_column("labels", Value("int64"))
test_dataset  = test_dataset.cast_column("labels",  Value("int64"))


def preprocess(data):
    print(data.keys())
    res = tokenizer(data['text'], truncation=True, padding='max_length', max_length=64)
    return res

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

cols = ["input_ids", "attention_mask", "labels"]
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in cols])
test_dataset  = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in cols])

print('--------------')
print(train_dataset)
print(train_dataset[0])
print('--------------')

train_args = TrainingArguments(
    output_dir='./saved_models/basic_sentiment2',
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
    logging_strategy='epoch',
    use_cpu=True
)

def accuracy_score(predict):
    preds = np.argmax(predict.predictions, axis=1)
    acc = (preds == predict.label_ids).mean()
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=accuracy_score
)

print(trainer.train())
trainer.save_model('./save_models/basic_sentiment2')
tokenizer.save_pretrained('.saved_models/basic_sentiment2')

print('--------------')
print(train_dataset)
print(train_dataset[0])
print('--------------')

text = "이 영화 너무 감동적이지 않았어! 최고야!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
sentiment = torch.argmax(logits).item()

print("감성 분석 결과:", "긍정" if sentiment == 1 else "부정")