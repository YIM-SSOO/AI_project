import os
import pandas as pd
import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# 설정
device = torch.device("cpu")  # CPU 환경 설정
max_len = 32
batch_size = 32
num_epochs = 3
learning_rate = 3e-5

# 데이터셋 클래스 정의
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()  # 정수 레이블 사용
        return item

# 데이터 로드
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/validation.csv')

# 토크나이저와 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=6)

# 데이터셋 준비
train_encodings = tokenizer.batch_encode_plus(train_df['Text'].tolist(), truncation=True, padding=True, max_length=max_len)
val_encodings = tokenizer.batch_encode_plus(val_df['Text'].tolist(), truncation=True, padding=True, max_length=max_len)

train_dataset = EmotionDataset(train_encodings, train_df['Emotion'].tolist())
val_dataset = EmotionDataset(val_encodings, val_df['Emotion'].tolist())

# 메트릭 계산 함수
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 학습 설정
training_args = TrainingArguments(
    output_dir='./models/best_model',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 트레이너 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 모델 학습
trainer.train()

# 모델 및 토크나이저 저장
trainer.save_model('./models/best_model')
tokenizer.save_pretrained('./models/best_model')

print("★ 학 습 이 완 료 되 었 습 니 다 ★")
