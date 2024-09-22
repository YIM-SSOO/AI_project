import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 32
batch_size = 32  # 배치 크기를 64로 조정하여 메모리 효율을 높임
num_epochs = 3
learning_rate = 3e-5 # 학습률을 높여서 초기 학습 속도를 향상시킴
fp16 = True if torch.cuda.is_available() else False

# 데이터셋 클래스 정의
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# 데이터 로드 및 크기 줄이기 (훈련 데이터의 50%만 사용하여 학습 시간 단축)
train_df = pd.read_csv('./data/train.csv').sample(frac=0.5)
val_df = pd.read_csv('./data/validation.csv')

# KoBERT 토크나이저 로드 (BPE-dropout 적용)
sp_model_kwargs = {'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True}
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs=sp_model_kwargs)

# 모델 로드
model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=6)
model.to(device)

# 데이터셋 준비
train_encodings = tokenizer(train_df['Text'].tolist(), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
val_encodings = tokenizer(val_df['Text'].tolist(), truncation=True, padding=True, max_length=max_len, return_tensors="pt")

train_dataset = EmotionDataset(train_encodings, train_df['Emotion'].tolist())
val_dataset = EmotionDataset(val_encodings, val_df['Emotion'].tolist())

# DataLoader 사용하여 데이터 로딩 속도 개선
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
    warmup_steps=50,  # Warmup 스텝을 줄여서 초기 학습 속도를 높임
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    fp16=fp16  # GPU에서만 Mixed Precision Training 사용
)

# 트레이너 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 모델 학습
trainer.train()

# 모델 및 토크나이저 저장
trainer.save_model('./models/best_model')
tokenizer.save_pretrained('./models/best_model')

print("★ 학 습 이 완 료 되 었 습 니 다 ★")
