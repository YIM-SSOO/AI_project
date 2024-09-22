import os
import json
import pandas as pd
import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(directory):
    data = {
        'Text': [],
        'Emotion': []
    }
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for conversation in json_data['Conversation']:
                    data['Text'].append(conversation['Text'])
                    emotion_category = conversation['SpeakerEmotionCategory']
                    if emotion_category == "긍정":
                        label = 1
                    elif emotion_category == "부정":
                        label = 0
                    else:
                        label = 2
                    data['Emotion'].append(label)
    
    return pd.DataFrame(data)

# Dataset 클래스 정의
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# 데이터 로드 및 전처리
train_df = load_and_preprocess_data('./data/train/labeled/')
val_df = load_and_preprocess_data('./data/validation/labeled/')

# 토크나이저와 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=3)

# 데이터셋 준비
train_encodings = tokenizer.batch_encode_plus(train_df['Text'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer.batch_encode_plus(val_df['Text'].tolist(), truncation=True, padding=True)

train_dataset = EmotionDataset(train_encodings, train_df['Emotion'].tolist())
val_dataset = EmotionDataset(val_encodings, val_df['Emotion'].tolist())

# 학습 설정
training_args = TrainingArguments(
    output_dir='./models/best_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# 트레이너 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 모델 학습
trainer.train()

# 모델 및 토크나이저 저장
trainer.save_model('./models/best_model')
tokenizer.save_pretrained('./models/best_model')
