import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 감정 라벨 매핑
emotion_mapping = {
    'E18': 0,  # 분노
    'E01': 1,  # 기쁨
    'E12': 2,  # 당황
    'E11': 3,  # 불안
    'E19': 4,  # 슬픔
    'E09': 5   # 상처
}

# JSON 데이터 로드 및 전처리
def load_and_preprocess_data(directory):
    data = {
        'Text': [],
        'Emotion': []
    }

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for entry in json_data:
                    emotion_type = entry['profile']['emotion']['type']
                    if emotion_type in emotion_mapping:
                        label = emotion_mapping[emotion_type]
                        talk_content = entry['talk']['content']
                        for key in ['HS01', 'HS02', 'HS03']:
                            if talk_content[key]:
                                data['Text'].append(talk_content[key])
                                data['Emotion'].append(label)

    return pd.DataFrame(data)

# 데이터 로드 및 전처리
train_df = load_and_preprocess_data('./data/train/labeled/')
val_df = load_and_preprocess_data('./data/validation/labeled/')

# 데이터 확인
print(train_df.head())
print(val_df.head())

# 학습/검증 데이터 분할
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Emotion'])

# CSV로 저장
train_df.to_csv('./data/train.csv', index=False)
val_df.to_csv('./data/validation.csv', index=False)
