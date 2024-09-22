import json
import pandas as pd
from sklearn.model_selection import train_test_split

# JSON 데이터 로드
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 데이터프레임으로 변환
def json_to_dataframe(data):
    conversations = data['Conversation']
    records = []

    for conv in conversations:
        record = {
            'Text': conv['Text'],
            'SpeakerNo': conv['SpeakerNo'],
            'EmotionCategory': conv['SpeakerEmotionCategory'],
            'EmotionLevel': conv['SpeakerEmotionLevel'],
            'EmotionTarget': conv['SpeakerEmotionTarget']
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

# 데이터 로드 및 전처리
def load_and_preprocess_data(json_path):
    data = load_json_data(json_path)
    df = json_to_dataframe(data)
    return df

# JSON 파일 경로 설정 및 데이터 로드
json_path = './data/conversation_data.json'
df = load_and_preprocess_data(json_path)

# 데이터 확인
print(df.head())

# 학습/검증 데이터 분할
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['EmotionCategory'])

# CSV로 저장
train_df.to_csv('./data/train.csv', index=False)
val_df.to_csv('./data/validation.csv', index=False)
