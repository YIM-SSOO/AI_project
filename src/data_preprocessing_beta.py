# import os
# import json
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 감정 라벨 매핑
# emotion_mapping = {
#     'E18': 0,  # 분노
#     'E01': 1,  # 기쁨
#     'E12': 2,  # 당황
#     'E11': 3,  # 불안
#     'E19': 4,  # 슬픔
#     'E09': 5   # 상처
# }

# # JSON 데이터 로드 및 전처리 함수
# def load_json_data(directory):
#     data = {
#         'Text': [],     # 텍스트 데이터를 저장할 리스트
#         'Emotion': []   # 감정 라벨을 저장할 리스트
#     }

#     # 주어진 디렉토리에서 JSON 파일을 순회
#     for filename in os.listdir(directory):
#         if filename.endswith('.json'):  # 확장자가 .json인 파일만 처리
#             with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
#                 json_data = json.load(f)  # JSON 파일을 로드
#                 for entry in json_data:   # JSON의 각 엔트리(데이터) 처리
#                     emotion_type = entry['profile']['emotion']['type']  # 감정 유형 추출
#                     if emotion_type in emotion_mapping:  # 감정 유형이 매핑된 경우에만 처리
#                         label = emotion_mapping[emotion_type]  # 해당 감정 유형의 라벨을 가져옴
#                         talk_content = entry['talk']['content']  # 대화 내용을 가져옴
#                         for key in ['HS01', 'HS02', 'HS03']:  # 각 대화 내용 필드를 순회
#                             if talk_content[key]:  # 텍스트가 비어 있지 않으면
#                                 data['Text'].append(talk_content[key])  # 텍스트 추가
#                                 data['Emotion'].append(label)  # 라벨 추가

#     return pd.DataFrame(data)  # DataFrame 형태로 반환

# # 엑셀 데이터 로드 및 전처리 함수
# def load_excel_data(file_path):
#     df = pd.read_excel(file_path)  # 엑셀 파일 로드
    
#     # '감정_대분류'를 사용하여 감정 라벨을 매핑
#     # df['Emotion'] = df['감정_대분류'].map(emotion_mapping)  # 감정 라벨 매핑
    
#     # 텍스트 데이터를 '사람문장1', '사람문장2', '사람문장3'에서 추출하여 하나의 리스트로 병합
#     data = {
#         'Text': [],     # 텍스트 데이터를 저장할 리스트
#         'Emotion': []   # 감정 라벨을 저장할 리스트
#     }
    
#     # 각 행을 순회하며 텍스트 데이터를 수집
#     for index, row in df.iterrows():
#         for col in ['사람문장1', '사람문장2', '사람문장3']:  # 각 텍스트 열을 순회
#             if pd.notna(row[col]):  # 텍스트가 존재하는 경우만 추가
#                 data['Text'].append(row[col])  # 텍스트 추가
#                 data['Emotion'].append(row['Emotion'])  # 라벨 추가
    
#     # NaN 값이 있는 행을 제거
#     final_df = pd.DataFrame(data).dropna()  # DataFrame으로 변환 후 NaN 값 제거
#     return final_df

# # JSON 및 엑셀 데이터 로드
# json_train_df = load_json_data('./data/train/labeled/')  # 학습용 JSON 데이터 로드
# json_val_df = load_json_data('./data/validation/labeled/')  # 검증용 JSON 데이터 로드
# excel_train_df = load_excel_data('./data/training_data.xlsx')  # 학습용 엑셀 데이터 로드
# excel_val_df = load_excel_data('./data/validation_data.xlsx')  # 검증용 엑셀 데이터 로드

# # JSON 데이터와 엑셀 데이터를 병합
# train_df = pd.concat([json_train_df, excel_train_df], ignore_index=True)  # 학습 데이터 병합
# val_df = pd.concat([json_val_df, excel_val_df], ignore_index=True)  # 검증 데이터 병합

# # NaN 값이 있는 행을 제거
# train_df = train_df.dropna()  # 학습 데이터에서 NaN 값 제거
# val_df = val_df.dropna()  # 검증 데이터에서 NaN 값 제거

# # 학습/검증 데이터 분할
# train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Emotion'])  # 학습 데이터를 학습/검증 세트로 분할 (stratify로 라벨 비율 유지)

# # CSV로 저장
# train_df.to_csv('./data/train.csv', index=False)  # 학습 데이터를 CSV 파일로 저장
# val_df.to_csv('./data/validation.csv', index=False)  # 검증 데이터를 CSV 파일로 저장

# # 데이터 확인
# print("Training data preview:")  # 학습 데이터 미리보기 출력
# print(train_df.head())  # 학습 데이터의 첫 몇 줄 출력
# print("Validation data preview:")  # 검증 데이터 미리보기 출력
# print(val_df.head())  # 검증 데이터의 첫 몇 줄 출력

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

# JSON 데이터 로드 및 전처리 함수
def load_json_data(directory):
    data = {
        'Text': [],     # 텍스트 데이터를 저장할 리스트
        'Emotion': []   # 감정 라벨을 저장할 리스트
    }

    for filename in os.listdir(directory):
        if filename.endswith('.json'):  # 확장자가 .json인 파일만 처리
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                json_data = json.load(f)  # JSON 파일을 로드
                for entry in json_data:   # JSON의 각 엔트리(데이터) 처리
                    emotion_type = entry['profile']['emotion']['type']  # 감정 유형 추출
                    if emotion_type in emotion_mapping:  # 감정 유형이 매핑된 경우에만 처리
                        label = emotion_mapping[emotion_type]  # 해당 감정 유형의 라벨을 가져옴
                        talk_content = entry['talk']['content']  # 대화 내용을 가져옴
                        for key in ['HS01', 'HS02', 'HS03']:  # 각 대화 내용 필드를 순회
                            if talk_content[key]:  # 텍스트가 비어 있지 않으면
                                data['Text'].append(talk_content[key])  # 텍스트 추가
                                data['Emotion'].append(label)  # 라벨 추가

    return pd.DataFrame(data)  # DataFrame 형태로 반환

# 엑셀 데이터 로드 및 전처리 함수
def load_excel_data(file_path):
    df = pd.read_excel(file_path)  # 엑셀 파일 로드
    
    # '감정_대분류'를 사용하여 감정 라벨을 매핑
    df['Emotion'] = df['감정_대분류'].map(emotion_mapping)  # 감정 라벨 매핑
    
    # 텍스트 데이터를 '사람문장1', '사람문장2', '사람문장3'에서 추출하여 하나의 리스트로 병합
    data = {
        'Text': [],     # 텍스트 데이터를 저장할 리스트
        'Emotion': []   # 감정 라벨을 저장할 리스트
    }
    
    for index, row in df.iterrows():
        for col in ['사람문장1', '사람문장2', '사람문장3']:  # 각 텍스트 열을 순회
            if pd.notna(row[col]):  # 텍스트가 존재하는 경우만 추가
                data['Text'].append(row[col])  # 텍스트 추가
                data['Emotion'].append(row['Emotion'])  # 라벨 추가
    
    final_df = pd.DataFrame(data).dropna()  # DataFrame으로 변환 후 NaN 값 제거
    return final_df

# JSON 및 엑셀 데이터 로드
json_train_df = load_json_data('./data/train/labeled/')  # 학습용 JSON 데이터 로드
json_val_df = load_json_data('./data/validation/labeled/')  # 검증용 JSON 데이터 로드
excel_train_df = load_excel_data('./data/training_data.xlsx')  # 학습용 엑셀 데이터 로드
excel_val_df = load_excel_data('./data/validation_data.xlsx')  # 검증용 엑셀 데이터 로드

# JSON 데이터와 엑셀 데이터를 병합
train_df = pd.concat([json_train_df, excel_train_df], ignore_index=True)  # 학습 데이터 병합
val_df = pd.concat([json_val_df, excel_val_df], ignore_index=True)  # 검증 데이터 병합

# NaN 값이 있는 행을 제거
train_df = train_df.dropna()  # 학습 데이터에서 NaN 값 제거
val_df = val_df.dropna()  # 검증 데이터에서 NaN 값 제거

# 학습/검증 데이터 분할
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Emotion'])  # 학습 데이터를 학습/검증 세트로 분할 (stratify로 라벨 비율 유지)

# CSV로 저장
train_df.to_csv('./data/train.csv', index=False)  # 학습 데이터를 CSV 파일로 저장
val_df.to_csv('./data/validation.csv', index=False)  # 검증 데이터를 CSV 파일로 저장

# 데이터 확인
print("Training data preview:")  # 학습 데이터 미리보기 출력
print(train_df.head())  # 학습 데이터의 첫 몇 줄 출력
print("Validation data preview:")  # 검증 데이터 미리보기 출력
print(val_df.head())  # 검증 데이터의 첫 몇 줄 출력

