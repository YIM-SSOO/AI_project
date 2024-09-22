# import torch
# from transformers import BertForSequenceClassification
# from kobert_tokenizer import KoBERTTokenizer

# def load_model():
#     model = BertForSequenceClassification.from_pretrained('./models/best_model')
#     tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
#     return model, tokenizer

# def predict_emotions(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     outputs = model(**inputs)
#     probabilities = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
#     return probabilities

# import torch
# from kobert_tokenizer import KoBERTTokenizer
# from transformers import BertForSequenceClassification

# def load_model():
#     model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=4)
#     tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
#     return model, tokenizer

# def predict_emotions(sentence, model, tokenizer):
#     inputs = tokenizer.encode_plus(sentence, return_tensors='pt', truncation=True, padding=True)
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     return probs.detach().numpy()[0]  # 모델의 출력을 numpy 배열로 변환하여 반환

import torch
import numpy as np  # numpy 패키지 import 추가
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

def load_model():
    model = BertForSequenceClassification.from_pretrained('./models/best_model')
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    return model, tokenizer

def predict_emotions(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
    
    # Ensure that probabilities array has exactly 4 elements
    # If not, pad the result with 0s
    if len(probabilities) != 4:
        print(f"Warning: Expected 4 probabilities, but got {len(probabilities)}. Padding with 0s.")
        padded_probabilities = np.zeros(4)
        padded_probabilities[:len(probabilities)] = probabilities
        return padded_probabilities

    return probabilities
