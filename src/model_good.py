import torch
import numpy as np
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

def load_model():
    model = BertForSequenceClassification.from_pretrained('./models/best_model', num_labels=6)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    return model, tokenizer

def predict_emotions(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
    
    if len(probabilities) != 6:
        print(f"Warning: Expected 6 probabilities, but got {len(probabilities)}. Padding with 0s.")
        padded_probabilities = np.zeros(6)
        padded_probabilities[:len(probabilities)] = probabilities
        return padded_probabilities

    return probabilities
