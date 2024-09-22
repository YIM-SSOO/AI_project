import torch
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

def load_model():
    model = BertForSequenceClassification.from_pretrained('./models/best_model', num_labels=6)
    tokenizer = KoBERTTokenizer.from_pretrained('./models/best_model')
    return model, tokenizer

def predict_emotions(sentence, model, tokenizer):
    inputs = tokenizer.encode_plus(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).flatten().tolist()

    return probabilities
