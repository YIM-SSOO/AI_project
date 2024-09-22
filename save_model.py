from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"  # t5-base, t5-large 등으로 변경 가능
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 로컬 디렉토리에 모델 저장
save_directory = "./models/best_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
