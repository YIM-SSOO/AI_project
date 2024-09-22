# Emotion Classification

This project is a multi-class emotion classification model using Hugging Face's Transformers library.

## Requirements

- Python 3.9
- PyTorch
- Hugging Face Transformers

## How to run

1. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Prepare your data in the `data/` directory.
3. Train the model:
    ```bash
    bash run.sh
    ```
4. Use the model for inference:
    ```bash
    python src/inference.py "Your text here"
    ```


    pip install transformers[torch] accelerate -U

# 시각화
pip install matplotlib

# Ko-Bert
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
pip install kobert-tokenizer

# Torch 
pip install torch==2.1.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html