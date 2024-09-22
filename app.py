from flask import Flask, render_template, request
from src.model import load_model, predict_emotions
import matplotlib
matplotlib.use('Agg')  # 'Agg' 백엔드를 사용해 GUI 없이 그래프를 생성
import matplotlib.pyplot as plt
import numpy as np
import logging

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 모델과 토크나이저 로드
model, tokenizer = load_model()

# 감정에 따른 이미지 파일 매핑
emotion_images = {
    "분노": "anger.png",
    "기쁨": "joy.png",
    "당황": "surprise.png",
    "불안": "fear.png",
    "슬픔": "sadness.png",
    "상처": "disgust.png"
}

# 감정에 따른 추천 멘트 매핑
emotion_messages = {
    "분노": "기분전환을 위해 산책을 나가보세요.",
    "기쁨": "기쁜 하루를 보내신 만큼, 내일도 멋진 하루 되시길 바랍니다!",
    "당황": "조금씩 마음을 가라앉히는 게 중요해요. <br>한 걸음씩 나아가면 됩니다.",
    "불안": "나 떨고 있니?🥶 <br>무엇이 자신을 불안하게 했는지 감정을 정리해보는건 어때요?",
    "슬픔": "자신을 위한 작은 행복을 찾아보세요. <br>좋아하는 음악을 들어보는건 어때요?",
    "상처": "힘든 하루였나보네요😥 <br>누군가와 대화하며 감정을 나누어 보세요."
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sentence = request.form.get("sentence")
        sentences = request.form.getlist("sentences")

        if sentences and isinstance(sentences[0], str):
            sentences = eval(sentences[0])  # ast.literal_eval 대신 eval 사용

        if sentence:
            sentences.append(sentence)

        logging.info(f"Input sentences: {sentences}")

        return render_template("index.html", sentences=sentences)

    return render_template("index.html", sentences=[])

@app.route("/analyze", methods=["POST"])
def analyze():
    sentences = request.form.getlist("sentences")

    if sentences and isinstance(sentences[0], str):
        sentences = eval(sentences[0])  # ast.literal_eval 대신 eval 사용

    # 문장을 하나로 통합하여 감정을 분석
    combined_sentence = " ".join(sentences)
    
    # 감정 예측
    probabilities = predict_emotions(combined_sentence, model, tokenizer)
    logging.info(f"Combined sentence: {combined_sentence}")
    logging.info(f"Predicted probabilities: {probabilities}")

    emotions = ["분노", "기쁨", "당황", "불안", "슬픔", "상처"]
    
    # Jinja2에서 min 함수 사용 문제 해결
    emotion_dict = {emotions[j]: min(probabilities[j] * 100, 100) for j in range(len(probabilities))}

    # 가장 높은 감정을 찾고 해당하는 캐릭터 이미지 선택
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    emotion_image = emotion_images[dominant_emotion]
    emotion_message = emotion_messages[dominant_emotion]

    logging.info(f"Emotion dict: {emotion_dict}")
    logging.info(f"Dominant emotion: {dominant_emotion}, Emotion image: {emotion_image}, Message: {emotion_message}")

    # 그래프 생성
    fig, ax = plt.subplots()
    ax.barh(list(emotion_dict.keys()), list(emotion_dict.values()), color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#FF6347'])
    ax.set_xlim(0, 100)  # X축 최대 범위를 100으로 설정
    ax.set_xlabel('Total Probability')
    ax.set_title("Overall Emotion Analysis")
    plt.tight_layout()

    # 그래프 파일 저장
    graph_filename = f"static/overall_emotion_graph.png"
    plt.savefig(graph_filename)
    plt.close()

    return render_template(
        "overall_results.html",
        sentences=sentences,
        emotion_dict=emotion_dict,
        graph_filename=graph_filename,
        emotion_image=emotion_image,
        emotion_message=emotion_message,
        max=max
    )

if __name__ == "__main__":
    app.run(debug=True)
