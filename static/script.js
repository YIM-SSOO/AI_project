document.getElementById('analyze_button').addEventListener('click', function() {
    analyzeEmotion();
});

document.getElementById('user_input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();  // 엔터 입력에 의해 새 줄이 추가되는 것을 방지
        analyzeEmotion();  // 엔터 키가 눌렸을 때 분석 실행
    }
});

function analyzeEmotion() {
    const userInput = document.getElementById('user_input').value;

    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `user_input=${encodeURIComponent(userInput)}`,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').style.display = 'block';
        const emotionsDiv = document.getElementById('emotions');
        emotionsDiv.innerHTML = ''; // 기존 결과 지우기

        for (const [emotion, value] of Object.entries(data)) {
            const bar = document.createElement('div');
            bar.classList.add('bar');

            const barFill = document.createElement('div');
            barFill.classList.add('bar-fill');
            barFill.style.width = `${value * 100}%`;
            barFill.style.backgroundColor = getEmotionColor(emotion);

            const label = document.createElement('div');
            label.classList.add('bar-label');
            label.textContent = `${emotion}: ${Math.round(value * 100)}%`;

            bar.appendChild(barFill);
            bar.appendChild(label);
            emotionsDiv.appendChild(bar);
        }
    })
    .catch(error => console.error('Error:', error));
}

function getEmotionColor(emotion) {
    const colors = {
        '기뻐요': '#77dd77',
        '슬퍼요': '#aec6cf',
        '우울해요': '#b39ddb',
        '화가 나요': '#ff6961',
    };
    return colors[emotion] || '#d3d3d3';
}
