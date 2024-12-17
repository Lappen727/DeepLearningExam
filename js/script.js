let model;

async function loadModel() {
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log('Model loaded successfully');
        document.getElementById('classifyBtn').disabled = false;
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Failed to load model. Check console for details.');
    }
}

function preprocessText(text) {
    // Базовая предобработка текста
    text = text.toLowerCase();
    text = text.replace(/[^\w\s]/g, '');
    return text;
}

async function classifyText() {
    const inputText = document.getElementById('inputText').value;
    const resultDiv = document.getElementById('result');

    if (!inputText.trim()) {
        resultDiv.innerHTML = 'Please enter some text to classify';
        return;
    }

    try {
        // Базовая предобработка
        const processedText = preprocessText(inputText);

        // Здесь должна быть логика предобработки, 
        // аналогичная той, что использовалась при обучении модели
        // Вам может потребоваться воссоздать токенизацию и pad_sequences

        // Пример (это нужно адаптировать под вашу конкретную модель):
        const sequences = tokenizer.texts_to_sequences([processedText]);
        const paddedSequence = padSequences(sequences);

        // Преобразование в тензор
        const tensor = tf.tensor2d(paddedSequence);

        // Предсказание
        const prediction = model.predict(tensor);
        const score = prediction.dataSync()[0];

        // Интерпретация результата
        const isSpam = score > 0.5;
        resultDiv.innerHTML = isSpam 
            ? `<div class="spam">Spam (${(score * 100).toFixed(2)}% confidence)</div>` 
            : `<div class="not-spam">Not Spam (${((1 - score) * 100).toFixed(2)}% confidence)</div>`;
        
        // Освобождаем ресурсы
        tensor.dispose();
        prediction.dispose();

    } catch (error) {
        console.error('Classification error:', error);
        resultDiv.innerHTML = 'Error classifying text. Check console for details.';
    }
}

// Функции для предобработки (заглушки - их нужно будет реализовать)
const tokenizer = {
    texts_to_sequences: function(texts) {
        // Реализуйте логику преобразования текста в последовательность индексов
        return texts.map(text => 
            text.split(/\s+/).map(word => this.wordIndex[word] || 0)
        );
    },
    wordIndex: {} // Заполнить словарем слов из вашей модели
};

function padSequences(sequences, maxLen = 100) {
    return sequences.map(seq => {
        if (seq.length > maxLen) return seq.slice(0, maxLen);
        while (seq.length < maxLen) seq.push(0);
        return seq;
    });
}

// Инициализация при загрузке страницы
async function init() {
    document.getElementById('classifyBtn').disabled = true;
    await loadModel();
}

init();