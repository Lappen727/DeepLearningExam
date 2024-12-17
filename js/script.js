let model;
let tokenizer;

// Функция для загрузки модели
async function loadModel() {
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log('Model loaded successfully');
        document.getElementById('classifyBtn').disabled = false;

        // Загружаем токенизатор, если он был сохранен
        const response = await fetch('model/tokenizer.json');
        const tokenizerData = await response.json();
        tokenizer = new Tokenizer(tokenizerData);
        
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Failed to load model. Check console for details.');
    }
}

// Функция для предобработки текста
function preprocessText(text) {
    text = text.toLowerCase(); // Преобразуем в нижний регистр
    text = text.replace(/[^\w\s]/g, ''); // Убираем все символы, кроме букв и пробелов
    return text;
}

// Функция для классификации текста
async function classifyText() {
    const inputText = document.getElementById('inputText').value;
    const resultDiv = document.getElementById('result');

    if (!inputText.trim()) {
        resultDiv.innerHTML = 'Please enter some text to classify';
        return;
    }

    try {
        const processedText = preprocessText(inputText);
        
        // Преобразуем текст в последовательность
        const sequences = tokenizer.textsToSequences([processedText]);
        const paddedSequence = padSequences(sequences);

        // Преобразуем в тензор
        const tensor = tf.tensor2d(paddedSequence);

        // Получаем предсказание
        const prediction = model.predict(tensor);
        const score = prediction.dataSync()[0]; // Получаем результат

        // Интерпретируем результат
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

// Функция для паддинга последовательности
function padSequences(sequences, maxLen = 100) {
    return sequences.map(seq => {
        if (seq.length > maxLen) return seq.slice(0, maxLen);
        while (seq.length < maxLen) seq.push(0);
        return seq;
    });
}

// Токенизатор для обработки текста
class Tokenizer {
    constructor(tokenizerData) {
        this.wordIndex = tokenizerData.wordIndex;
    }

    textsToSequences(texts) {
        return texts.map(text =>
            text.split(/\s+/).map(word => this.wordIndex[word] || 0)
        );
    }
}

// Инициализация модели и кнопки
async function init() {
    document.getElementById('classifyBtn').disabled = true;
    await loadModel();
}

init();