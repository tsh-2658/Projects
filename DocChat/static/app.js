const uploadForm = document.getElementById('upload-form');
const chatForm = document.getElementById('chat-form');
const uploadResult = document.getElementById('upload-result');
const chatWindow = document.getElementById('chat-window');
const chatStatus = document.getElementById('chat-status');
const voiceStatus = document.getElementById('voice-status');
const questionInput = document.getElementById('question');
const fileInput = document.getElementById('file-input');
const voiceButton = document.getElementById('voice-button');

// Voice Recognition Setup
const recognitionSupported = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
if (!recognitionSupported) {
  voiceButton.disabled = true;
  voiceButton.title = "Voice not supported in this browser";
}

function setStatus(element, text, success = true) {
  element.style.display = 'block';
  element.textContent = text;
  element.style.backgroundColor = success ? '#eef2ff' : '#fee2e2';
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || 'Server error');
  }
  return data;
}

// Upload Logic
uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  uploadResult.style.display = 'none';
  
  const files = Array.from(fileInput.files);
  if (!files.length) return;

  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  try {
    setStatus(uploadResult, 'Uploading and indexing...');
    const data = await fetchJson('/upload', {
      method: 'POST',
      body: formData,
    });
    setStatus(uploadResult, `Success: Indexed ${files.length} document(s).`);
    fileInput.value = '';
    appendChatMessage('system', `New files uploaded: ${files.map(f => f.name).join(', ')}`);
  } catch (err) {
    setStatus(uploadResult, err.message, false);
  }
});

// Chat Logic
chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendChatMessage('user', question);
  questionInput.value = '';
  chatStatus.style.display = 'none';

  try {
    const data = await fetchJson('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    appendChatMessage('assistant', data.answer);
  } catch (err) {
    setStatus(chatStatus, err.message, false);
  }
});

// Voice Input Logic
voiceButton.addEventListener('click', () => {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new Recognition();
  recognition.lang = 'en-US';

  recognition.onstart = () => setStatus(voiceStatus, 'Listening...');
  
  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    questionInput.value = transcript;
    setStatus(voiceStatus, `Captured: ${transcript}`);
    chatForm.dispatchEvent(new Event('submit'));
  };

  recognition.onerror = (err) => setStatus(voiceStatus, 'Voice error: ' + err.error, false);
  recognition.onend = () => setTimeout(() => voiceStatus.style.display = 'none', 3000);
  
  recognition.start();
});

function appendChatMessage(sender, text) {
  const message = document.createElement('div');
  message.className = 'chat-item';
  const title = document.createElement('strong');
  title.textContent = sender === 'user' ? 'You' : sender === 'assistant' ? 'Assistant' : 'System';
  const content = document.createElement('p');
  content.textContent = text;
  message.append(title, content);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}