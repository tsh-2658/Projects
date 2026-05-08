# 🎥 Minutes of Meetings

An AI-powered Streamlit application that extracts audio from meeting videos, transcribes speech, generates structured meeting summaries, and allows users to chat with the meeting transcript using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

### ✅ Video Upload & Processing
- Upload meeting recordings in:
  - `.mp4`
  - `.mkv`
  - `.mov`
  - `.avi`

### ✅ Audio Extraction
- Automatically extracts audio from uploaded videos using MoviePy.

### ✅ Speech-to-Text Transcription
- Uses Faster Whisper (`small` model) for accurate transcription.
- Optimized for better speed and CPU performance.

### ✅ AI Meeting Summarization
Generates structured meeting insights including:
- Overview
- Key Points
- Decisions
- Action Items
- Participants

### ✅ Interactive Chatbot
Ask questions about the uploaded meeting such as:
- "What decisions were made?"
- "Who is responsible for deployment?"
- "What are the pending tasks?"

### ✅ RAG-Based Retrieval
- Uses LangChain + FAISS vector database.
- Semantic search over transcript chunks.

### ✅ Smart Caching
- Hash-based duplicate detection.
- Prevents reprocessing of the same video.

### ✅ Transcript PDF Download
- Download the full transcript as a PDF file.

### ✅ Streaming Responses
- LLM responses stream token-by-token for better UX.

### ✅ Searchable Transcript
- Search keywords directly inside the transcript.

---

# 🏗️ Tech Stack

## Frontend
- Streamlit

## AI / LLM
- Google Gemini 2.5 Flash
- LangChain

## Speech Recognition
- Faster Whisper

## Vector Database
- FAISS

## Embeddings
- HuggingFace Sentence Transformers
  - `all-MiniLM-L6-v2`

## Video Processing
- MoviePy

## PDF Generation
- ReportLab

---

# 📂 Project Structure

```bash
project/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Setup Environment Variables

Create a `.env` file in the root directory.

```env
GOOGLE_API_KEY=your_google_api_key
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Application will start at:

```bash
http://localhost:8501
```

---

# 📦 Required Dependencies

Example `requirements.txt`:

```txt
streamlit
python-dotenv
moviepy
faster-whisper
reportlab
langchain
langchain-core
langchain-community
langchain-google-genai
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
nltk
```

---

# 🧠 Application Workflow

```text
User Uploads Video
        ↓
Audio Extraction
        ↓
Speech Transcription
        ↓
Transcript Chunking
        ↓
Embeddings Creation
        ↓
FAISS Vector Store
        ↓
Summary Generation
        ↓
Chat with Transcript
```

---

# 📋 Summary JSON Structure

The application generates structured summaries in the following format:

```json
{
  "overview": "",
  "key_points": [],
  "decisions": [],
  "action_items": [
    {
      "owner": "",
      "task": ""
    }
  ],
  "participants": []
}
```

---

# 💬 Example Questions

You can ask:

- What was discussed in the meeting?
- What are the action items?
- Who is responsible for each task?
- What decisions were made?
- Summarize the deployment discussion.
- Was there any deadline mentioned?

---

# 🔒 File Size Limit

Maximum upload size:

```text
500 MB
```

---

# ⚡ Performance Optimizations

- Cached LLM initialization
- Cached transcript embeddings
- Hash-based duplicate detection
- Sentence-aware chunking
- Streaming AI responses
- Partial state recovery

---

# 🛠️ Future Improvements

- Speaker diarization
- Multi-language transcription
- Cloud deployment
- Meeting sentiment analysis
- Action-item export to Jira/Trello
- Database storage
- User authentication

---

# 🐞 Troubleshooting

## GOOGLE_API_KEY not found

Ensure `.env` file exists and contains:

```env
GOOGLE_API_KEY=your_api_key
```

---

## ffmpeg Error

Install ffmpeg and add it to system PATH.

### Windows
Download from:
https://ffmpeg.org/download.html

### Ubuntu

```bash
sudo apt install ffmpeg
```

---

## FAISS Installation Issues

Use CPU version:

```bash
pip install faiss-cpu
```

---

# 📸 Application Screens

## Main Features
- Video Upload
- AI Summary
- Interactive Chat
- Transcript Viewer
- PDF Download

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed using:
- Streamlit
- LangChain
- Google Gemini
- Faster Whisper
- FAISS
- HuggingFace Embeddings

---