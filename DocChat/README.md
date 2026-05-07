# 🚀 GenAI RAG & Multi-Doc Summarizer

A robust document intelligence portal built with **FastAPI**, **LangChain**, and **Google Gemini**. This application seamlessly handles standard Q&A for specific details and heavy-duty summarization for batches of 100+ documents.

---

## 🌟 Key Features

### 1. Zero-Configuration Setup
* **Auto-Managed Directory:** The application automatically detects and creates the `/uploads` and `/vectorstore` directories on startup.
* **Self-Initializing Database:** Automatically generates the SQLite `app.db` and log files if they are missing.

### 2. Dual-Mode Intelligence (Unified Chat)
The system uses natural language intent detection to switch between two modes in the same chat box:
* **Specific RAG Search:** For questions like *"What was the total cost in the March invoice?"*, the system retrieves only the relevant snippets from your documents.
* **Global Map-Reduce Summarization:** For requests like *"Summarize all documents"*, the system triggers a Map-Reduce chain. This ensures that even if you have 100+ files, the AI reads everything without hitting context window limits.

### 3. Smart Document Handling
* **Multi-Format Support:** Natively supports `.pdf`, `.txt`, `.md`, and `.csv`.
* **Large Scale Processing:** Uses high-performance FAISS indexing for retrieval and recursive character splitting for high-quality summaries.

### 4. Audit & History Tracking
* **Upload Logs:** Tracks every file upload with timestamps and usernames in an SQLite database.
* **Queryable Logs:** You can ask the chatbot via chat or audio imput about your history or anything, e.g., *"What files did I upload in the last 7 days?"*

---

## 🛠️ Technical Stack
* **LLM:** Google Gemini 2.0 Flash (Optimized for speed and context)
* **Framework:** FastAPI
* **Vector Store:** FAISS
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Database:** SQLAlchemy / SQLite

---

## 📦 Setup & Installation

1.  **Clone the repository**
2.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn langchain langchain-google-genai langchain-community faiss-cpu sqlalchemy pypdf sentence-transformers python-dotenv
    ```
3.  **Environment Variables:** Create a `.env` file and add your Google API key:
    ```env
    GOOGLE_API_KEY_5=your_gemini_api_key_here
    ```
4.  **Run the application:**
    ```bash
    uvicorn app:app --reload
    ```

## 📝 Usage
* **Upload:** Use the UI or POST to `/upload` to add files.
* **Search:** Ask specific questions like "What is the total revenue in the Q3 report?"
* **Summarize:** Ask "Summarize everything I have uploaded."
* **Logs:** Ask "Show me my recent uploads."
