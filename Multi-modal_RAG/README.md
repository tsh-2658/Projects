# Generate README.md
readme_content = """# 🤖 Multimodal RAG Assistant

An advanced, multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Google Gemini**, and **Streamlit**. This application routes user queries to specialized agents based on the uploaded file formats, allowing for seamless querying across structured databases, tabular spreadsheets, and unstructured documents.

## 🌟 Key Features

* **Intelligent Routing**: A Supervisor Agent (powered by Gemini 2.5 Flash) analyzes the query and file context to decide which specialized worker to invoke.
* **Specialized Workers**:
    * **PDF/TXT Agent**: Uses FAISS vector stores and HuggingFace embeddings for semantic search.
    * **CSV/Excel Agent**: Utilizes a Pandas DataFrame agent for math and tabular data analysis.
    * **SQL Database Agent**: Queries SQLite/DB files using natural language.
* **Stateful Orchestration**: Built with LangGraph to manage complex workflows, including retry logic and multi-agent coordination.
* **Dynamic UI**: A Streamlit frontend featuring:
    * Side-bar file management (Upload/Toggle/Delete).
    * Persistent chat history.
    * Clean, enterprise-level styling.

## 🛠️ Technical Stack

* **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
* **LLM**: Google Gemini 2.5 Flash
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Store**: FAISS
* **UI**: Streamlit
* **Data Analysis**: Pandas, SQLAlchemy

## 🚀 Getting Started

1. Prerequisites
    Ensure you have Python 3.9+ installed and a Google AI Studio API Key.

2. Installation
    Clone the repository and install the dependencies:
    pip install -r requirements.txt

3. Environment Setup

    Create a .env file in the root directory and add your Google API Key:
    GOOGLE_API_KEY_1=your_api_key_here

4.Run the application

    streamlit run app.py