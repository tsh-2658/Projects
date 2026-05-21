# ✍️ AI Blog Copilot: Human-in-the-Loop Agentic Workflow

An enterprise-grade multi-agent content generation platform built with **LangGraph**, **LangChain**, and **Streamlit**. This application orchestrates a specialized content creation loop consisting of a Professional Writer Agent and an Editorial Critic Agent, fully managed by state interrupts for real-time human verification, adjustment, or structural pivots.

---

## 🏗️ Architecture Design & State Machine

The workflow transitions through three main phases using a compiled state graph, persisting execution state across sessions via an in-memory checkpointer (`MemorySaver`).


1. **Writer Agent (`ChatNVIDIA` / Llama 3.2 3B):** Dynamically generates or refines markdown-formatted blog content by evaluating the entire conversational state history, structured critiques, and manual feedback logs.
2. **Reviewer Agent (`ChatGoogleGenerativeAI` / Gemini 2.5 Flash):** Acts as an objective editorial critic utilizing structured outputs (`with_structured_output`) via a strict Pydantic parsing schema to evaluate layout formatting, actionable content delivery, and audience engagement points.
3. **Human-in-the-Loop Interruption:** Leverages LangGraph’s native `interrupt()` and `Command(resume=...)` signatures to cleanly break execution thread flow. This decouples logic from UI renders, letting the dashboard inspect intermediate artifacts and supply structural course-corrections or topic handshakes.

---

## 🛠️ Features

- **Objective Point-Based Rubric:** The Reviewer node uses an exact grading breakdown criteria across Structure (3 pts), Value (4 pts), and Engagement (3 pts) to ensure non-inflated, stable quality analytics.
- **Dynamic Topic Pivoting:** Incorporates overwrite heuristics inside the writer agent's prompt context. If a user alters the target topic mid-pipeline inside the critique interface, older draft directions are cleanly suppressed to build a fresh core post from scratch.
- **Streamlit Session Persistence:** Seamlessly handles Streamlit's full script top-to-bottom re-runs by binding UI actions directly to thread snapshots using `graph.get_state(config)`.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your workspace.

### 2. Clone & Environment Configuration
Set up your local directory and initialize your environment variable profile:
In .env file

# Initialize and install requirements
pip install -r requirements.txt

### 3. Execution
Launch the interactive Streamlit dashboard pipeline via your terminal:
streamlit run app.py
