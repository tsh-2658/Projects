import streamlit as st
import os
import operator
import pandas as pd
from typing import Annotated, List, TypedDict, Literal

# LangChain / LangGraph Imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Workers (Reusing your logic)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
import shutil

load_dotenv()

st.set_page_config(page_title="Enterprise Data Intel", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    </style>
    """, unsafe_allow_html=True)

memory = MemorySaver()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    file_paths: List[str]
    next: str
    final_answer: str
    used_agents: Annotated[List[str], operator.add]

def get_latest_user_query(messages):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def has_successful_answer(text: str):
    """Checks if the LLM actually found an answer or just returned a failure message."""
    failures = [
        "not possible", "cannot determine", "does not contain", 
        "no data", "not found", "missing", "unable to", 
        "could not", "no information", "error"
    ]
    txt = text.lower()
    return not any(word in txt for word in failures)

@st.cache_resource
def load_models():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embedding_model = load_models()

system_prompt = """You are a routing supervisor managing three specialized workers:
1. csv -> Handles .csv and .xlsx files (Best for tabular/math data)
2. db -> Handles .db and .sqlite files (Best for structured relational data)
3. pdf_txt -> Handles .pdf and .txt files (Best for unstructured text)

Available files:
{file_list}

Used agents:
{used_agents}

Decision Logic:
- If a relevant worker has not been tried, pick them.
- If the answer is likely in a database, pick 'db' first.
- Return FINISH only if the answer is found or all relevant agents are exhausted.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{messages}"),
    ("system", "Who should act next? Respond with only the name.")
])

class RouteResponse(TypedDict):
    next: Literal["pdf_txt", "csv", "db", "FINISH"]


def supervisor_node(state: AgentState):
    # If we already have an answer, just finish
    if state.get("final_answer"):
        return {"next": "FINISH"}

    chain = prompt | llm.with_structured_output(RouteResponse)
    
    # Extract only the names of agents already used
    tried = state.get("used_agents", [])
    
    response = chain.invoke({
        "messages": state["messages"],
        "file_list": "\n".join(state["file_paths"]),
        "used_agents": tried
    })
    
    next_step = response["next"]
    if next_step in tried:
        return {"next": "FINISH"}
        
    return {"next": next_step}


def pdf_txt_node(state: AgentState):
    files = [f for f in state["file_paths"] if f.lower().endswith((".pdf", ".txt"))]
    if not files: return {"used_agents": ["pdf_txt"]}
    
    docs = []
    for f in files:
        loader = PyPDFLoader(f) if f.endswith(".pdf") else TextLoader(f)
        docs.extend(loader.load())
    
    splits = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)
    vector = FAISS.from_documents(splits, embedding_model)
    query = get_latest_user_query(state["messages"])
    context = "\n".join([d.page_content for d in vector.similarity_search(query, k=3)])
    
    ans = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    res = {"messages": [AIMessage(content=ans.content)], "used_agents": ["pdf_txt"]}
    if has_successful_answer(ans.content):
        res["final_answer"] = ans.content
    return res

def csv_node(state: AgentState):
    files = [f for f in state["file_paths"] if f.lower().endswith((".csv", ".xlsx"))]
    if not files: return {"used_agents": ["csv"]}
    
    df = pd.concat([pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f) for f in files])
    agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True, verbose=True,handle_parsing_errors=True)
    
    # query = get_latest_user_query(state["messages"])
    # ans = agent.invoke({"input": query})

    query = get_latest_user_query(state["messages"])
    instruction = f"""
    Search for the answer to: {query}. 
    If you cannot find an exact numerical match, provide the closest values 
    available or the maximum/minimum values in that column to help the user.
    """
    
    ans = agent.invoke({"input": instruction})

    res = {"messages": [AIMessage(content=ans["output"])], "used_agents": ["csv"]}
    if has_successful_answer(ans["output"]):
        res["final_answer"] = ans["output"]
    return res

def db_node(state: AgentState):
    files = [f for f in state["file_paths"] if f.lower().endswith((".db", ".sqlite"))]
    if not files: return {"used_agents": ["db"]}
    
    all_res = []
    query = get_latest_user_query(state["messages"])
    for f in files:
        db_engine = SQLDatabase.from_uri(f"sqlite:///{f}")
        agent = create_sql_agent(llm, db=db_engine, verbose=False)
        ans = agent.invoke({"input": query})
        all_res.append(ans["output"])
    
    combined = "\n".join(all_res)
    res = {"messages": [AIMessage(content=combined)], "used_agents": ["db"]}
    if has_successful_answer(combined):
        res["final_answer"] = combined
    return res

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("pdf_txt", pdf_txt_node)
builder.add_node("csv", csv_node)
builder.add_node("db", db_node)

builder.set_entry_point("supervisor")

def route_next(state: AgentState):
    return state["next"]

builder.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "pdf_txt": "pdf_txt",
        "csv": "csv", 
        "db": "db", 
        "FINISH": END
    }
)
builder.add_edge("pdf_txt", "supervisor")
builder.add_edge("csv", "supervisor")
builder.add_edge("db", "supervisor")

app = builder.compile(checkpointer=memory)

if "file_registry" not in st.session_state:
    st.session_state.file_registry = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("📊 Control Center")
    
    st.header("📁 Upload Files")
    new_uploads = st.file_uploader(
        "Drop PDFs, CSVs, or DB files", 
        accept_multiple_files=True, 
        type=['pdf', 'txt', 'csv', 'xlsx', 'db', 'sqlite']
    )
    
    if new_uploads:
        if not os.path.exists("temp_uploads"):
            os.makedirs("temp_uploads")
        for f in new_uploads:
            if f.name not in st.session_state.file_registry:
                f_path = os.path.join("temp_uploads", f.name)
                with open(f_path, "wb") as buffer:
                    buffer.write(f.getbuffer())
                st.session_state.file_registry[f.name] = {"path": f_path, "active": True}
        st.success(f"Successfully loaded {len(new_uploads)} file(s)!")

    st.divider()
    
    st.header("🛠️ Manage Files")
    if not st.session_state.file_registry:
        st.info("No files uploaded yet.")
    else:
        files_to_remove = []
        for fname, info in st.session_state.file_registry.items():
            col_active, col_name, col_del = st.columns([0.15, 0.65, 0.2])

            is_active = col_active.checkbox("", value=info["active"], key=f"active_{fname}", label_visibility="collapsed")
            st.session_state.file_registry[fname]["active"] = is_active

            col_name.text(fname)

            if col_del.button("🗑️", key=f"del_{fname}"):
                files_to_remove.append(fname)

        for fname in files_to_remove:
            info = st.session_state.file_registry.pop(fname)
            if os.path.exists(info["path"]):
                os.remove(info["path"])
            st.rerun()

    st.divider()

    st.subheader("🧹 System Actions")
    
    if st.button("Clear Chat History", use_container_width=True, help="Deletes the conversation but keeps your files."):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Delete All Files", use_container_width=True, type="secondary", help="Physically deletes all uploaded files."):
        if os.path.exists("temp_uploads"):
            shutil.rmtree("temp_uploads")
            os.makedirs("temp_uploads") # Recreate empty directory for future use
        st.session_state.file_registry = {}
        st.warning("All files have been deleted.")
        st.rerun()

st.title("🤖 Multimodal RAG Assistant")
st.caption("Query across structured databases, tabular spreadsheets, and unstructured documents.")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_query := st.chat_input("Ask a question about your active data..."):
    active_paths = [v["path"] for v in st.session_state.file_registry.values() if v["active"]]
    
    if not active_paths:
        st.error("Error: No active files selected. Please upload files and ensure the 'Active' checkbox is checked in the sidebar.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing across specialized agents..."):
                try:
                    config = {"configurable": {"thread_id": "streamlit_user_session"}}
                    
                    result = app.invoke({
                        "messages": [HumanMessage(content=user_query)],
                        "file_paths": active_paths,
                        "used_agents": [],
                        "final_answer": ""
                    }, config=config)
                    
                    final_txt = result.get("final_answer") or "I searched the active files but couldn't find a conclusive answer."
                    st.markdown(final_txt)
                    st.session_state.chat_history.append(AIMessage(content=final_txt))
                
                except Exception as e:
                    error_msg = f"An error occurred during analysis: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))

st.divider()

