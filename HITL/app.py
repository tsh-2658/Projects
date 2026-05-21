import os
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

load_dotenv()

@st.cache_resource
def get_compiled_graph():
    writer_llm = ChatNVIDIA(
        model="meta/llama-3.2-3b-instruct",
        temperature=0.3, 
    )

    reviewer_base = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY_1"),
        temperature=0.2,
    )

    class ReviewSchema(BaseModel):
        score: float = Field(..., description="The score of the blog post evaluated on a scale from 1 to 10.")
        feedback: str = Field(..., description="Constructive critique detailing what to improve or change in the next iteration.")

    reviewer_llm = reviewer_base.with_structured_output(ReviewSchema)

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        current_draft: Optional[str]
        current_score: Optional[float]

    def writer_node(state: AgentState):
        history = "\n".join([f"{type(m).__name__}: {m.content}" for m in state["messages"]])
        prompt = f"""
You are a senior professional blog writer. Review the complete history and write/revise the blog post.

CRITICAL INSTRUCTION:
Look at the very last message from the User. If the user has changed the topic, requested a complete pivot, or provided instructions that contradict previous drafts, IGNORE the older drafts and write a brand new post based strictly on their latest request.

Conversation History & Feedback:
{history}

Guidelines:
- Compelling, keyword-rich headline
- Value-driven sections with clear H2/H3 subheads
- Short, scannable paragraphs and bullets
- Strong conclusion and clear CTA
"""
        ai_msg = writer_llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [ai_msg], "current_draft": ai_msg.content}

    def reviewer_node(state: AgentState):
        draft = state.get("current_draft", "No draft found.")
        prompt = f"""
You are an expert editorial critic. Your goal is to evaluate the blog post objectively using a strict point-based rubric. 
Start from 0 and add points based on the following criteria:

1. STRUCTURE (Up to 3 points):
   - 1 pt: Has a compelling, clear headline.
   - 1 pt: Uses organized H2/H3 subheadings.
   - 1 pt: Features short, scannable paragraphs and bullet points.

2. CONTENT & VALUE (Up to 4 points):
   - 2 pts: The content deeply addresses the topic and provides actionable insights.
   - 1 pt: Smooth transitions between paragraphs.
   - 1 pt: Strong concluding summary.

3. ENGAGEMENT (Up to 3 points):
   - 2 pts: Strong, clear Call-To-Action (CTA) at the end.
   - 1 pt: Engaging tone tailored for a professional audience.

Calculate the total score by summing up the points earned (Maximum 10, Minimum 1).

BLOG TO EVALUATE:
{draft}
"""
        parsed_review: ReviewSchema = reviewer_llm.invoke([HumanMessage(content=prompt)])
        review_content = f"Reviewer Score: {parsed_review.score}/10\nReviewer Critique: {parsed_review.feedback}"
        return {"messages": [SystemMessage(content=review_content)], "current_score": parsed_review.score}

    def human_review_node(state: AgentState):
        user_response = interrupt({
            "action_required": "Review blog draft",
            "draft": state.get("current_draft"),
            "score": state.get("current_score")
        })
        if user_response.get("approved"):
            return {"messages": [HumanMessage(content="Workflow Approved by User.")]}
        else:
            fb = user_response.get("feedback", "No feedback provided.")
            return {"messages": [HumanMessage(content=f"User Feedback: {fb}")]}

    def should_continue(state: AgentState):
        last_msg = state["messages"][-1].content
        if "Workflow Approved by User" in last_msg:
            return END
        return "writer"

    builder = StateGraph(AgentState)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("review", human_review_node)

    builder.add_edge("writer", "reviewer")
    builder.add_edge("reviewer", "review")
    builder.add_conditional_edges("review", should_continue)
    builder.set_entry_point("writer")

    return builder.compile(checkpointer=MemorySaver())


# ---------------------------
st.set_page_config(page_title="AI Blog Copilot", page_icon="✍️", layout="wide")
st.title("AI Blog Generator (Human-in-the-Loop)")
# st.caption("Orchestrating specialized LLM agents with real-time state interrupts.")

graph = get_compiled_graph()
config = {"configurable": {"thread_id": "streamlit-blog-session"}}

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "finalized" not in st.session_state:
    st.session_state.finalized = False

with st.sidebar:
    st.header("Control Sidebar")
    topic_input = st.text_input("Blog Topic", placeholder="e.g., Aritificial Intelligence")
    
    if st.button("🚀 Generate Blog", use_container_width=True, disabled=st.session_state.initialized):
        if topic_input:
            with st.spinner("Initializing agents and drafting..."):
                graph.invoke({"messages": [HumanMessage(content=f"Topic: {topic_input}")]}, config=config)
                st.session_state.initialized = True
                st.rerun()
        else:
            st.error("Please provide a topic first.")
            
    if st.button("🔄 Reset Entire Session", use_container_width=True, type="secondary"):
        st.session_state.clear()
        st.rerun()

if st.session_state.initialized:
    state_info = graph.get_state(config)
    current_state = state_info.values
    
    if not state_info.next:
        st.success("🎉 Workflow Finalized & Approved Successfully!")
        st.subheader("Final Optimized Blog Post")
        st.markdown(current_state.get("current_draft"))
        st.balloons()
        st.session_state.finalized = True

    elif "review" in state_info.next:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Current Generated Draft")
            with st.container(border=True):
                st.markdown(current_state.get("current_draft"))
                
        with col2:
            st.subheader("📊 Editorial Quality Evaluation")
            score = current_state.get("current_score", 0)

            st.metric(label="Reviewer Quality Rating", value=f"{score} / 10")
            
            critique_text = "No critique found."
            for msg in reversed(current_state.get("messages", [])):
                if isinstance(msg, SystemMessage) and "Reviewer Critique:" in msg.content:
                    critique_text = msg.content.split("Reviewer Critique:")[1].strip()
                    break
                    
            st.info(f"**Reviewer Critique:**\n{critique_text}")
            
            st.subheader("🛠️ Human-in-the-Loop")
            
            if st.button("✅ Approve & Finalize Blog", use_container_width=True, type="primary"):
                with st.spinner("Finishing workflow..."):
                    graph.invoke(Command(resume={"approved": True}), config=config)
                    st.rerun()
                    
            st.markdown("---")
            feedback_input = st.text_area("Request Revisions / Modify Topic Instruction", placeholder="e.g., Make the intro hook punchier, or write it about Quantum Computing instead.")
            
            if st.button("🔄 Send Feedback to Writer", use_container_width=True):
                if feedback_input.strip():
                    with st.spinner("Re-routing state back to agents..."):
                        graph.invoke(Command(resume={"approved": False, "feedback": feedback_input.strip()}), config=config)
                        st.rerun()
                else:
                    st.warning("Please enter feedback description before submitting revisions.")
else:
    st.info("👈 Enter a blog topic in the sidebar and click **Generate Blog** to trigger the writer and reviewer agent loop.")