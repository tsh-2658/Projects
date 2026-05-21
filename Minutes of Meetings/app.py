
import os
import hashlib
import logging
import tempfile
import textwrap
import warnings
import json
import streamlit as st
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

warnings.filterwarnings("ignore")
for _name in ["langchain", "langchain_core", "faiss", "httpx", "httpcore"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error(
        "⚠️ GOOGLE_API_KEY not found. "
        "Set it in a .env file or in Streamlit Cloud → Secrets."
    )
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

try:
    from moviepy import VideoFileClip
    from faster_whisper import WhisperModel
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate
except ImportError as e:
    st.error(f"Missing dependency: {e}. Run `pip install -r requirements.txt`.")
    st.stop()

try:
    import nltk
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

MAX_UPLOAD_MB = 500

def file_hash(uploaded_file) -> str:
    """SHA-256 of the uploaded bytes — used as a stable cache key."""
    uploaded_file.seek(0)
    h = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h

def extract_audio(video_path: str) -> str:
    with VideoFileClip(video_path) as clip:
        if clip.audio is None:
            raise ValueError("The uploaded video has no audio track.")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.close()
        clip.audio.write_audiofile(tmp.name, logger=None)
    return tmp.name


def transcribe_audio(audio_path: str) -> str:
    model = WhisperModel("small", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)

def text_to_pdf(text: str) -> str:
    """Write transcript to a temp PDF for user download only."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    c = canvas.Canvas(tmp.name, pagesize=A4)
    page_w, page_h = A4
    margin, line_height, font_size = 60, 14, 11
    chars_per_line = int((page_w - 2 * margin) / (font_size * 0.55))
    c.setFont("Helvetica", font_size)
    y = page_h - margin
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph or " ", width=chars_per_line):
            if y < margin + line_height:
                c.showPage()
                c.setFont("Helvetica", font_size)
                y = page_h - margin
            c.drawString(margin, y, line)
            y -= line_height
        y -= line_height // 2
    c.save()
    return tmp.name


@st.cache_resource(show_spinner=False)
def build_retriever(text: str, _file_hash: str):  # _file_hash is the stable cache key
    """
    Embed transcript text directly — no PDF loader needed.
    _file_hash is prefixed with _ so Streamlit doesn't hash it (we manage it).
    """
    if _NLTK_OK:
        sentences = sent_tokenize(text)
        chunks_text, current, current_len = [], [], 0
        for sent in sentences:
            if current_len + len(sent) > 500 and current:
                chunks_text.append(" ".join(current))
                current = current[-1:]          # 1-sentence overlap
                current_len = len(current[0])
            current.append(sent)
            current_len += len(sent)
        if current:
            chunks_text.append(" ".join(current))
        documents = [Document(page_content=c) for c in chunks_text]
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = splitter.split_documents([Document(page_content=text)])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(documents, embeddings)
    return db.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert meeting analyst.
Below is a transcript extracted from a group discussion video.

Transcript:
{context}

Task: {question}

Respond ONLY with a JSON object — no markdown fences, no extra text — using this exact schema:
{{
  "overview": "<2-3 sentence meeting overview>",
  "key_points": ["<point 1>", "<point 2>", ...],
  "decisions": ["<decision 1>", ...],
  "action_items": [
    {{"owner": "<name or 'TBD'>", "task": "<what needs to be done>"}},
    ...
  ],
  "participants": ["<name or role>", ...]
}}
If a section has no content, use an empty list [].""",
)

CHAT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant with access to a meeting transcript.
Answer the user's question using only the information in the transcript.
If the answer is not in the transcript, say so clearly.

Transcript context:
{context}

User question: {question}

Answer:""",
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_qa(retriever, prompt_template: PromptTemplate, query: str) -> str:
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | get_llm()
        | StrOutputParser()
    )
    return chain.invoke(query).strip()


def run_qa_stream(retriever, prompt_template: PromptTemplate, query: str):
    """Generator — yields string tokens as they arrive from the LLM."""
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | get_llm()
        | StrOutputParser()
    )
    yield from chain.stream(query)


def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def render_summary(raw: str):
    """Parse the JSON summary and render it as structured cards."""
    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        st.write(raw)
        return

    if overview := data.get("overview"):
        st.info(overview)

    col1, col2 = st.columns(2)

    with col1:
        if kp := data.get("key_points"):
            with st.expander("📌 Key points", expanded=True):
                for point in kp:
                    st.markdown(f"- {point}")

        if decisions := data.get("decisions"):
            with st.expander("✅ Decisions", expanded=True):
                for d in decisions:
                    st.markdown(f"- {d}")

    with col2:
        if ai := data.get("action_items"):
            with st.expander("📋 Action items", expanded=True):
                for item in ai:
                    owner = item.get("owner", "TBD")
                    task = item.get("task", "")
                    st.markdown(f"- **{owner}**: {task}")

        if participants := data.get("participants"):
            with st.expander("👥 Participants"):
                st.markdown(", ".join(participants))



st.set_page_config(page_title="Meeting Analyzer", page_icon="🎥", layout="wide")
st.title("🎥 Meeting Video Analyzer")
st.caption("Upload a meeting recording to get a summary and ask questions about it.")

for key in ["transcript", "summary", "pdf_path", "retriever", "chat_history", "file_hash"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

uploaded_file = st.file_uploader(
    "Upload a meeting video", type=["mp4", "mkv", "mov", "avi"]
)

if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File is {size_mb:.0f} MB — max allowed is {MAX_UPLOAD_MB} MB.")
        st.stop()

    current_hash = file_hash(uploaded_file)
    if st.session_state.file_hash == current_hash and st.session_state.transcript:
        st.success("✅ Same file detected — using cached results.")
    else:
        st.session_state.file_hash = current_hash
        for key in ["transcript", "summary", "pdf_path", "retriever"]:
            st.session_state[key] = None
        st.session_state.chat_history = []

        tmp_video = tmp_audio = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as f:
                f.write(uploaded_file.read())
                tmp_video = f.name

            with st.status("Processing video…", expanded=True) as status:
                st.write("🔊 Extracting audio…")
                tmp_audio = extract_audio(tmp_video)

                st.write("📝 Transcribing audio (this may take a minute)…")
                transcript = transcribe_audio(tmp_audio)
                if not transcript.strip():
                    st.error("Transcription produced no text. Check that the video has clear speech.")
                    st.stop()
                st.session_state.transcript = transcript   # ← saved early

                # Step 3 — PDF for download (text_to_pdf unchanged)
                st.write("📄 Building download PDF…")
                st.session_state.pdf_path = text_to_pdf(transcript)

                # Step 4 — embedding (no PDF loader, uses text directly)
                st.write("🔍 Creating vector index…")
                st.session_state.retriever = build_retriever(transcript, current_hash)

                # Step 5 — structured summary
                st.write("✍️ Generating summary…")
                st.session_state.summary = run_qa(
                    st.session_state.retriever,
                    SUMMARY_PROMPT,
                    "Summarize the key points, decisions, action items, and participants.",
                )
                status.update(label="✅ Done!", state="complete")

        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()
        finally:
            cleanup(tmp_video, tmp_audio)

if st.session_state.transcript:
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "💬 Chat", "📄 Full Transcript"])

    with tab1:
        st.subheader("Meeting Summary")

        render_summary(st.session_state.summary)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Regenerate Summary"):
                with st.spinner("Regenerating…"):
                    st.session_state.summary = run_qa(
                        st.session_state.retriever,
                        SUMMARY_PROMPT,
                        "Summarize the key points, decisions, action items, and participants.",
                    )
                st.rerun()
        with col2:
            if st.session_state.pdf_path:
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Transcript PDF",
                        f,
                        file_name="transcript.pdf",
                        mime="application/pdf",
                    )

    with tab2:
        st.subheader("Ask Questions About the Meeting")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Ask anything about the meeting…")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    run_qa_stream(st.session_state.retriever, CHAT_PROMPT, user_question)
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()

    with tab3:
        st.subheader("Full Transcript")
        # Search filter
        search = st.text_input("🔍 Search transcript", placeholder="Type to highlight…")
        transcript_text = st.session_state.transcript
        if search and search.lower() in transcript_text.lower():
            # Simple highlight: show only surrounding context
            idx = transcript_text.lower().find(search.lower())
            start = max(0, idx - 200)
            end = min(len(transcript_text), idx + 500)
            st.info(f"Showing match near position {idx}:")
            st.markdown(
                transcript_text[start:end].replace(
                    transcript_text[idx: idx + len(search)],
                    f"**{transcript_text[idx: idx + len(search)]}**",
                )
            )
        st.text_area(
            label="",
            value=transcript_text,
            height=400,
            label_visibility="collapsed",
        )

if st.session_state.transcript:
    st.divider()
    if st.button("🔁 Analyze a new video"):
        cleanup(st.session_state.pdf_path)
        for key in ["transcript", "summary", "pdf_path", "retriever", "file_hash"]:
            st.session_state[key] = None
        st.session_state.chat_history = []
        st.rerun()
