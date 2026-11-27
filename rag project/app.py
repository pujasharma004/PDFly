import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:1b"

st.set_page_config(page_title="DocChat", page_icon="💬", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #212121; color: #ececec;}

/* sidebar */
section[data-testid="stSidebar"] {background: #171717; border-right: 1px solid #2a2a2a;}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left; background: transparent; color: #ececec;
    border: 1px solid #333; border-radius: 10px; padding: 0.55rem 0.8rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {background: #2a2a2a; border-color: #444;}
.file-chip {
    background: #212121; border: 1px solid #2f2f2f; border-radius: 8px;
    padding: 6px 10px; margin: 4px 0; font-size: 0.85rem; color: #cfcfcf;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* chat column */
.block-container {max-width: 780px; padding-top: 2rem; padding-bottom: 7rem;}

/* messages */
[data-testid="stChatMessage"] {background: transparent; padding: 0.6rem 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
    background: #303030; border-radius: 18px; padding: 0.6rem 1.1rem;
    max-width: 80%; flex: 0 1 auto;
}
[data-testid="stChatMessageAvatarUser"] {display: none;}

/* input */
[data-testid="stBottom"] > div {background: #212121;}
[data-testid="stChatInput"] {
    border-radius: 28px; background: #303030; border: 1px solid #3a3a3a;
}
[data-testid="stChatInput"] textarea {background: transparent; color: #ececec;}

/* empty state */
.hero {text-align: center; margin-top: 18vh;}
.hero h1 {font-size: 2rem; font-weight: 600; margin-bottom: 0.4rem;}
.hero p {color: #9a9a9a;}

/* sources */
[data-testid="stExpander"] {border: 1px solid #2f2f2f; border-radius: 12px; background: #1c1c1c;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm(model_name: str):
    return ChatOllama(model=model_name)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a PDF-based AI assistant. Read and understand the entire uploaded PDF. Whenever the user asks a question, understand their intent and retrieve all relevant information from the PDF, even if it is spread across different pages or topics. You can logically connect, compare, and derive answers from related information present in the PDF, even if the exact answer is not explicitly written.

Never use external or general knowledge. If the answer is neither present nor derivable from the PDF, clearly say that the information is not available in the uploaded document.

Always provide accurate, relevant, and complete answers based strictly on the PDF. """,
        ),
        ("human", "content: {context}\n\nquestion: {question}"),
    ]
)


def load_existing_db():
    if os.path.isdir(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    return None


def build_db(uploaded_files):
    """Create a fresh vector DB from the uploaded PDFs."""
    docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for p in pages:
                p.metadata["source"] = f.name  # show real file name, not temp path
            docs.extend(pages)
        finally:
            os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # drop the old collection instead of deleting the folder (avoids file locks on Windows)
    old = st.session_state.get("vectorstore") or load_existing_db()
    if old is not None:
        try:
            old.delete_collection()
        except Exception:
            pass

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
    )


# ------------------------------------------------------------ session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_db()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 💬 DocChat")

    if st.button("＋  New chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Your documents**")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Process documents", disabled=not uploaded):
        with st.spinner("Reading and indexing PDFs..."):
            st.session_state.vectorstore = build_db(uploaded)
            st.session_state.indexed_files = [f.name for f in uploaded]
            st.session_state.messages = []
        st.success("Ready. Ask me anything.")

    if st.session_state.indexed_files:
        for name in st.session_state.indexed_files:
            st.markdown(f'<div class="file-chip">📄 {name}</div>', unsafe_allow_html=True)
    elif st.session_state.vectorstore is not None:
        st.caption("Using the existing database from the last session.")

    st.markdown("---")
    with st.expander("Settings"):
        k = st.slider("Chunks to retrieve", 2, 10, 4)
        show_sources = st.toggle("Show sources", value=True)
        model_name = st.text_input("Ollama model", value=LLM_MODEL)

# --------------------------------------------------------------- chat area
if st.session_state.vectorstore is None:
    st.markdown(
        """<div class="hero"><h1>Chat with your PDFs</h1>
        <p>Upload one or more PDFs in the sidebar, click <b>Process documents</b>, then ask a question.</p></div>""",
        unsafe_allow_html=True,
    )
    st.chat_input("Upload a document to start chatting", disabled=True)
    st.stop()

if not st.session_state.messages:
    st.markdown(
        """<div class="hero"><h1>What would you like to know?</h1>
        <p>Ask anything about your documents.</p></div>""",
        unsafe_allow_html=True,
    )

# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['file']} · page {s['page']}")
                    st.write(s["text"])

# new question
query = st.chat_input("Ask about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, k * 2), "lambda_mult": 0.5},
    )

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Thinking..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(d.page_content for d in docs)
                prompt = PROMPT.invoke({"context": context, "question": query})

            llm = get_llm(model_name)

            def token_stream():
                for chunk in llm.stream(prompt):
                    yield chunk.content

            answer = st.write_stream(token_stream())  # typing effect like ChatGPT
        except Exception as e:
            answer = f"Something went wrong: `{e}`\n\nMake sure Ollama is running and the model is pulled."
            docs = []
            st.error(answer)

        sources = [
            {
                "file": os.path.basename(d.metadata.get("source", "unknown")),
                "page": int(d.metadata.get("page", 0)) + 1,
                "text": d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""),
            }
            for d in docs
        ]
        if show_sources and sources:
            with st.expander("Sources"):
                for s in sources:
                    st.caption(f"📄 {s['file']} · page {s['page']}")
                    st.write(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources if show_sources else []}
    )
