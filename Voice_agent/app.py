import streamlit as st
import speech_recognition as sr
import pyttsx3
import pythoncom
import re
import os
import io
from PIL import Image
from datetime import datetime
from agent import describe_image, run_agent, send_email_to_father, embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# --- Configuration & State ---
if not os.path.exists("data"):
    os.makedirs("data")

if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'listening' not in st.session_state: st.session_state.listening = False
if 'agent_log' not in st.session_state: st.session_state.agent_log = []
if 'last_retrieval' not in st.session_state: st.session_state.last_retrieval = "No context retrieved yet."

def add_log(msg):
    """Appends a timestamped log to the activity history."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_log.append(f"[{timestamp}] ⚡ {msg}")
    if len(st.session_state.agent_log) > 15: st.session_state.agent_log.pop(0)

# --- Visual Styling (Transparent Terminal CSS) ---
st.set_page_config(page_title="NVIDIA Voice Pro", layout="wide")
st.markdown("""
    <style>
    .terminal-log {
        background-color: rgba(10, 10, 10, 0.85);
        color: #00FF41; /* Matrix Green */
        font-family: 'Consolas', 'Courier New', monospace;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #003300;
        font-size: 11px;
        height: 350px;
        overflow-y: auto;
    }
    .stChatFloatingInputContainer { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LEFT SIDEBAR: Logs & Dashboard ---
with st.sidebar:
    st.header("📋 Activity Log")
    # Render the transparent terminal log
    log_html = "".join([f"<div style='margin-bottom:4px;'>{log}</div>" for log in st.session_state.agent_log])
    st.markdown(f'<div class="terminal-log">{log_html}</div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Logs", use_container_width=True):
        st.session_state.agent_log = []
        st.rerun()

    st.markdown("---")
    st.header("📂 Knowledge Dashboard")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    if uploaded_file and st.button("Update Brain"):
        with st.spinner("Processing..."):
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(file_path) if uploaded_file.name.endswith(".pdf") else TextLoader(file_path)
            splits = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(loader.load())
            FAISS.from_documents(splits, embeddings).save_local("faiss_index")
            add_log(f"Knowledge Source Indexed: {uploaded_file.name}")
            st.success("Brain Updated!")

    st.markdown("---")
    uploaded_img = st.file_uploader("Upload Technical Diagram", type=["png", "jpg", "jpeg"])
    if uploaded_img and st.button("👁️ Analyze Diagram"):
        with st.spinner("VLM Analyzing..."):
            temp_path = f"data/{uploaded_img.name}"
            with open(temp_path, "wb") as f: f.write(uploaded_img.getbuffer())
            desc = describe_image(temp_path)
            new_doc = Document(page_content=f"Image Description ({uploaded_img.name}): {desc}")
            if os.path.exists("faiss_index"):
                vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                vs.add_documents([new_doc])
                vs.save_local("faiss_index")
            else:
                FAISS.from_documents([new_doc], embeddings).save_local("faiss_index")
            add_log(f"Visual Context Added: {uploaded_img.name}")
            st.success("Visual knowledge stored!")

    st.markdown("---")
    if st.button("📧 Email to Father", use_container_width=True):
        if st.session_state.chat_history:
            add_log("Triggering Email Tool...")
            res = send_email_to_father(st.session_state.chat_history[-1]["agent"])
            st.success(res)
        else: st.warning("No conversation history found.")

# --- MAIN LAYOUT ---
main_col, debug_col = st.columns([0.7, 0.3])

with debug_col:
    st.header("🔍 Debug Window")
    with st.container(border=True):
        st.subheader("Retrieved Context")
        # Visualizes the math retrieved from the FAISS index
        st.info(st.session_state.last_retrieval)

with main_col:
    st.title("🎙️ NVIDIA AI Voice Pro")
    
    def process_interaction(query):
        add_log(f"New Query: {query[:25]}...")
        with st.chat_message("assistant"):
            # Retrieval Logic
            context = ""
            if os.path.exists("faiss_index"):
                add_log("Retrieving Context from VectorDB...")
                vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = vs.as_retriever().invoke(query)
                context = "\n".join([d.page_content for d in docs])
                st.session_state.last_retrieval = context if context else "Zero matches found in Knowledge Base."
            
            with st.status("Thinking...") as s:
                add_log("Consulting NVIDIA Reasoning Engine...")
                response = run_agent(query, context_text=context)
                res_text = response if isinstance(response, str) else response.content
                s.update(label="Response Generated", state="complete")
            
            st.markdown(res_text)
            st.session_state.chat_history.append({"user": query, "agent": res_text})
            add_log("Interaction Finalized.")
            # speak(res_text) # Trigger TTS engine here

    # Chat Input
    manual_query = st.chat_input("Enter a technical question...")
    for chat in st.session_state.chat_history:
        with st.chat_message("user"): st.write(chat["user"])
        with st.chat_message("assistant"): st.write(chat["agent"])
    
    if manual_query:
        with st.chat_message("user"): st.write(manual_query)
        process_interaction(manual_query)

    # Voice Controls
    st.markdown("---")
    if not st.session_state.listening:
        if st.button("🎙️ Start Voice Mode", type="primary", use_container_width=True):
            st.session_state.listening = True
            st.rerun()
    else:
        if st.button("🛑 Stop Voice Mode", use_container_width=True):
            st.session_state.listening = False
            st.rerun()