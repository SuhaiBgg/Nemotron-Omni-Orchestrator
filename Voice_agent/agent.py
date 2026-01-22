import os
import base64
import yagmail
import re
import streamlit as st 
from dotenv import load_dotenv
from PIL import Image
import io
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import StructuredTool

load_dotenv()

# --- 1. Setup Models (NVIDIA NGC Catalog) ---
# Models initialized within a try-except block for connection resilience
try:
    # Multimodal model for analyzing diagrams like your CFG image
    vlm_model = ChatNVIDIA(model="nvidia/nemotron-nano-12b-v2-vl")
    # Moderation layer to ensure safe technical interactions
    safety_model = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-safety-guard-8b-v3")
    # Reasoning model with "Thinking" capability enabled
    chat_model = ChatNVIDIA(
        model="nvidia/nemotron-3-nano-30b-a3b",
        temperature=0.1,
        max_tokens=1000,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}} 
    )
except Exception as e:
    st.error(f"NVIDIA API Connection Error: {e}")

# --- 2. Setup Agentic Tools ---
tavily_key = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(tavily_api_key=tavily_key))

def send_email_to_father(content: str):
    """Sends a summary of the AI conversation to your father's email."""
    try:
        # Requires Google App Password
        yag = yagmail.SMTP(os.getenv("MY_GMAIL"), os.getenv("GMAIL_APP_PASSWORD"))
        yag.send(to=os.getenv("FATHER_GMAIL"), subject="AI Technical Report", contents=content)
        return "SUCCESS: Email sent."
    except Exception as e:
        return f"ERROR: {str(e)}"

email_tool = StructuredTool.from_function(
    func=send_email_to_father,
    description="Use this tool to send a summary of the technical discussion to the user's father via email."
)

# Bind tools to allow the model to choose actions
chat_model_with_tools = chat_model.bind_tools([search_tool, email_tool])
embeddings = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2")

# --- 3. Memory & RAG Logic ---
def get_session_history(session_id: str):
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

def run_agent(user_query, context_text="", session_id="default_user"):
    # Safety Check
    if "unsafe" in safety_model.invoke(user_query).content.lower():
        return "I cannot respond to this query due to safety policy violations."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical research assistant. Use LaTeX ($...$) for math. Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = prompt | chat_model_with_tools | StrOutputParser()

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return with_history.invoke(
        {"context": context_text, "input": user_query},
        config={"configurable": {"session_id": session_id}}
    )

def describe_image(image_path):
    """Handles image processing and VLM analysis."""
    img = Image.open(image_path)
    # Convert RGBA (transparency) to RGB for JPEG compatibility
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # Resize to stay within token limits
    img.thumbnail((1024, 1024)) 
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85) 
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    prompt_str = f'Describe all formulas and diagrams in this image: <img src="data:image/jpeg;base64,{img_b64}" />'
    try:
        response = vlm_model.invoke(prompt_str)
        return response.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"