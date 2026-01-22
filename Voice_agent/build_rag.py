import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Load your API Key from the .env file
load_dotenv()

def setup_knowledge_base():
    # 2. Load the document
    loader = TextLoader("data/manual.txt")
    docs = loader.load()

    # 3. Split text into small chunks so the AI can find specific info
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 4. Use NVIDIA's Cloud to turn text into vectors (numbers)
    # This happens in the cloud, so your Windows RAM is safe!
    embeddings = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2")

    # 5. Create the Vector Store (The Brain) and save it locally
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    print("Knowledge base created successfully!")

if __name__ == "__main__":
    setup_knowledge_base()