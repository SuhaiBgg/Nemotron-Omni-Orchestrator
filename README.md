# Nemotron-Omni-Orchestrator

**Nemotron-Omni-Orchestrator** is a multimodal Agentic RAG system powered by the NVIDIA NIM ecosystem. It orchestrates high-fidelity reasoning, technical vision analysis, and real-time tool execution including web search and email reporting through a professional voice-enabled interface.

## Key Features

* **Agentic Orchestration**: The system uses a reasoning engine to determine whether to retrieve local knowledge, perform a web search via Tavily, or trigger external actions like email reporting.
* **Technical Vision (VLM)**: Integrated NVIDIA Nemotron-12B-V2-VL for the analysis of complex technical diagrams, mathematical formulas, and formal grammars such as .
* **Multimodal RAG**: Combines FAISS vector storage with NVIDIA Embeddings to provide high-accuracy retrieval for PDF/TXT documents and visual data descriptions.
* **Professional Terminal Interface**: A customized Streamlit UI featuring a Transparent Activity Log for real-time orchestration tracking and a dedicated Debug Window for inspection of retrieved context.
* **Voice-to-Action**: Full speech-to-text and text-to-speech integration, allowing for hands-free technical research.
* **Enterprise Safety**: Implements the Llama-3.1-Nemotron-Safety-Guard to filter and moderate technical queries.

---

## Tech Stack

* **LLM/VLM**: NVIDIA Nemotron-3-30B for reasoning and Nemotron-12B for vision tasks.
* **Orchestration**: LangChain Expression Language (LCEL) and RunnableWithMessageHistory.
* **Vector Database**: FAISS (Facebook AI Similarity Search).
* **Tools**: Tavily Search API and Yagmail for SMTP orchestration.
* **Frontend**: Streamlit with custom CSS terminal styling.

---

## Getting Started

### 1. Prerequisites

* Python 3.10 or higher
* NVIDIA NGC API Key
* Tavily API Key
* Google App Password for email orchestration

### 2. Installation

```bash
git clone https://github.com/yourusername/Nemotron-Omni-Orchestrator.git
cd Nemotron-Omni-Orchestrator
pip install -r requirements.txt

```

### 3. Environment Setup

Create a .env file in the root directory:

```env
NVIDIA_API_KEY=your_nvidia_key
TAVILY_API_KEY=your_tavily_key
MY_GMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
FATHER_GMAIL=recipient_email@gmail.com

```

### 4. Running the Application

```bash
streamlit run app.py

```

---

## System Architecture

1. **User Input**: The user provides a query via Voice or Text.
2. **Orchestrator**: The Nemotron-30B model analyzes the user intent.
3. **Retrieval**: The system queries the FAISS index for relevant PDF text or previously analyzed image descriptions.
4. **Action Decision**: If the internal context is insufficient, the Orchestrator triggers a Tavily Search.
5. **Output**: The final response is rendered in LaTeX for mathematical notation and spoken back to the user.
6. **Hand-off**: If requested, the Email Tool sends a summary of the findings to the designated recipient.


---

Would you like me to generate the requirements.txt file with the specific library versions required for this build?
