# NVIDIA-Powered Voice RAG Assistant

A professional-grade voice agent built with **NVIDIA Nemotron** and **LangGraph**. This system features multimodal RAG grounding, real-time safety guardrails, and long-context reasoning.

## Features

- **Speech-to-Text:** Real-time voice processing.
- **Safety Layer:** Llama-3.1-Nemotron-Safety-Guard filters all inputs/outputs.
- **RAG Engine:** Local FAISS vector store for domain-specific grounding.
- **Reasoning:** Nemotron-3-Nano with "Thinking Mode" enabled.
- **UI:** Streamlit dashboard with internal reasoning visibility.

## Tech Stack

- **Framework:** LangChain & LangGraph
- **Models:** NVIDIA Nemotron-3-Nano, Llama-3.1-Safety-Guard
- **Database:** FAISS (CPU)
- **Frontend:** Streamlit
