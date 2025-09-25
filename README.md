# Project Title: AI-Powered-Q/A-Chatbot

# Description:
This project is a Question-Answering (QA) chatbot that lets users upload PDFs (like lecture notes, research papers, or Wikipedia dumps) and ask questions. The chatbot retrieves the most relevant parts of the document and generates accurate answers using Hugging Face Transformers and LangChain.

# Technologies:
1. LangChain 🦜
   A framework for building apps with LLMs. It helps connect language models with data sources (like PDFs, APIs, or vector databases).

2. FAISS 📚
   (Facebook AI Similarity Search) — A vector database to store and quickly search embeddings.

3. Hugging Face Transformers 🤗
   Provides pre-trained models for NLP tasks (e.g., text embeddings, Q&A, summarization).

4. Gradio 🎨
   A lightweight framework to build web UIs for ML apps.

5. PyPDFLoader 📄
   Loads and splits PDF documents into text chunks for embedding.

# Workflow:
1. Upload a PDF.

2. Text is split into chunks and converted into embeddings.

3. Embeddings are stored in FAISS for efficient retrieval.

4. When you ask a question:

    -> Relevant chunks are retrieved from FAISS.

    -> HuggingFacePipeline generates the final answer.

5. Answer is displayed in the Gradio chat UI.
