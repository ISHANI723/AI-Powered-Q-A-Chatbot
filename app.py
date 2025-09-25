import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# LangChain (new imports for v0.2+)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # (Facebook AI Similarity Search)
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline


# Function to extract text from PDF
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


# Function to split text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    return splitter.split_text(text)


# Function to create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


# Function to build QA chain with HuggingFace model
def build_qa_chain(vector_store):
    model_name = "google/flan-t5-base"  # good lightweight model for Q&A

    qa_pipeline = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain


# Streamlit UI
st.set_page_config(page_title="📚 PDF Q&A Chatbot (Offline)", layout="wide")
st.title("📚 AI-Powered PDF Q&A Chatbot using HuggingFace Transformer (Offline)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.info("Processing PDF... Please wait ⏳")

    # Extract text
    text = load_pdf(uploaded_file)

    if not text.strip():
        st.error("❌ Could not extract any text from this PDF.")
    else:
        # Split, embed, and build chain
        chunks = split_text(text)
        vector_store = create_vector_store(chunks)
        qa_chain = build_qa_chain(vector_store)

        st.success("✅ PDF processed! You can now ask questions.")

        # Chat interface
        if "history" not in st.session_state:
            st.session_state.history = []

        query = st.text_input("Ask a question about the PDF:")

        if query:
            answer = qa_chain.run(query)
            st.session_state.history.append((query, answer))

        # Display history
        for q, a in st.session_state.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
