# app.py
import os
import streamlit as st
import pandas as pd
from io import StringIO
import PyPDF2

from vector import create_vector_store, process_csv, process_pdf
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.title("Chat with Your Documents")

# Section 1: File Upload and Processing
st.header("Upload a Document")
uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])

if uploaded_file is not None:
    file_name = uploaded_file.name.split('.')[0]  # use file name (without extension) as collection name
    st.write(f"Processing **{uploaded_file.name}**...")
    documents = None

    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.write("CSV Preview:")
        st.dataframe(df.head())
        documents = process_csv(df)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        st.write("PDF Text Preview (first 500 characters):")
        st.text(text[:500] + '...')
        documents = process_pdf(text)

    if documents:
        st.write("Creating/updating vector store for this collection...")
        vector_store = create_vector_store(documents, collection_name=file_name)
        st.success(f"Collection **{file_name}** is ready!")

# Section 2: Choose a Collection to Query
st.header("Select a Collection to Query")
# List all available collections by scanning the base vector database directory
base_dir = "vector_dbs"
collections = []
if os.path.exists(base_dir):
    collections = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

if collections:
    selected_collection = st.selectbox("Choose a collection", collections)
    
    # Load the selected collection's vector store
    # (Since the collection already exists, no need to add documents)
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db_location = os.path.join(base_dir, selected_collection)
    vector_store = Chroma(
        collection_name=selected_collection,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Section 3: Chat Interface
    st.header("Ask a Question")
    question = st.text_input("Enter your question for the selected collection:")
    
    if st.button("Submit Question") and question:
        # Retrieve relevant documents
        reviews = retriever.invoke(question)
        
        # Prepare the prompt for your chain
        template = """
You are an expert in answering questions based on the provided document excerpts.

Here are some relevant excerpts: {reviews}

Here is the question to answer: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model="llama3.2")
        chain = prompt | model
        
        # Get the answer from the model
        result = chain.invoke({"reviews": reviews, "question": question})
        
        st.subheader("Answer")
        st.write(result)
else:
    st.info("No collections available yet. Please upload a file to create a new collection.")
