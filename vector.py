# vector.py
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def create_vector_store(documents, collection_name, base_dir="vector_dbs"):
    """
    Create (or load) a vector store for a given collection.
    
    Parameters:
      documents: list of Document objects.
      collection_name: name of the collection (unique per uploaded file).
      base_dir: base directory to store vector databases.
    """
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    db_location = os.path.join(base_dir, collection_name)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    add_documents = not os.path.exists(db_location)
    
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    if add_documents:
        ids = [doc.id for doc in documents]
        vector_store.add_documents(documents=documents, ids=ids)
    
    return vector_store

# Helper functions to process files
def process_csv(df):
    """
    Process a CSV DataFrame into a list of Document objects.
    This version automatically uses all columns in each row:
    - All column values are concatenated to form the document's content.
    - The full row is stored as metadata.
    """
    documents = []
    for i, row in df.iterrows():
        # Concatenate all column values into a single string.
        content = " ".join([str(val) for val in row.values])
        # Save the full row as metadata.
        metadata = row.to_dict()
        documents.append(Document(page_content=content, metadata=metadata, id=str(i)))
    return documents


def process_pdf(text, chunk_size=500):
    """
    Split the text from a PDF into smaller chunks (documents).
    """
    words = text.split()
    documents = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        documents.append(Document(page_content=chunk, metadata={"chunk": i}, id=str(i)))
    return documents
