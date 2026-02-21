import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "legal_passages"

def get_embeddings():
    # Use a lightweight, fast local embedding model
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore() -> Chroma:
    """Returns the Chroma vector store instance."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

def load_passages_to_chroma(passages_csv_path: str):
    """Loads passages from a CSV into ChromaDB if it doesn't already exist."""
    print(f"Loading passages from {passages_csv_path}...")
    df = pd.read_csv(passages_csv_path)
    
    # For quick evaluation/demonstration, only load the first 1000 passages.
    df = df.head(1000)
    
    # Expected columns: idx, text (and others we can store in metadata)
    documents = []
    for _, row in df.iterrows():
        # Ensure 'idx' and 'text' exist
        if 'idx' not in row or 'text' not in row or pd.isna(row['text']):
            continue
            
        metadata = {
            "faiss_id": str(row.get('faiss_id', '')),
            "idx": str(row['idx']),
            "source": str(row.get('source', ''))
        }
        
        doc = Document(page_content=str(row['text']), metadata=metadata)
        documents.append(doc)
        
    print(f"Prepared {len(documents)} documents. Initializing vectorstore...")
    vectorstore = get_vectorstore()
    
    # Check if we already have documents
    existing_count = vectorstore._collection.count()
    if existing_count > 0:
        print(f"Vectorstore already contains {existing_count} documents. Skipping injection to save time.")
        return vectorstore
        
    # Add in batches to avoid overwhelming memory/Chroma
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Adding batch {i} to {i+len(batch)}...")
        vectorstore.add_documents(batch)
        
    print("Finished loading documents into ChromaDB.")
    return vectorstore

def get_retriever(k: int = 5):
    """Returns a retriever interface for the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})

def retrieve_documents(query: str, k: int = 5) -> List[Document]:
    """Retrieves top k documents for a given query."""
    retriever = get_retriever(k=k)
    return retriever.invoke(query)

if __name__ == "__main__":
    # Test loading validation set
    valid_passages = "barexam_qa/passages/barexam_qa_validation.csv"
    if os.path.exists(valid_passages):
        load_passages_to_chroma(valid_passages)
    else:
        print(f"Could not find {valid_passages}")
