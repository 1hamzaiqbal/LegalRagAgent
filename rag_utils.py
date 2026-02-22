import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

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


def compute_confidence(query: str, docs: List[Document]) -> float:
    """Compute confidence as mean cosine similarity between query and doc embeddings."""
    if not docs:
        return 0.0

    embeddings = get_embeddings()
    query_emb = np.array(embeddings.embed_query(query))
    doc_texts = [doc.page_content for doc in docs]
    doc_embs = np.array(embeddings.embed_documents(doc_texts))

    # Cosine similarity between query and each doc
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    similarities = doc_norms @ query_norm

    return float(np.mean(similarities))

@tool
def retrieve_legal_passages(query: str, k: int = 5) -> str:
    """Retrieve relevant legal passages from the bar exam corpus.

    Args:
        query: A legal research query to search for relevant passages.
        k: Number of passages to retrieve (default 5).

    Returns:
        Formatted string with numbered passages and their source IDs.
    """
    docs = retrieve_documents(query, k=k)
    if not docs:
        return "No relevant passages found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source_id = doc.metadata.get("idx", "unknown")
        parts.append(f"[Passage {i}] (source: {source_id})\n{doc.page_content}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    # Test loading validation set
    valid_passages = "barexam_qa/passages/barexam_qa_validation.csv"
    if os.path.exists(valid_passages):
        load_passages_to_chroma(valid_passages)
    else:
        print(f"Could not find {valid_passages}")
