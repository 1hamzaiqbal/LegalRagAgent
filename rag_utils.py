import functools
import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "legal_passages"
QA_MEMORY_COLLECTION = "qa_memory"

@functools.lru_cache(maxsize=1)
def get_embeddings():
    # Use a lightweight, fast local embedding model
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

_vectorstore_instance = None

def get_vectorstore() -> Chroma:
    """Returns the Chroma vector store singleton."""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        embeddings = get_embeddings()
        _vectorstore_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
    return _vectorstore_instance

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


_memory_store_instance = None

def get_memory_store() -> Chroma:
    """Returns the Chroma QA memory collection singleton (cosine distance)."""
    global _memory_store_instance
    if _memory_store_instance is None:
        embeddings = get_embeddings()
        _memory_store_instance = Chroma(
            collection_name=QA_MEMORY_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _memory_store_instance


def check_memory(query: str, threshold: float = 0.92) -> Dict[str, Any]:
    """Check if a similar question has been answered before.

    Uses cosine similarity. Threshold of 0.92 requires near-exact match to
    avoid serving cached answers for substantially different questions.

    Returns {"found": bool, "answer": str, "confidence": float, "question": str}.
    """
    store = get_memory_store()
    # Only search if the collection has documents
    if store._collection.count() == 0:
        return {"found": False, "answer": "", "confidence": 0.0, "question": ""}

    results = store.similarity_search_with_relevance_scores(query, k=1)
    if results:
        doc, score = results[0]
        if score >= threshold:
            return {
                "found": True,
                "answer": doc.metadata.get("answer", ""),
                "confidence": score,
                "question": doc.page_content,
            }
    return {"found": False, "answer": "", "confidence": 0.0, "question": ""}


def write_to_memory(question: str, answer: str, confidence: float) -> None:
    """Store a question-answer pair in the QA memory collection."""
    store = get_memory_store()
    doc = Document(
        page_content=question,
        metadata={
            "answer": answer,
            "confidence": str(confidence),
            "timestamp": str(time.time()),
        },
    )
    store.add_documents([doc])


if __name__ == "__main__":
    # Test loading validation set
    valid_passages = "barexam_qa/passages/barexam_qa_validation.csv"
    if os.path.exists(valid_passages):
        load_passages_to_chroma(valid_passages)
    else:
        print(f"Could not find {valid_passages}")
