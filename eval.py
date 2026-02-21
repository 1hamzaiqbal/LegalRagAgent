import os
import pandas as pd
from tqdm import tqdm
from rag_utils import load_passages_to_chroma, retrieve_documents

def evaluate_retrieval(passages_csv: str, qa_csv: str, k: int = 5):
    """Evaluates the retrieval performance using Recall@K and MRR."""
    print("Setting up vectorstore...")
    if not os.path.exists(passages_csv):
        print(f"Error: {passages_csv} not found.")
        return
        
    if not os.path.exists(qa_csv):
        print(f"Error: {qa_csv} not found.")
        return
        
    # Ensure passages are loaded
    load_passages_to_chroma(passages_csv)
    
    print(f"Loading QA dataset from {qa_csv}...")
    qa_df = pd.read_csv(qa_csv)
    
    # Restrict to first 200 for faster demonstration
    qa_df = qa_df.head(200)
    
    total_queries = 0
    hits_at_k = 0
    mrr_sum = 0.0
    
    # Process queries
    print("Evaluating queries...")
    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df)):
        if 'question' not in row or 'gold_idx' not in row or pd.isna(row['question']):
            continue
            
        query = str(row['question'])
        gold_idx = str(row['gold_idx'])
        
        # Retrieve top k documents
        retrieved_docs = retrieve_documents(query, k=k)
        retrieved_ids = [doc.metadata.get('idx', '') for doc in retrieved_docs]
        
        # Calculate Hit (Recall@K)
        if gold_idx in retrieved_ids:
            hits_at_k += 1
            
        # Calculate Reciprocal Rank
        try:
            rank = retrieved_ids.index(gold_idx) + 1
            mrr_sum += 1.0 / rank
        except ValueError:
            pass # gold_idx not in retrieved_ids
            
        total_queries += 1
        
    if total_queries == 0:
        print("No valid queries found.")
        return
        
    recall_at_k = hits_at_k / total_queries
    mrr = mrr_sum / total_queries
    
    print("\n--- Evaluation Results ---")
    print(f"Total Queries: {total_queries}")
    print(f"Recall@{k}: {recall_at_k:.4f}")
    print(f"MRR: {mrr:.4f}")

if __name__ == "__main__":
    valid_passages = "barexam_qa/passages/barexam_qa_validation.csv"
    valid_qa = "barexam_qa/qa/barexam_qa_validation.csv"
    evaluate_retrieval(valid_passages, valid_qa, k=5)
