"""Download and prepare legal-rag-qa, open-australian-legal-qa, and casehold datasets."""
import json
import os
import sys

import pandas as pd
from datasets import load_dataset


def prep_legal_rag_qa():
    """legal-rag-qa: 190 passages + 138 QA pairs with gold passage IDs."""
    out_dir = "datasets/legal_rag_qa"
    os.makedirs(out_dir, exist_ok=True)

    # Corpus: 190 passages
    corpus_ds = load_dataset("isaacus/legal-rag-qa")["test"]
    corpus_rows = []
    for row in corpus_ds:
        corpus_rows.append({
            "idx": row["id"],
            "title": row["title"],
            "text": row["text"],
            "section": row["section"],
            "is_supplemental": row["is_supplemental"],
        })
    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_csv(os.path.join(out_dir, "passages.csv"), index=False)
    print(f"legal-rag-qa corpus: {len(corpus_df)} passages -> {out_dir}/passages.csv")

    # QA pairs: 138 questions
    qa_ds = load_dataset("isaacus/legal-rag-qa", "qa")["test"]
    qa_rows = []
    for row in qa_ds:
        qa_rows.append({
            "idx": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "requires_supplemental": row["requires_supplemental"],
            "relevant_passages": json.dumps(row["relevant_passages"]),
        })
    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_csv(os.path.join(out_dir, "questions.csv"), index=False)
    print(f"legal-rag-qa QA: {len(qa_df)} questions -> {out_dir}/questions.csv")


def prep_australian_legal_qa():
    """open-australian-legal-qa: 2124 QA pairs with source document snippets."""
    out_dir = "datasets/australian_legal_qa"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("isaacus/open-australian-legal-qa", "default")["train"]

    # Extract source passages as corpus
    corpus_rows = []
    qa_rows = []
    for i, row in enumerate(ds):
        # Parse source metadata
        try:
            source = json.loads(row["source"].replace("'", '"'))
        except (json.JSONDecodeError, AttributeError):
            source = {}

        passage_idx = f"aus_{i}"

        # Extract the actual source text from the prompt
        # The prompt contains <document_text>...</document_text>
        prompt = row["prompt"]
        text_start = prompt.find("<document_text>")
        text_end = prompt.find("</document_text>")
        if text_start >= 0 and text_end >= 0:
            source_text = prompt[text_start + len("<document_text>"):text_end].strip()
        else:
            source_text = ""

        corpus_rows.append({
            "idx": passage_idx,
            "text": source_text if source_text else row["answer"],
            "citation": source.get("citation", ""),
            "jurisdiction": source.get("jurisdiction", ""),
            "source_type": source.get("type", ""),
            "url": source.get("url", ""),
        })

        qa_rows.append({
            "idx": passage_idx,
            "question": row["question"],
            "answer": row["answer"],
            "gold_idx": passage_idx,
            "jurisdiction": source.get("jurisdiction", ""),
            "citation": source.get("citation", ""),
        })

    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_csv(os.path.join(out_dir, "passages.csv"), index=False)
    print(f"australian QA corpus: {len(corpus_df)} passages -> {out_dir}/passages.csv")

    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_csv(os.path.join(out_dir, "questions.csv"), index=False)
    print(f"australian QA: {len(qa_df)} questions -> {out_dir}/questions.csv")


def prep_casehold():
    """CaseHOLD: 5-way MC holding identification."""
    out_dir = "datasets/casehold"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("coastalcph/lex_glue", "case_hold")

    for split in ["test", "train"]:
        rows = []
        for i, row in enumerate(ds[split]):
            endings = row["endings"]
            label = row["label"]
            rows.append({
                "idx": f"ch_{split}_{i}",
                "question": row["context"],
                "choice_a": endings[0],
                "choice_b": endings[1],
                "choice_c": endings[2],
                "choice_d": endings[3],
                "choice_e": endings[4],
                "answer": chr(ord("A") + label),  # 0->A, 1->B, etc.
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        print(f"casehold {split}: {len(df)} rows -> {out_dir}/{split}.csv")

    # Build holdings corpus from train set for retrieval
    # Each unique holding becomes a passage
    holdings = set()
    corpus_rows = []
    for i, row in enumerate(ds["train"]):
        for j, ending in enumerate(row["endings"]):
            if ending not in holdings:
                holdings.add(ending)
                corpus_rows.append({
                    "idx": f"holding_{len(corpus_rows)}",
                    "text": ending,
                    "source": "train",
                })
    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_csv(os.path.join(out_dir, "holdings_corpus.csv"), index=False)
    print(f"casehold holdings corpus: {len(corpus_df)} unique holdings -> {out_dir}/holdings_corpus.csv")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("legal_rag", "all"):
        prep_legal_rag_qa()
    if target in ("australian", "all"):
        prep_australian_legal_qa()
    if target in ("casehold", "all"):
        prep_casehold()
