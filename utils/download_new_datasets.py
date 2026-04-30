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
    corpus_text_by_id = {}
    for row in corpus_ds:
        corpus_text_by_id[str(row["id"])] = row["text"]
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
        relevant_passages = row["relevant_passages"]
        relevant_ids = []
        for passage in relevant_passages:
            if isinstance(passage, dict):
                pid = passage.get("id") or passage.get("idx") or passage.get("passage_id")
            else:
                pid = passage
            if pid is not None:
                relevant_ids.append(str(pid))
        qa_rows.append({
            "idx": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "requires_supplemental": row["requires_supplemental"],
            "relevant_passages": json.dumps(relevant_passages),
            "gold_idx": ",".join(relevant_ids),
            "gold_passage": "\n\n".join(corpus_text_by_id.get(pid, "") for pid in relevant_ids),
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


def prep_musique():
    """MuSiQue-Ans: 2-4 hop compositional multi-hop QA.

    Schema per question (from `dgslibisey/MuSiQue`):
      - id, question, answer, answer_aliases (list)
      - paragraphs: [{idx, title, paragraph_text, is_supporting}, ...] (~20)
      - question_decomposition: [{question, answer, paragraph_support_idx}, ...]

    We write two CSVs:
      - questions.csv: one row per question with answer + gold_idx (comma-sep
        list of supporting paragraph idxs, formatted as 'musique_{qid}_{pidx}')
        + gold_passage (newline-joined supporting paragraph texts)
      - passages.csv: one row per paragraph across all questions, idx
        formatted as 'musique_{qid}_{pidx}'. Used to build the
        `musique_passages` ChromaDB collection.
    """
    out_dir = "datasets/musique"
    os.makedirs(out_dir, exist_ok=True)

    # Use the validation split — has gold labels and is what published baselines
    # report on. Train is much larger but we don't need it for evaluation.
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    qa_rows = []
    corpus_rows = []
    seen_paragraph_idx = set()

    for row in ds:
        qid = row["id"]

        # Build answer alias list — answer_aliases may be None
        aliases = row.get("answer_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        # Index gold paragraphs from question_decomposition
        decomp = row.get("question_decomposition") or []
        gold_para_idxs = []
        for hop in decomp:
            psidx = hop.get("paragraph_support_idx")
            if psidx is not None:
                gold_para_idxs.append(int(psidx))

        # Walk paragraphs, write each to corpus, collect supporting texts
        gold_global_idxs = []
        gold_texts = []
        for p in row["paragraphs"]:
            local_idx = int(p["idx"])
            global_idx = f"musique_{qid}_{local_idx}"

            # Corpus row (dedupe in case the same paragraph appears across rows)
            if global_idx not in seen_paragraph_idx:
                seen_paragraph_idx.add(global_idx)
                corpus_rows.append({
                    "idx": global_idx,
                    "title": p.get("title", ""),
                    "text": p.get("paragraph_text", ""),
                    "q_id": qid,
                    "is_supporting": bool(p.get("is_supporting", False)),
                })

            if local_idx in gold_para_idxs:
                gold_global_idxs.append(global_idx)
                gold_texts.append(p.get("paragraph_text", ""))

        qa_rows.append({
            "idx": qid,
            "question": row["question"],
            "answer": row["answer"],
            "answer_aliases": json.dumps(aliases),
            "gold_idx": ",".join(gold_global_idxs),
            "gold_passage": "\n\n".join(gold_texts),
            "n_hops": len(gold_global_idxs),
            "answerable": bool(row.get("answerable", True)),
        })

    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_csv(os.path.join(out_dir, "questions.csv"), index=False)
    print(f"musique QA: {len(qa_df)} questions -> {out_dir}/questions.csv")

    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_csv(os.path.join(out_dir, "passages.csv"), index=False)
    print(f"musique corpus: {len(corpus_df)} unique paragraphs -> {out_dir}/passages.csv")


def prep_legalbench_scalr():
    """LegalBench SCALR: 5-way MC over Supreme Court holdings (571 test items).

    Schema source (`nguha/legalbench`, scalr config):
      - index, question, choice_0..choice_4, answer (int 0-4)

    We rewrite to CaseHOLD-compatible schema:
      - idx, question, choice_a..choice_e, answer (letter A-E)
      so that the existing `format_casehold_prompt` and `extract_answer_mc5`
      can be reused verbatim.

    Also build a holdings corpus: union of all 5*571=2855 choice texts,
    deduped, suitable for `casehold_holdings`-style retrieval.
    """
    out_dir = "datasets/legalbench_scalr"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("nguha/legalbench", "scalr")["test"]

    # First pass: build holdings corpus with stable idx assignments.
    holdings = {}  # text -> idx for dedup
    for row in ds:
        for c in (row["choice_0"], row["choice_1"], row["choice_2"], row["choice_3"], row["choice_4"]):
            c = (c or "").strip()
            if c and c not in holdings:
                holdings[c] = f"scalr_holding_{len(holdings)}"

    # Second pass: write questions with gold_idx pointing to the corpus holding
    # that matches the correct displayed choice. Lets gold_retrieved become a
    # meaningful retrieval-quality metric (the corpus contains the gold holding).
    rows = []
    for row in ds:
        ans_int = int(row["answer"])
        gold_text = row[f"choice_{ans_int}"].strip()
        gold_idx = holdings.get(gold_text, "")
        rows.append({
            "idx": f"scalr_{row['index']}",
            "question": row["question"],
            "choice_a": row["choice_0"],
            "choice_b": row["choice_1"],
            "choice_c": row["choice_2"],
            "choice_d": row["choice_3"],
            "choice_e": row["choice_4"],
            "answer": chr(ord("A") + ans_int),
            "gold_idx": gold_idx,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"legalbench_scalr test: {len(df)} rows -> {out_dir}/test.csv")

    corpus_rows = [{"idx": idx, "text": text} for text, idx in holdings.items()]
    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_csv(os.path.join(out_dir, "holdings_corpus.csv"), index=False)
    print(f"legalbench_scalr holdings corpus: {len(corpus_df)} unique holdings -> {out_dir}/holdings_corpus.csv")


def prep_mleb_scalr():
    """MLEB-SCALR: retrieval-only SCALR packaging with corpus, queries, qrels."""
    out_dir = "datasets/mleb_scalr"
    os.makedirs(out_dir, exist_ok=True)

    corpus_ds = load_dataset("isaacus/mleb-scalr", "corpus")["corpus"]
    corpus_rows = []
    for row in corpus_ds:
        corpus_rows.append({
            "idx": row["_id"],
            "title": row.get("title", ""),
            "text": row["text"],
        })
    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_csv(os.path.join(out_dir, "corpus.csv"), index=False)
    print(f"mleb_scalr corpus: {len(corpus_df)} holdings -> {out_dir}/corpus.csv")

    query_ds = load_dataset("isaacus/mleb-scalr", "queries")["queries"]
    query_rows = []
    for row in query_ds:
        query_rows.append({
            "idx": row["_id"],
            "query": row["text"],
        })
    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(os.path.join(out_dir, "queries.csv"), index=False)
    print(f"mleb_scalr queries: {len(query_df)} queries -> {out_dir}/queries.csv")

    qrels_ds = load_dataset("isaacus/mleb-scalr", "default")["test"]
    qrel_rows = []
    for row in qrels_ds:
        qrel_rows.append({
            "query_id": row["query-id"],
            "doc_id": row["corpus-id"],
            "score": row.get("score", 1.0),
        })
    qrels_df = pd.DataFrame(qrel_rows)
    qrels_df.to_csv(os.path.join(out_dir, "qrels.csv"), index=False)
    print(f"mleb_scalr qrels: {len(qrels_df)} rows -> {out_dir}/qrels.csv")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("legal_rag", "all"):
        prep_legal_rag_qa()
    if target in ("australian", "all"):
        prep_australian_legal_qa()
    if target in ("casehold", "all"):
        prep_casehold()
    if target in ("musique", "all"):
        prep_musique()
    if target in ("legalbench_scalr", "scalr", "all"):
        prep_legalbench_scalr()
    if target in ("mleb_scalr", "mleb-scalr", "all"):
        prep_mleb_scalr()
