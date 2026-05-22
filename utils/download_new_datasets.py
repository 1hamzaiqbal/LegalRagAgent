"""Download and prepare supplemental legal QA/RAG datasets."""
import hashlib
import json
import os
import sys
import urllib.request

import pandas as pd
from datasets import Dataset, load_dataset


def stable_holding_id(text: str, prefix: str = "holding") -> str:
    normalized = " ".join(str(text or "").split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


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

    holding_sources: dict[str, set[str]] = {}
    holding_text: dict[str, str] = {}
    for split in ["test", "train"]:
        rows = []
        for i, row in enumerate(ds[split]):
            endings = row["endings"]
            label = row["label"]
            answer = chr(ord("A") + label)
            gold_text = endings[label]
            gold_idx = stable_holding_id(gold_text, prefix="casehold")
            for ending in endings:
                idx = stable_holding_id(ending, prefix="casehold")
                holding_text.setdefault(idx, ending)
                holding_sources.setdefault(idx, set()).add(split)
            rows.append({
                "idx": f"ch_{split}_{i}",
                "question": row["context"],
                "choice_a": endings[0],
                "choice_b": endings[1],
                "choice_c": endings[2],
                "choice_d": endings[3],
                "choice_e": endings[4],
                "answer": answer,  # 0->A, 1->B, etc.
                "gold_idx": gold_idx,
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        print(f"casehold {split}: {len(df)} rows -> {out_dir}/{split}.csv")

    # Build a holdings corpus from all displayed train/test options. This makes
    # answer-choice retrieval diagnostics meaningful: test rows now have a
    # stable gold_idx that can be checked against retrieved_ids.
    corpus_rows = []
    for idx, text in sorted(holding_text.items()):
        corpus_rows.append({
            "idx": idx,
            "text": text,
            "source": "casehold_" + "+".join(sorted(holding_sources[idx])),
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


def prep_mas_legal_bench():
    """MASLegalBench: GDPR legal reasoning with provided context rows.

    Upstream packages all data in one Arrow split with mixed row types:
    questions and context items. The paper/code evaluates exact answer accuracy
    after retrieving context rows from the same case source. For our harness we
    keep the main question file to the four-way MC subset only, because the
    comprehensive grid assumes lettered choices for MC benchmarks. The Yes/No
    rows are retained in ``questions_all.csv`` for later if we want a separate
    binary-answer variant.

    Corpus note: this is an independent context corpus, not answer choices. It
    contains background/legal-framework/entity/relation/inferred-alignment rows.
    MASLegalBench does not provide per-question gold passage IDs, so
    ``gold_idx`` is intentionally left empty; downstream logs can still report
    same-source evidence exposure from passage metadata.
    """
    out_dir = "datasets/mas_legal_bench"
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    raw_arrow = os.path.join(raw_dir, "data-00000-of-00001.arrow")
    if not os.path.exists(raw_arrow):
        url = (
            "https://raw.githubusercontent.com/HKUST-KnowComp/MASLegalBench/"
            "main/dataset/train/data-00000-of-00001.arrow"
        )
        print(f"downloading MASLegalBench Arrow from {url}")
        urllib.request.urlretrieve(url, raw_arrow)

    ds = Dataset.from_file(raw_arrow)

    def type_slug(value: str) -> str:
        return "_".join(str(value or "").strip().lower().split())

    context_rows = []
    question_rows_all = []
    source_to_context_ids: dict[str, list[str]] = {}
    seen_context_ids: set[str] = set()
    question_counter: dict[str, int] = {}

    for row in ds:
        source = str(row.get("source") or "").strip()
        row_type = str(row.get("type") or "").strip()
        content = str(row.get("content") or "")
        if row_type == "question":
            payload = json.loads(content)
            options = payload.get("options") or {}
            if not isinstance(options, dict):
                options = {}
            qn = question_counter.get(source, 0)
            question_counter[source] = qn + 1
            idx = f"maslb_{hashlib.sha1((source + ':q:' + str(qn)).encode()).hexdigest()[:16]}"

            # Preserve native A-D/Yes-No labels in the all-questions export.
            row_out = {
                "idx": idx,
                "source": source,
                "question": payload.get("question", ""),
                "answer": str(payload.get("correct_answer", "")).strip(),
                "whether_contains_decision": payload.get("whether_contains_decision", ""),
                "option_count": len(options),
                "gold_idx": "",
                "source_context_ids": "",
            }
            for label, text in options.items():
                label_norm = str(label).strip()
                if len(label_norm) == 1 and label_norm.upper() in {"A", "B", "C", "D"}:
                    row_out[f"choice_{label_norm.lower()}"] = text
                elif label_norm.lower() == "yes":
                    row_out["choice_a"] = "Yes"
                elif label_norm.lower() == "no":
                    row_out["choice_b"] = "No"
            question_rows_all.append(row_out)
            continue

        context_type = type_slug(row_type)
        digest = hashlib.sha1(
            (source + "\n" + context_type + "\n" + content).encode("utf-8", errors="ignore")
        ).hexdigest()[:20]
        idx = f"maslb_ctx_{digest}"
        if idx in seen_context_ids:
            continue
        seen_context_ids.add(idx)
        context_rows.append({
            "idx": idx,
            "text": content,
            "source": source,
            "context_type": context_type,
        })
        source_to_context_ids.setdefault(source, []).append(idx)

    for row in question_rows_all:
        ids = source_to_context_ids.get(str(row.get("source") or ""), [])
        row["source_context_ids"] = ",".join(ids)

    qa_all = pd.DataFrame(question_rows_all)
    qa_mc = qa_all[qa_all["option_count"] == 4].copy()

    corpus_df = pd.DataFrame(context_rows)
    corpus_df.to_csv(os.path.join(out_dir, "passages.csv"), index=False)
    qa_all.to_csv(os.path.join(out_dir, "questions_all.csv"), index=False)
    qa_mc.to_csv(os.path.join(out_dir, "questions.csv"), index=False)

    print(f"MASLegalBench corpus: {len(corpus_df)} context rows -> {out_dir}/passages.csv")
    print(f"MASLegalBench questions_all: {len(qa_all)} rows -> {out_dir}/questions_all.csv")
    print(f"MASLegalBench questions.csv: {len(qa_mc)} four-way MC rows -> {out_dir}/questions.csv")


def prep_legal_link_eu():
    """Legal-Link-EU: four-way MC over provided EUR-Lex evidence contexts.

    The dataset ships row-level evidence, not a separate global corpus. We build
    a fixed retrieval corpus from the original ``contexts`` field only and keep
    the adversarial ``perturbed_contexts`` out of the default corpus. Gold
    retrieval labels are the five original context IDs for each question.
    """
    out_dir = "datasets/legal_link_eu"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("disi-unibo-nlp/legal-link-eu", split="test")
    relation_types = [
        "extends_application",
        "rendered_obsolete_by",
        "implicitly_repeals",
        "extends_validity",
        "completes",
        "corrects",
        "repeals",
    ]

    def parse_id(example_id: str) -> tuple[str, str, str]:
        for relation in relation_types:
            suffix = "_" + relation
            if example_id.endswith(suffix):
                pair = example_id[: -len(suffix)].removeprefix("complex_legallink_")
                parts = pair.split("_", 1)
                if len(parts) == 2:
                    return parts[0], parts[1], relation
                return "", "", relation
        return "", "", "unknown"

    def stable_context_id(title: str, text: str) -> str:
        digest = hashlib.sha1((title + "\n" + text).encode("utf-8", errors="ignore")).hexdigest()[:20]
        return f"lle_ctx_{digest}"

    def celex_from_title(title: str) -> str:
        if "(" in title and title.endswith(")"):
            return title[title.find("(") + 1:-1]
        return title

    passages: dict[str, dict] = {}
    qa_rows = []
    perturbed_rows = []

    for row_number, row in enumerate(ds):
        example_id = str(row["id"])
        question_id = f"{example_id}__row{row_number:04d}"
        source_doc, target_doc, relation = parse_id(example_id)
        gold_ids = []
        gold_texts = []
        for position, (title, text) in enumerate(zip(row["context_titles"], row["contexts"])):
            title = str(title)
            text = str(text)
            if not text.strip():
                perturbed_rows.append({
                    "idx": f"{question_id}_perturbed_{position}",
                    "example_id": example_id,
                    "question_idx": question_id,
                    "context_title": title,
                    "text": row["perturbed_contexts"][position],
                    "relation_type": relation,
                    "original_context_empty": True,
                })
                continue
            passage_id = stable_context_id(title, text)
            gold_ids.append(passage_id)
            gold_texts.append(f"## {title}\n{text}")
            if passage_id not in passages:
                role = "source" if title.startswith("Source") else "target" if title.startswith("Target") else "context"
                celex_id = celex_from_title(title)
                passages[passage_id] = {
                    "idx": passage_id,
                    "text": text,
                    "title": title,
                    "source": celex_id,
                    "citation": celex_id,
                    "role": role,
                    "context_title": title,
                }
            perturbed_rows.append({
                "idx": f"{question_id}_perturbed_{position}",
                "example_id": example_id,
                "question_idx": question_id,
                "context_title": title,
                "text": row["perturbed_contexts"][position],
                "relation_type": relation,
                "original_context_empty": False,
            })

        options = list(row["options"])
        qa_rows.append({
            "idx": question_id,
            "example_id": example_id,
            "question": row["question"],
            "choice_a": options[0],
            "choice_b": options[1],
            "choice_c": options[2],
            "choice_d": options[3],
            "answer": row["correct_label"],
            "correct_index": row["correct_index"],
            "gold_idx": ",".join(gold_ids),
            "gold_passage": "\n\n".join(gold_texts),
            "relation_type": relation,
            "source_doc": source_doc,
            "target_doc": target_doc,
            "subject": relation,
        })

    passages_df = pd.DataFrame(list(passages.values())).sort_values("idx")
    qa_df = pd.DataFrame(qa_rows)
    perturbed_df = pd.DataFrame(perturbed_rows)

    passages_df.to_csv(os.path.join(out_dir, "passages.csv"), index=False)
    qa_df.to_csv(os.path.join(out_dir, "questions.csv"), index=False)
    perturbed_df.to_json(os.path.join(out_dir, "perturbed_contexts.jsonl"), orient="records", lines=True)

    print(f"Legal-Link-EU corpus: {len(passages_df)} unique context rows -> {out_dir}/passages.csv")
    print(f"Legal-Link-EU questions: {len(qa_df)} rows -> {out_dir}/questions.csv")
    print(f"Legal-Link-EU perturbed contexts: {len(perturbed_df)} rows -> {out_dir}/perturbed_contexts.jsonl")


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
    if target in ("mas_legal_bench", "maslegalbench", "mas", "all"):
        prep_mas_legal_bench()
    if target in ("legal_link_eu", "legallink", "legal-link-eu", "lle", "all"):
        prep_legal_link_eu()
