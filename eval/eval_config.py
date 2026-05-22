"""Evaluation configuration, question loading, and answer extraction.

Shared by eval_harness.py and eval_analyze.py.
"""
import os
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class EvalConfig:
    mode: str = "full_pipeline"       # key in EVAL_MODES; default kept for backward compatibility
    provider: str = "deepseek"        # any key from llm_config.PROVIDERS
    questions: str = "30"             # "curated" | "full" | integer N
    seed: int = 42
    skill_dir: str = "skills"
    verbose: bool = False
    tag: str = ""                     # optional label for the run
    source_filter: str = ""           # optional metadata filter, e.g. "mbe" to search MBE docs only
    dataset: str = "barexam"          # "barexam" | "housing" | "legal_rag" | "legal_rag_bench" | "mas_legal_bench" | "legal_link_eu" | "australian" | "casehold" | "musique" | "legalbench_scalr"
    embedding_model: str = ""         # override embedding model for retrieval (e.g., "BAAI/bge-m3")
    retrieval_k: int = 5              # final top-k after rerank for retrieval modes
    sample_start: int = 0             # optional slice start after deterministic sampling
    sample_end: int | None = None     # optional slice end after deterministic sampling
    hyre_cache_path: str = ""         # optional JSONL cache for replaying snap/HyRE generations
    retrieval_cache_path: str = ""    # optional JSONL cache of retrieved passage ids for top-k replay
    housing_state_filter: bool = False  # constrain HousingQA retrieval to the question state


EVAL_MODES = {
    "full_pipeline":       "Full agentic pipeline (planner → executor → synthesizer)",
    "llm_only":            "Direct LLM answer, no retrieval",
    "rag_rewrite":         "Query rewrite → retrieval → synthesize",
    "rag_simple":          "Raw question → retrieval → synthesize",
    "rag_state_filter":    "HousingQA state-filtered RAG: retrieve only statutes matching the question state, then synthesize",
    "golden_passage":      "LLM answer with gold passage injected as context",
    "golden_plus_neighbors": "Gold passage plus retrieved neighboring passages, for testing whether gold-only context is under-specified",
    "golden_arbitration":       "LLM answers naive, then reviews golden passage (neutral framing)",
    "golden_arb_conservative":  "LLM answers naive, then reviews golden passage (biased toward keeping)",
    "rag_arbitration":          "LLM answers naive, then reviews retrieved passages (conservative)",
    "rag_hyde":                 "HyDE: LLM generates hypothetical answer, embeds it to retrieve",
    "rag_hyde_exemplar":        "Probe-only HyDE with dataset-specific passage-style guidance, no answer evidence",
    "rag_hyde_arb":             "HyDE retrieval + snap-then-review arbitration (conservative)",
    "rag_multi_hyde":           "Multi-HyDE: 3 hypothetical passages (rule/exception/application)",
    "rag_snap_hyde":            "Snap-informed HyDE: answer first, then targeted retrieval",
    "rag_snap_hyde_1call":      "1-call ablation: retrieve on bare question (rag_simple style), then 1 LLM call producing snap reasoning + final answer (tests whether 2nd LLM call is necessary)",
    "snap_hyre":                "Snap-HyRE: snap reasoning + HyRE passage in one LLM call, then retrieve + final synth",
    "snap_hyre_exemplar":       "Probe-only Snap-HyRE with dataset-specific passage-style guidance, no retrieved exemplar evidence",
    "snap_choice_hyre":         "Choice-conditioned Snap-HyRE probe: one call predicts a primary/alternative and emits diverse candidate-theory HyRE passages, then retrieve + final synth",
    "rag_snap_hyde_2call":      "2-call snap+HyDE: snap reasoning + HyDE passage in one LLM call, then retrieve + final synth (efficiency variant of rag_snap_hyde)",
    "adaptive_snap_route":      "Bottleneck-adaptive routing: 1 LLM call produces snap + ROUTE (SUFFICIENT|NEEDS_RETRIEVAL) + HyDE; if SUFFICIENT return snap (1 call), else retrieve + synth (2 calls). Per-question bottleneck-aware variant of snap_hyde_2call.",
    "snap_hyde_aligned":        "Snap-HyDE aligned: HyDE for dense retrieval, raw question for cross-encoder reranking",
    "snap_hyre_option":         "HyRE option grounding: snap-conditioned HyDE retrieval plus an option-aware final synthesis prompt for MC holding tasks",
    "snap_hyre_state":          "HyRE state grounding: snap-conditioned HyDE retrieval with HousingQA state metadata filtering when state metadata is present",
    "adaptive_snap_hyre":       "Adaptive HyRE: one bottleneck-conditioned Snap-HyDE policy for legal RAG using option grounding, state filtering, or aligned HyDE reranking based on task shape",
    "adaptive_snap_hyre_anchor": "Adaptive HyRE + raw-question anchor: same legal bottleneck routing as adaptive_snap_hyre, but retrieves with both the HyRE passage and the original question text",
    "adaptive_snap_hyre_diverse": "Adaptive HyRE + diverse anchors: same legal bottleneck routing as adaptive_snap_hyre, but retrieves with HyRE passage, raw question, and sanitized snap reasoning",
    "adaptive_snap_hyre_v2":    "Adaptive HyRE v2: dataset/task-shaped legal controller using state+diverse anchors for Housing, option+diverse for CaseHOLD, plain 2-call Snap-HyDE for SCALR, and option grounding for BarExam",
    "adaptive_snap_hyre_frontier": "Adaptive HyRE frontier selector: audited N=200 frontier policy using BarExam v2, Housing/CaseHOLD diverse anchors, and SCALR plain 2-call Snap-HyDE",
    "adaptive_snap_hyre_stability": "Adaptive HyRE stability arbitration: run the frontier selector plus a dataset control, keep agreement, and arbitrate disagreements",
    "adaptive_snap_hyre_housing_verifier": "Adaptive HyRE Housing verifier: cached/diverse Housing retrieval with a conservative yes/no statutory entailment final prompt; non-Housing datasets fall back to the frontier selector",
    "adaptive_snap_hyre_candidate_verifier": "Adaptive HyRE candidate verifier: for CaseHOLD/SCALR, compare displayed holdings first and use retrieved evidence as support/tie-breaker; other datasets fall back to the frontier selector",
    "adaptive_snap_hyre_option_reranker": "Adaptive HyRE option reranker: CaseHOLD-focused per-candidate retrieval bundles before final holding selection; other datasets fall back to the frontier selector",
    "adaptive_snap_hyre_option_score": "Adaptive HyRE option score: CaseHOLD non-generative selector that chooses the candidate with the strongest candidate-conditioned retrieval score",
    "adaptive_snap_hyre_option_table": "Adaptive HyRE option table: CaseHOLD compact selector over per-candidate retrieval snippets and cross-encoder scores",
    "gap_hyde":                 "Gap-informed HyDE: snap + gaps + evidence in final (full context)",
    "gap_hyde_ev":              "Gap-informed HyDE: evidence only in final (no snap, no gap labels)",
    "gap_hyde_nosnap":          "Gap-informed HyDE: gaps + evidence in final (no snap answer)",
    "gap_hyde_flat":            "Gap-informed HyDE: snap + flat evidence in final (no gap structure)",
    "gap_rag":                  "Gap-informed RAG: snap + gaps + evidence in final (full context)",
    "gap_rag_nosnap":           "Gap RAG without snap in final — tests anchoring hypothesis",
    "gap_vectorless":           "Gap + historical 'vectorless' reasoning: per-gap generated knowledge reports, no corpus retrieval",
    "subagent_hyde":            "Subagent HyDE: per-gap HyDE retrieval + LLM summarization → reports only (no snap)",
    "subagent_rag":             "Subagent RAG: per-gap RAG + LLM summarization → reports only (no snap)",
    "subagent_hybrid":          "Subagent hybrid: per-gap RAG + LLM knowledge → combined reports (no snap)",
    "subagent_rag_evidence":    "Subagent RAG + evidence: reports + raw passages (no snap)",
    "subagent_rag_snap":        "Subagent RAG + snap: reports + snap answer in final (tests anchoring with reports)",
    "subagent_rag_full":        "Subagent RAG maximum info: reports + snap + raw passages in final",
    "snap_hyde_report":         "Snap-HyDE + summarization: snap_hyde retrieval → summarize → report only (no snap, no raw)",
    "snap_hyde_report_snap":    "Snap-HyDE + summarization + snap: report + snap answer in final",
    "snap_rag":                 "Snap + simple RAG: snap answer then retrieve with raw question, re-answer with both",
    "snap_rag_nosnap":          "Snap + simple RAG: snap then retrieve, but final call only sees evidence (control)",
    "vectorless_direct":        "Historical 'vectorless' reasoning: snap → generate doctrinal note from parametric knowledge → answer",
    "vectorless_role":          "Historical 'vectorless' reasoning: snap → role-conditioned parametric note (textbook/casebook/barprep via --tag) → answer",
    "vectorless_elements":      "Historical 'vectorless' reasoning: snap → identify dispositive legal elements → answer",
    "vectorless_choice_map":    "Historical 'vectorless' reasoning: snap → map rule + distractor + decisive fact → answer",
    "vectorless_nosnap":        "Historical 'vectorless' reasoning without snap: question → generate knowledge → answer (2-call snap ablation)",
    "vectorless_hybrid":        "Hybrid: generated parametric knowledge + vector RAG evidence pooled → answer (4 calls)",
    "vectorless_keyword":       "Historical 'vectorless' keyword baseline: snap → generate search terms → corpus retrieval → answer",
    "entity_search":            "Entity graph search: NLP inverted index → real corpus passages → cross-encoder rerank → answer (1 LLM call, zero embeddings)",
    "snap_entity_search":       "Snap + entity search: snap first, then entity graph corpus search, answer fresh without snap (2 LLM calls)",
    "snap_entity_informed":     "Snap-informed entity search: extract entities from snap reasoning + question for better search terms (2 LLM calls)",
    "rag_devil_hyde":           "Devil's advocate HyDE: retrieve for AND against snap answer",
    "rag_top2_hyde":            "Top-2 HyDE: retrieve for snap answer + second-choice answer",
    "confidence_gated":         "Confidence-gated: 3 snap votes, unanimous=skip RAG, disagreement=Snap-HyDE",
    "decompose":                "Decompose-then-answer: split into sub-questions, answer each, synthesize (no RAG)",
    "decompose_rag":            "Decompose + Snap-HyDE: sub-questions with per-issue retrieval, then synthesize",
    "ce_threshold":             "CE-thresholded Snap-HyDE: discard low-scoring evidence, fall back to snap answer",
    "conf_ce_threshold":        "Confidence-gated + CE threshold: 3-vote gating, then CE threshold on RAG path",
    "snap_hyde_aspect":         "Snap-HyDE + aspect queries: HyDE passage + rule/exception queries for diverse retrieval",
    "ce_threshold_k3":          "CE-thresholded Snap-HyDE with k=3: fewer passages, higher quality",
    "self_verify":              "Self-verification: snap answer then review for errors (2 calls, no RAG)",
    "double_snap":              "Double-snap: two answers, agree=use, disagree=CE-threshold RAG (2-4 calls)",
    "snap_debate":              "Snap-debate: snap then adversarial critique (2 calls, no RAG)",
    "snap_only_in_final":       "Ablation cell: snap reasoning visible to final agent, NO retrieval (2 calls)",
    "planning_table":           "Snap → plan TODOs → per-TODO retrieve+finding → final with populated table as scratchpad (≈5-7 calls)",
    "planning_table_no_snap":   "Ablation of planning_table: TODOs generated from QUESTION ALONE (no snap). Tests if snap-bias is the multi-hop failure source",
    "rag_multi_query":          "Multi-query rag_simple: 2 question rewrites + original → pool retrievals → answer once. Tests if retrieval diversity alone beats single-query (no snap, no per-hop)",
    "iterative_planning_table": "Multi-round planning_table (deep research style): generate ONE focused TODO per round, retrieve + find, then decide READY-or-NEXT-TODO. Up to 3 rounds. Each next TODO is conditioned on prior findings.",
    "advisor_planning_table":   "Two-LLM advisor pattern: cheap LLM (Llama 8B) does plan + per-TODO findings; STRONG LLM (config.provider) does final synthesis. Tests if allocating reasoning capacity to synthesis (vs intermediates) helps.",
    "multi_hyde_diverse":       "Multi-HyDE diverse: 3 candidate hypothetical answer-passages with DIFFERENT entities/angles, pool retrievals across all + raw question. Targets the single-hop commitment bias on multi-hop QA (HyDE commits to one wrong entity → biased retrieval).",
    "iter_hyde":                "Iterative HyDE (multi-round): generate ONE HyDE passage per round conditioned on prior findings, retrieve, write finding, then decide READY-or-NEXT-HYDE (max 3 rounds). HyDE-style analog of iterative_planning_table. Targets multi-hop COMPOSITION bottleneck (not just retrieval coverage like multi_hyde_diverse).",
    "friend_foe_attribution":   "Friend/foe attribution probe: snap answer + three review passes (self / foe / control) on SAME snap content. Tests whether synth applies different scrutiny based on attribution string. 4 LLM calls per question, no retrieval.",
}


def load_questions(config: EvalConfig) -> pd.DataFrame:
    """Load questions based on config.questions: 'curated', 'full', or integer N."""
    if config.dataset == "housing":
        return _load_housing_questions(config)
    if config.dataset == "legal_rag":
        return _load_generic_questions(config, "datasets/legal_rag_qa/questions.csv")
    if config.dataset == "legal_rag_bench":
        return _load_generic_questions(config, "datasets/legal_rag_bench/questions.csv")
    if config.dataset == "mas_legal_bench":
        return _load_generic_questions(config, "datasets/mas_legal_bench/questions.csv")
    if config.dataset == "legal_link_eu":
        return _load_generic_questions(config, "datasets/legal_link_eu/questions.csv")
    if config.dataset == "australian":
        return _load_generic_questions(config, "datasets/australian_legal_qa/questions.csv")
    if config.dataset == "casehold":
        return _load_generic_questions(config, "datasets/casehold/test.csv")
    if config.dataset == "musique":
        return _load_generic_questions(config, "datasets/musique/questions.csv")
    if config.dataset == "legalbench_scalr":
        return _load_generic_questions(config, "datasets/legalbench_scalr/test.csv")

    if config.questions == "curated":
        path = os.path.join(os.path.dirname(__file__), "question_sets", "curated_30.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Curated question set not found at {path}. "
                "Run eval/curate_questions.py first to generate it."
            )
        return pd.read_csv(path)

    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")

    if config.questions == "full":
        return qa.reset_index(drop=True)

    n = int(config.questions)
    return qa.sample(n=min(n, len(qa)), random_state=config.seed).reset_index(drop=True)


def _load_housing_questions(config: EvalConfig) -> pd.DataFrame:
    """Load HousingQA questions (Yes/No format)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa = pd.read_csv(os.path.join(base, "datasets/housing_qa/questions.csv"))

    if config.questions == "full":
        return qa.reset_index(drop=True)

    n = int(config.questions)
    return qa.sample(n=min(n, len(qa)), random_state=config.seed).reset_index(drop=True)


def _load_generic_questions(config: EvalConfig, csv_path: str) -> pd.DataFrame:
    """Load questions from a CSV, sample N if requested."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa = pd.read_csv(os.path.join(base, csv_path))

    if config.questions == "full":
        return qa.reset_index(drop=True)

    n = int(config.questions)
    return qa.sample(n=min(n, len(qa)), random_state=config.seed).reset_index(drop=True)


def extract_answer_mc(text: str) -> str | None:
    """Extract multiple-choice answer letter (A-D) from LLM response.

    Audit 2026-04-26 caught: housing extractor falls back to last
    standalone Y/N when no explicit "Answer:" line; MC extractor had
    no fallback → silent FAIL on prose-style answers like "the answer
    is C" without the marker. Added a "last standalone A-D letter"
    fallback gated to runs without an explicit Answer block.
    """
    cleaned = (text or "").replace('*', '')
    patterns = [
        r'(?:Answer|ANSWER)[:\s]*\(?([A-D])\)?',
        r'\b([A-D])\b\s*(?:is correct|is the (?:best|correct|strongest))',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            return matches[-1]  # last match = conclusion
    # Fallback: last standalone A-D letter anywhere in the text
    matches = re.findall(r'\b([A-D])\b', cleaned)
    if matches:
        return matches[-1]
    return None


def extract_answer_mc5(text: str) -> str | None:
    """Extract 5-way multiple-choice answer letter (A-E) from LLM response.

    Same "last standalone letter" fallback as extract_answer_mc.
    """
    cleaned = (text or "").replace('*', '')
    patterns = [
        r'(?:Answer|ANSWER)[:\s]*\(?([A-E])\)?',
        r'\b([A-E])\b\s*(?:is correct|is the (?:best|correct|strongest))',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            return matches[-1]
    matches = re.findall(r'\b([A-E])\b', cleaned)
    if matches:
        return matches[-1]
    return None


def extract_answer_yn(text: str) -> str | None:
    """Extract Yes/No answer from LLM response."""
    cleaned = text.replace('*', '').strip()
    patterns = [
        r'(?:Answer|ANSWER)[:\s]*(Yes|No)\b',
        r'(?:Final answer|FINAL ANSWER)[:\s]*(Yes|No)\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        if matches:
            return matches[-1].capitalize()
    # Fallback: last standalone Yes/No in the text
    matches = re.findall(r'\b(Yes|No)\b', cleaned, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    return None


def format_question_prompt(row: pd.Series, dataset: str = "barexam") -> str:
    """Format a question into a standard prompt string."""
    if dataset == "housing":
        return format_housing_prompt(row)
    if dataset == "casehold":
        return format_casehold_prompt(row)
    if dataset == "legalbench_scalr":
        return format_casehold_prompt(row)  # same 5-way MC schema as CaseHOLD
    if dataset == "mas_legal_bench":
        return format_mas_legal_bench_prompt(row)
    if dataset == "legal_link_eu":
        return format_legal_link_eu_prompt(row)
    if dataset in ("legal_rag", "legal_rag_bench", "australian"):
        return format_open_prompt(row)
    if dataset == "musique":
        return format_musique_prompt(row)

    # Many BarExam items share a fact pattern across multiple sub-questions
    # (same prompt_id). The 'prompt' column carries that shared fact pattern;
    # without it, 37% of questions are missing their context entirely.
    parts = []
    prompt = row.get("prompt", "")
    if pd.notna(prompt) and str(prompt).strip():
        parts.append(str(prompt).strip())
    parts.append(str(row["question"]))

    choices = []
    for letter in ["A", "B", "C", "D"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]):
            choices.append(f"  ({letter}) {row[col]}")

    if choices:
        parts.append("\n".join(choices))

    parts.append("\nProvide your answer as: Answer: (X)")
    return "\n\n".join(parts)


def format_housing_prompt(row: pd.Series) -> str:
    """Format a HousingQA Yes/No question into a prompt string."""
    state = str(row.get("state", ""))
    question = str(row["question"])
    prompt = f"Regarding {state} housing law:\n\n{question}"
    prompt += "\n\nAnswer Yes or No. Provide your answer as: Answer: Yes or Answer: No"
    return prompt


def format_casehold_prompt(row: pd.Series) -> str:
    """Format a CaseHOLD 5-way MC question."""
    context = str(row["question"])  # 'question' col holds citing context
    choices = []
    for letter in ["A", "B", "C", "D", "E"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]):
            choices.append(f"  ({letter}) {row[col]}")

    prompt = (
        f"The following excerpt from a court opinion cites a legal holding. "
        f"Which of the following holdings is most likely being referenced?\n\n"
        f"## Citing Context\n{context}\n\n"
        f"## Holdings\n" + "\n".join(choices) +
        f"\n\nProvide your answer as: Answer: (X)"
    )
    return prompt


def format_mas_legal_bench_prompt(row: pd.Series) -> str:
    """Format a MASLegalBench four-way GDPR/legal reasoning MC question."""
    question = str(row["question"])
    choices = []
    for letter in ["A", "B", "C", "D"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            choices.append(f"  ({letter}) {row[col]}")

    return (
        "Answer the following legal question using the provided choices.\n\n"
        f"## Question\n{question}\n\n"
        "## Choices\n"
        + "\n".join(choices)
        + "\n\nProvide your answer as: Answer: (X)"
    )


def format_legal_link_eu_prompt(row: pd.Series) -> str:
    """Format a Legal-Link-EU four-way MC question."""
    question = str(row["question"])
    relation = str(row.get("relation_type", "") or "").replace("_", " ")
    choices = []
    for letter in ["A", "B", "C", "D"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            choices.append(f"  ({letter}) {row[col]}")

    prefix = "Answer the following EU legal authority question"
    if relation:
        prefix += f" about {relation}"
    return (
        f"{prefix} using the provided choices.\n\n"
        f"## Question\n{question}\n\n"
        "## Choices\n"
        + "\n".join(choices)
        + "\n\nProvide your answer as: Answer: (X)"
    )


def format_open_prompt(row: pd.Series) -> str:
    """Format an open-ended legal question (legal-rag-qa, australian)."""
    question = str(row["question"])
    return f"{question}\n\nProvide a detailed answer."


def format_musique_prompt(row: pd.Series) -> str:
    """Format a MuSiQue multi-hop QA question.

    Short-answer prompt — the model should produce a single span/entity, not
    a paragraph. We keep the framing tight so EM/F1 against the gold span
    works without judging every minor wording difference.

    Placeholder history:
      - Originally `<span>` — Llama 3.3 70b echoed `Answer: <span>Foo</span>`
        (commit 97c204a fixed the extractor to strip wrapping HTML tags).
      - Then `<your answer here>` — Gemma 4 26B echoed the literal placeholder
        text on a 2026-04-26 multi_hyde_diverse N=30 run.
      - Now `[your answer here]` (square brackets) — conventional, neither
        model echoes square-bracket placeholders.
    Extractor also defensively strips wrapping HTML-like tags.
    """
    question = str(row["question"])
    return (
        f"{question}\n\n"
        "Answer with a brief span — a single entity, date, or short phrase. "
        "Provide your answer in the exact form: Answer: [your answer here]"
    )


def extract_answer_musique(text: str) -> str:
    """Extract the short-answer span after 'Answer:'.

    Open-ended; we return the raw span text and let the EM/F1 scorer
    handle alias matching downstream. Defensively strips HTML-like wrapping
    tags (`<span>...</span>`, `<answer>...</answer>`) some instruction-tuned
    models emit when the prompt mentions a placeholder.
    """
    cleaned = (text or "").replace("*", "")
    m = re.search(r"(?:Answer|ANSWER)\s*[:\s]\s*(.+?)(?:\n|$)", cleaned)
    span = m.group(1).strip().rstrip(".").strip() if m else (
        cleaned.strip().splitlines()[-1].rstrip(".").strip() if cleaned.strip() else ""
    )
    # Strip wrapping HTML-like tags: <span>X</span>, <answer>X</answer>, etc.
    span = re.sub(r"^\s*<[a-zA-Z_][a-zA-Z0-9_]*>\s*", "", span)
    span = re.sub(r"\s*</[a-zA-Z_][a-zA-Z0-9_]*>\s*$", "", span)
    return span.strip()


def musique_em_f1(predicted: str, gold: str, aliases: list[str] | None = None) -> tuple[bool, float]:
    """SQuAD/MuSiQue-style EM and F1 over normalized tokens.

    Compares predicted against (gold + aliases). Returns (em, f1) where
    em is True if any of the gold/alias forms matches exactly after
    normalization, and f1 is the maximum token-overlap F1 across all
    accepted forms.
    """
    def norm(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    candidates = [gold] + list(aliases or [])
    pred_n = norm(predicted)
    pred_toks = pred_n.split()
    if not pred_toks:
        return False, 0.0

    em = False
    best_f1 = 0.0
    for cand in candidates:
        cand_n = norm(cand)
        if cand_n == pred_n:
            em = True
        cand_toks = cand_n.split()
        if not cand_toks:
            continue
        common = set(pred_toks) & set(cand_toks)
        if not common:
            continue
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(cand_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return em, best_f1
