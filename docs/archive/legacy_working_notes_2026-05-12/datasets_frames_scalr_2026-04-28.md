# Dataset Scoping: FRAMES and SCALR (MLEB)

Author: Claude scoping pass, 2026-04-28
Inputs: HuggingFace dataset cards, FRAMES arXiv (2409.12941), LegalBench SCALR task page, MLEB paper (2510.19365), and the team's `eval/eval_config.py` + `utils/fast_embed.py` patterns.

Out-of-scope: no evaluations were run, no code was modified.

---

## 1. TL;DR

- **SCALR (MLEB version) is the easier add — but it is a pure retrieval benchmark, not a QA benchmark, so it does not slot into the existing MC/Yes-No/short-span answer extractors.** 120 queries against a fixed 523-document holding corpus, parquet on HF, CC BY 4.0, no gating.
- **FRAMES is the more taxonomically-interesting add — it is the closest off-the-shelf analog to MuSiQue at a harder difficulty (824 multi-hop questions, oracle≈73% vs. naive zero-shot≈41% on Gemini-Pro-1.5)** but it ships **no corpus** — only Wikipedia URLs per question. To run RAG against it we either (a) scrape Wikipedia on the fly, (b) embed a Wikipedia snapshot, or (c) accept that corpus build is the bottleneck.
- **Recommended order: SCALR first as a 1-day side-quest** (only meaningful as a retrieval-quality probe, not a QA accuracy probe), **then FRAMES as a multi-week effort** with a deliberately scoped corpus strategy. FRAMES is a strict superset-in-difficulty of MuSiQue and pairs naturally with the bottleneck taxonomy (it is explicitly billed as factuality + retrieval + reasoning).
- Note: There are TWO SCALRs. **MLEB-SCALR (the one the team listed) is retrieval (queries + corpus + qrels)**. The older **LegalBench-SCALR is 5-way MC over 571 items** (the same items, packaged differently). LegalBench-SCALR would actually fit the harness more naturally — see Risk Register.

---

## 2. FRAMES — `google/frames-benchmark`

### 2.1 What it actually is
- **Format**: open-ended short/medium answer (gold answers range 1 to ~1.36k chars). Official metric is **LLM-as-judge** (Gemini auto-rater; Cohen's κ=0.889 vs. human).
- **Question count**: **824** (single `test` split; CSV → auto-converted parquet on HF).
- **Hop count**: explicitly multi-hop — most items list 2-11+ Wikipedia articles required. Reasoning types tagged include "Multiple constraints", "Numerical reasoning", "Tabular reasoning", "Temporal reasoning", "Post processing".
- **Language**: English only.
- **License**: **Apache 2.0**.
- **Publisher**: Google + Harvard, Sept 2024 (arXiv 2409.12941, "Fact, Fetch, and Reason").

### 2.2 What corpus it ships
- **No bundled retrieval corpus.** Each row carries `wikipedia_link_1` … `wikipedia_link_11+` columns (URLs as strings) and an aggregated `wiki_links` column.
- Format: pure URL strings. **The team owns the corpus problem.** Three realistic options:
  1. **Per-question Wikipedia scrape** (offline-cache the linked articles only). ~824 × ~5 articles ≈ ~4k unique Wikipedia pages; downloadable in <1 hr but copyright-clean depends on storage strategy (Wikipedia is CC BY-SA 4.0 — citation required).
  2. **Embed a Wikipedia snapshot** (e.g., `wikipedia_passages` from KILT or DPR-Wiki, ~21M passages). This dwarfs `housing_statutes` (1.84M) and would be the largest collection in the project. ~1+ days GPU time on 3070.
  3. **MuSiQue-style "oracle plus distractors"**: only embed the linked pages + a small distractor pool. Compromise between (1) and the team's existing per-question-passage pattern.
- The FRAMES paper itself reports BM25 over Wikipedia and an oracle-gold-passage condition, so option (3) cleanly reproduces their setup.

### 2.3 Bottleneck-taxonomy fit
- **Mixed, leaning retrieval-bottlenecked at the high end.** Reported numbers (Gemini-Pro-1.5):
  - Zero-shot (no retrieval): **0.408**
  - BM25 single-step RAG (k=4): 0.474 (+6.6pp)
  - Multi-step iterative retrieval: 0.66 (+25pp over zero-shot)
  - Oracle gold passages: **0.729** (the ceiling)
- The 25-pp gap between single-step RAG and multi-step retrieval is the largest delta among any benchmark the team currently runs — this is exactly the regime where `multi_hyde_diverse` and `iter_hyde` should win, and where `rag_simple` will badly underperform.
- **Reasoning types are explicit metadata**, so the team can slice results by reasoning-type cluster — directly useful for the bottleneck-taxonomy paper.
- Pre-classification: **retrieval-bottlenecked + composition-bottlenecked**. Naive single-step retrieval leaves ~25pp on the table even on a strong model, which means it tests both retrieval coverage (the multi-hop coverage problem `multi_hyde_diverse` targets) and composition over retrieved facts (what `iter_hyde` targets). Closer cousin to MuSiQue than to BarExam.

### 2.4 Implementation effort against `eval/eval_harness.py`
Patterns the team already has (verified in `eval/eval_config.py`):

| Need | Existing pattern to clone | Effort |
|---|---|---|
| `load_frames_questions` | `_load_generic_questions` (line 137) reads `datasets/frames/questions.csv`. Direct reuse — just dump HF parquet to CSV. | ~30 min |
| `format_frames_prompt` | Closest analog is `format_musique_prompt` (line 279) — short-span placeholder framing. **But FRAMES gold answers are longer (1-1.36k chars) and LLM-as-judge is the official metric.** Need a paragraph-length open-ended formatter (closer to `format_open_prompt`) plus a judge call. | ~2 hr (formatter + judge prompt) |
| Answer extractor | **Open-ended; no clean string match.** Need an LLM-as-judge function analogous to `extract_answer_musique` + `musique_em_f1`. The paper's auto-rater prompt is published in their Fig 6 — reuse verbatim. ~1 LLM call per question per run. Adds ~824 calls per N=full eval — within Groq llama70b 1K RPD limit but eats budget fast. | ~3 hr (prompt + plumbing + judge-provider config) |
| `EVAL_MODES` registration | Modes are dataset-agnostic in this harness — no new mode needed; `--dataset frames --mode rag_simple` should just work once loader/formatter/extractor are wired. | 0 hr |
| `CORPORA` entry in `fast_embed.py` | Depends on corpus strategy. **Option (3) (linked-pages-only)** is a clean fit: dump scraped paragraphs to `datasets/frames/passages.csv` with `idx`/`text` columns, add a `frames` entry to the dict. Embedding ~4k Wikipedia pages chunked = trivial (<10 min on the 3070). | ~4 hr (scrape + chunk + dedupe + CSV) for option 3; ~1 day for option 1; ~2 days for option 2 |
| Per-question gold passage hookup | The harness has `golden_passage` mode (line 32). FRAMES has explicit per-question links → easy to populate a `golden_idxs` column on the questions CSV. **This gives an oracle ceiling identical to the paper's 0.729.** | ~1 hr |
| `dataset` enum extension | Add `"frames"` to the `EvalConfig.dataset` docstring (line 22) and to the `load_questions` / `format_question_prompt` dispatchers. | ~15 min |

**Total effort estimate: 1.5-2 days for option (3) corpus strategy** end-to-end (loader + formatter + LLM-judge + scraper + corpus build + smoke-test on N=30). This excludes any methodology work to validate the LLM-as-judge against a held-out human spot-check (recommended ~30 questions, ~1 hr).

### 2.5 Cluster vs. local feasibility
- **Questions**: yes, single CSV (824 rows). Trivially loadable.
- **Corpus**: option (3) is local-feasible (~4k pages, <10 min GPU). Option (2) is a cluster job (Wikipedia snapshot).
- **Per-question passages**: yes, FRAMES ships URLs that map directly to a per-question gold-passage list — much more like MuSiQue than like BarExam (no shared corpus needed if option 3).

### 2.6 Anything weird
- **No bundled corpus** is the headline gotcha — every other team that uses FRAMES has to make the same Wikipedia decision.
- **LLM-as-judge is the official metric**, so EM-on-a-string will under-report — the team's existing `extract_answer_musique` is not appropriate. Without a judge, FRAMES numbers will not be comparable to the paper.
- Wikipedia content is **CC BY-SA 4.0** (attribution + share-alike) — anything published from FRAMES retrieval needs Wikipedia citation in any output corpus dump.
- Schema has a `Unnamed: 0` column (raw CSV index) — drop on load.
- `Prompt`/`Answer` are capitalized (BarExam uses lowercase `question`/`prompt`); rename or accommodate in the loader.
- The MarkTechPost coverage credits Google + Harvard. arXiv v3 is the current canonical reference.
- Public leaderboard (llm-stats.com/benchmarks/frames) currently shows Kimi K2-Thinking-0905 = 0.870 (SOTA) and DeepSeek-V3 = 0.733 — sparse data, but **0.40 zero-shot / 0.66-0.73 with retrieval is the realistic operating range for Llama 3.3 70b**.

---

## 3. SCALR — `isaacus/mleb-scalr`

### 3.1 What it actually is
- **Format**: pure information retrieval (BEIR-style queries + corpus + qrels). **Not QA.**
- **Question count**: **120 queries** in `test` split.
- **Corpus size**: **523 holdings** (single shared corpus split).
- **Qrels** (default split): query-id ↔ corpus-id ↔ score(=1).
- **Language**: English. Jurisdiction: US (Supreme Court holdings, 2001 Term onward).
- **License**: **CC BY 4.0**, ungated, public.
- **Publisher**: Isaacus, packaged as part of MLEB (Massive Legal Embeddings Benchmark, paper 2510.19365, Oct 2025). The underlying SCALR data is older — from the LegalBench paper (2308.11462, Aug 2023) and the lexeme-dev/scalr GitHub repo.

### 3.2 What corpus it ships
- **Yes** — corpus is bundled. 523 holding texts (parquet, also JSON-convertible).
- Schema: `_id`, `text`, `title` (empty string). BEIR-compatible.
- Single shared corpus across all 120 queries. Like CaseHOLD but tiny (50,291 holdings → 523).

### 3.3 Bottleneck-taxonomy fit
- **Pure retrieval-bottlenecked, but it does NOT test reasoning** — there is no answer to extract, no model output to grade beyond retrieval rank.
- This is fundamentally a **different evaluation surface** than the harness was built for. It tests embedding/retriever quality directly. The team's `eval/run_embedding_comparison.py` is a closer fit than `eval_harness.py`.
- It would be a good probe for the **embedding-model A/B story** (which `fast_embed.py` already supports via `EMBEDDING_MODELS`) — does `gte-large` vs. `bge-m3` vs. `legal-bert` measurably differ on legal-reasoning retrieval?
- **Within the bottleneck taxonomy it is "retrieval-only-bottlenecked"** — a degenerate endpoint of the spectrum. Useful as a calibration point ("how good is our retriever?"), not as a QA evaluation.
- **The LegalBench-SCALR (5-way MC, 571 items, GPT-4 = 81.4%, human = 86%) IS what fits the QA harness** — see Risk Register.

### 3.4 Implementation effort
**The harness is built around question-level QA, not BEIR-style retrieval.** Wiring in MLEB-SCALR as a QA dataset requires either:

**Option A — Shoe-horn into harness as 120 retrieval questions, score nDCG@10/Recall@k separately.**
| Need | Effort |
|---|---|
| `load_scalr_questions` from queries split | 30 min (similar to `_load_generic_questions`) |
| `format_scalr_prompt` — but it's not a prompt, it's a query for the retriever | n/a — retrieval not generation |
| Answer extractor | n/a — score retrieved-doc-IDs against qrels |
| New harness mode `retrieval_only` that returns top-k IDs and scores nDCG/Recall vs. qrels | **2-3 days** — this is a new evaluation primitive the harness doesn't currently have |
| `CORPORA` entry | 30 min (523-doc embed is <1 min) |

**Option B — Treat MLEB-SCALR queries as input to a synthetic 5-way MC by sampling distractors from the corpus.** Re-creates the original LegalBench-SCALR task. ~4 hr to build the distractor sampler; reuses `format_casehold_prompt` (line 254) almost verbatim — same 5-way MC + holding format, same `extract_answer_mc5` extractor. But this is basically just **reconstructing LegalBench-SCALR from scratch — the team should just download `nguha/legalbench` SCALR config directly** (~30 min, plug-and-play with `format_casehold_prompt`).

**Option C — Use `eval/run_embedding_comparison.py` only.** Skip the QA harness; treat SCALR as a retrieval A/B fixture. ~1 day.

**Total recommended effort:**
- Option C (retrieval-only A/B): **~1 day**
- Option B via direct LegalBench download (5-way MC): **~4 hr**, slots into the existing CaseHOLD pipeline almost verbatim
- Option A (new harness primitive): **~3 days**, only worth it if the team plans to add more BEIR-style benchmarks later

### 3.5 Cluster vs. local feasibility
- 120 queries + 523 corpus docs is **trivially local** — embed the corpus in <1 min on any GPU; no cluster needed.
- Smaller than every other corpus in the project by 3+ orders of magnitude.

### 3.6 Anything weird
- **Two-SCALRs gotcha** — `isaacus/mleb-scalr` (retrieval, 120 q / 523 corpus) is NOT the same packaging as LegalBench `scalr` (5-way MC, 571 items, also CC BY 4.0). The data overlaps but the task framing is different. The 2026-04-27 meeting note says "MLEB-SCALR" — verify the team intends retrieval, not MC.
- The acronym SCALR is **not defined** in the LegalBench task page or the MLEB paper as cited. lexeme-dev/scalr GitHub repo is the original source. Plausibly "Supreme Court Appellate Legal Retrieval" but unsourced.
- The MLEB version drops the **multiple-choice candidate pool** — the LegalBench version includes 5 holding candidates per query (1 gold + 4 distractors); the MLEB version flattens this to 1 gold per query against the 523-doc corpus, deriving distractors implicitly.
- **Tiny corpus = ceiling effects**: a 523-doc corpus with 120 queries means even mediocre retrievers will hit Recall@10 > 0.9. The discrimination power of the benchmark is questionable for state-of-the-art embedders — see MLEB paper Table for variance.
- License-clean. No gating. No reproduction issues known.

---

## 4. Implementation order recommendation (for the team to greenlight)

If the team agrees with the scoping above, the suggested order is:

### Phase 1 — SCALR retrieval-only A/B (1 day, low-risk)
- [ ] Download `isaacus/mleb-scalr` (3 splits) to `datasets/scalr/` as parquet → CSV.
- [ ] Add `scalr` entry to `CORPORA` in `utils/fast_embed.py` (523 docs, `text_col="text"`, `idx_col="_id"`).
- [ ] Embed once with `gte-large` (default), once with `bge-m3`, once with `legal-bert` — A/B retrieval quality against qrels.
- [ ] Add a small Recall@10 / nDCG@10 scorer to `eval/run_embedding_comparison.py`.
- [ ] Output: a 1-page table for the paper sprint showing retriever-quality variance on legal reasoning vs. legal lookup.

### Phase 2 — FRAMES with linked-pages-only corpus (1.5-2 days, medium-risk)
- [ ] Download `google/frames-benchmark` parquet → `datasets/frames/questions.csv`.
- [ ] Write a one-shot Wikipedia scraper (`utils/scrape_frames_passages.py`) that pulls every URL listed across all 824 questions, chunks into ~512-tok passages, dedupes by URL+chunk-idx, writes `datasets/frames/passages.csv` with `idx`, `text`, `source_url`, `question_ids` columns.
- [ ] Add `frames` entry to `CORPORA`. Embed with default `gte-large` (~10 min on 3070).
- [ ] Add `frames` branch to `load_questions` and `format_question_prompt` in `eval/eval_config.py`. Formatter: open-ended, paragraph-length, no MC, similar to `format_open_prompt`.
- [ ] Add an LLM-as-judge function (verbatim from FRAMES paper Fig 6) — this is the official metric.
- [ ] Wire `golden_passage` mode by adding a `golden_idxs` column on `questions.csv` (mapping from `wikipedia_link_*` to embedded chunk IDs). This reproduces the paper's 0.729 ceiling.
- [ ] Smoke-test N=30 on `groq-llama70b` `rag_simple` (expected ~0.40-0.50 from paper) and `golden_passage` (expected ~0.70-0.75).

### Phase 3 — FRAMES paper-grade run (after Phase 2 smoke-test passes)
- [ ] N=200 paired McNemar across `rag_simple`, `multi_hyde_diverse`, `iter_hyde`, `golden_passage` on Llama 70b. Expected `multi_hyde_diverse` lift is **larger than on MuSiQue** (FRAMES has more retrieval-coverage gap).
- [ ] If lift replicates, FRAMES becomes the second multi-hop confirmation point alongside MuSiQue.

### Phase 4 (optional) — Add LegalBench-SCALR as a 5-way MC dataset
- [ ] If the team wants a third legal-MC point on the BarExam axis, downloading `nguha/legalbench` SCALR config and reusing `format_casehold_prompt` is ~4 hr. **But this duplicates BarExam/CaseHOLD's option-disambiguation regime** without adding much taxonomic signal.

---

## 5. Risk register

### FRAMES
| Risk | Severity | Mitigation |
|---|---|---|
| **No bundled corpus** — corpus strategy choice has 5x variance in effort. | High | Commit to option (3) "linked-pages-only" up-front. Defer Wikipedia-snapshot to a later paper. |
| **LLM-as-judge is the official metric** — string EM under-reports by ~10-20pp. | High | Reuse paper's Fig 6 auto-rater prompt. Validate κ on a 30-question human spot-check before publishing. |
| **Judge LLM cost** — adds 1 LLM call per question per run × N modes. | Medium | Use a cheap judge (DeepSeek/Gemini Flash); cache judgments by (question_id, predicted_answer) hash. |
| **Wikipedia link rot** — papers from 2024 already see ~1-3% dead-link rates. | Low | Snapshot once in Phase 2; freeze. |
| **CC BY-SA 4.0 share-alike** on Wikipedia content, if redistributed. | Low | Don't redistribute the embedded corpus; cite Wikipedia in any dump. |
| **Overlap with MuSiQue** — both are multi-hop + Wikipedia-flavored. The team may end up with two correlated multi-hop signals. | Medium | Use FRAMES specifically for the 2-step vs. multi-step delta (paper shows +25pp gap, larger than MuSiQue). |
| **Reasoning-type column** is single-string with 30 values, some compound. | Low | Treat as categorical; one-hot at analysis time. |

### SCALR (MLEB)
| Risk | Severity | Mitigation |
|---|---|---|
| **Wrong-SCALR risk** — meeting note says MLEB-SCALR (retrieval) but the team may actually want LegalBench-SCALR (5-way MC). | High | **Confirm with the team in the next standup before any code is written.** |
| **Tiny corpus (523 docs)** → ceiling effects; modern embedders cluster near R@10 = 0.95+, low discrimination power. | Medium | Use Recall@1 / nDCG@1 instead of @10 to recover discrimination. |
| **Harness mismatch** — eval_harness.py is QA-shaped, MLEB-SCALR is BEIR-shaped. | High | Use `run_embedding_comparison.py` instead of `eval_harness.py`, OR reconstruct the LegalBench MC version. |
| **Acronym undefined** — slightly annoying for paper-writing; "SCALR" appears unexpanded across all primary sources. | Low | Cite as "SCALR (Zheng et al., LegalBench 2023; lexeme-dev)" without expansion, mirroring how the original authors cite it. |
| **No reported leaderboard** for MLEB-SCALR specifically — the MLEB paper has aggregate retrieval scores but per-task breakdowns are sparse. | Medium | Run baseline `gte-large` first as the calibration point; nDCG@10 ≈ 0.7-0.8 is the realistic range for legal retrieval at this corpus scale. |
| **License**: CC BY 4.0 — clean. | None | Standard attribution. |

---

## 6. Sources

- HuggingFace: https://huggingface.co/datasets/google/frames-benchmark
- HuggingFace: https://huggingface.co/datasets/isaacus/mleb-scalr
- FRAMES paper: https://arxiv.org/abs/2409.12941 ("Fact, Fetch, and Reason", Krishna et al., Google + Harvard, Sept 2024)
- FRAMES leaderboard: https://llm-stats.com/benchmarks/frames
- LegalBench SCALR: https://hazyresearch.stanford.edu/legalbench/tasks/scalr.html
- LegalBench paper: https://arxiv.org/pdf/2308.11462
- MLEB paper: https://arxiv.org/pdf/2510.19365
- lexeme-dev/scalr: https://github.com/lexeme-dev/scalr (referenced; SCALR acronym not expanded)
- Verified against: `eval/eval_config.py` (lines 27-300), `utils/fast_embed.py` (CORPORA dict, lines 85-122).
