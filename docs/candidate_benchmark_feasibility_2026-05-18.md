# Candidate Legal Benchmark Feasibility - 2026-05-18

This note tracks the post-pivot candidate datasets that could replace or
supplement weaker main-grid benchmarks after the CaseHOLD/SCALR removal. It is
operational, not a citation gate.
Use `docs/signoff_log.md` for promoted result claims.

## Summary

| Dataset | Local status | Retrieval corpus | Answer format | Fit for Snap-HyRE |
|---|---|---:|---|---|
| Legal RAG Bench | pulled, converted, embedded | 4,876 fixed passages | open-ended answers, 100 QA rows | high retrieval fit; downstream needs LLM judge |
| LegalSearchQA | raw HF parquet pulled | none yet; 50 source URLs need frozen crawl/strip | 4-way MC, 50 rows | high conceptual fit after corpus build; lower priority |
| LEXam | raw HF parquets pulled | none shipped | MC 4/8/16/32 plus open questions | strong legal reasoning benchmark, weak RAG fit without external corpus |
| MLEB retrieval sets | inspected remotely | fixed MTEB-style corpora | retrieval-only query-to-document labels | strong retrieval-fit appendix candidates |
| LegalBench-RAG | not pulled | fixed corpus via external download | retrieval-only query-to-snippet labels | strong retrieval-only benchmark, contracts-heavy |
| Legal-Link-EU | converted, embedded, smoke-tested | 3,688 fixed EUR-Lex evidence contexts | 4-way MC, 1,127 rows | strongest exact-scored MC retrieval candidate found so far |
| MASLegalBench | converted, embedded, smoke-tested | GDPR case/context items in repo Arrow dataset | 303 four-way MC rows wired; 950 total rows preserved | strong answer fit; no explicit per-question gold passage qrels |

## Legal RAG Bench

Source: `https://huggingface.co/datasets/isaacus/legal-rag-bench`

Local files:

- `datasets/legal_rag_bench/corpus.jsonl`
- `datasets/legal_rag_bench/qa.jsonl`
- `datasets/legal_rag_bench/passages.csv`
- `datasets/legal_rag_bench/questions.csv`

Local Chroma collection:

- `legal_rag_bench_passages`
- embedded rows: 4,876

Verified shape:

- Corpus rows: 4,876
- QA rows: 100
- Unique gold passage ids: 95
- Gold-id alignment: 95/95 found in Chroma

Raw-question retrieval baseline:

- Cache: `caches/retrieval/full/legal_rag_bench_qfull_seed42_raw_question_k10.jsonl`
- Hit@1: 0.0700
- Hit@5: 0.2400
- Hit@10: 0.3500
- MRR@10: 0.1345

Harness state:

- `legal_rag_bench` is wired as an open-ended dataset in
  `eval/eval_config.py`, `eval/eval_harness.py`,
  `scripts/build_generation_cache.py`, `scripts/build_retrieval_cache.py`,
  `scripts/audit_retrieval_id_alignment.py`, and `utils/fast_embed.py`.
- Exact downstream accuracy is not directly available because answers are
  open-ended. Current harness would use the open-answer LLM judge path.
- Retrieval metrics are already valid and should be the first probe.

Recommended next step:

1. Run `rag_hyde` and `snap_hyre` generation caches on all 100 questions.
2. Build generated retrieval caches.
3. Compare Hit@1/5/10 and MRR@10 against raw question retrieval before any
   downstream judged answer runs.

## LegalSearchQA

Source: `https://huggingface.co/datasets/boqiny/LegalSearchQA`

Local files:

- `datasets/legal_search_qa/raw/test.parquet`

Verified shape:

- Rows: 50
- Columns: `id`, `question`, `choice_A`, `choice_B`, `choice_C`, `choice_D`,
  `answer`, `rationale`, `domain`, `category`, `difficulty`, `source_name`,
  `source_url`, `date_verified`
- Date verified: 2026-03-26
- Domains: 10
- Most common domains: labor law 11, immigration law 9, tax law 6, technology
  regulation 5, drug policy 5

Source URL/domain shape:

- 50 source URLs over 27 domains.
- Largest domains: `www.uscis.gov` 7, `www.irs.gov` 5, `www.congress.gov` 5,
  `www.federalregister.gov` 3, `www.paycor.com` 3, `www.sec.gov` 3.
- Some sources are PDFs, e.g. Supreme Court opinions.

Harness state:

- Not wired as a dataset yet.
- It can become a clean 4-way MC row once `choice_A` through `choice_D` are
  normalized to the repo's lowercase choice columns.
- It is not a fair RAG benchmark until the linked sources are frozen into a
  local corpus.

Recommended crawl/build path:

1. Convert the parquet to `datasets/legal_search_qa/questions.csv`.
2. Download each distinct `source_url` to `datasets/legal_search_qa/raw/html/`
   or `raw/pdf/`.
3. Strip HTML/PDF into text with source metadata and chunk into
   `datasets/legal_search_qa/passages.csv`.
4. Embed into `legal_search_qa_passages`.
5. Audit whether each question's source URL maps to at least one corpus passage.
6. Only then run raw/HyDE/Snap-HyRE retrieval curves.

Priority:

- Lower than Legal RAG Bench because corpus construction is custom and the
  answers are current-law sensitive. Useful as an appendix/current-law probe,
  not a near-term replacement unless we specifically want web-source RAG.

## LEXam

Source: `https://huggingface.co/datasets/LEXam-Benchmark/LEXam`

Upstream repo checked: `https://github.com/LEXam-Benchmark/LEXam`

Local files:

- `datasets/lexam/raw/mcq_4_choices/test.parquet`
- `datasets/lexam/raw/mcq_8_choices/test.parquet`
- `datasets/lexam/raw/mcq_16_choices/test.parquet`
- `datasets/lexam/raw/mcq_32_choices/test.parquet`
- `datasets/lexam/raw/open_question/dev.parquet`
- `datasets/lexam/raw/open_question/test.parquet`

Verified shape:

| Split | Rows | Notes |
|---|---:|---|
| `mcq_4_choices/test` | 1,655 | German/English, Swiss/international law |
| `mcq_8_choices/test` | 1,463 | same ids as some MC4 rows, expanded option set |
| `mcq_16_choices/test` | 1,028 | harder expanded option set |
| `mcq_32_choices/test` | 550 | hardest expanded option set |
| `open_question/dev` | 300 | open legal exam answers |
| `open_question/test` | 2,541 | open legal exam answers |

Observed schema:

- MC: `question`, `choices`, `gold`, `course`, `language`, `area`,
  `jurisdiction`, `year`, `n_statements`, `none_as_an_option`, `id`,
  `negative_question`
- Open: `question`, `answer`, `course`, `language`, `area`, `jurisdiction`,
  `year`, `id`

Corpus/source-material check:

- The Hugging Face dataset and upstream GitHub `data/` directory contain only
  exam-question exports (`MCQs_test_{4,8,16,32}.xlsx`,
  `open_questions_dev.xlsx`, `open_questions_test.xlsx`) and no statute,
  case-law, course-material, or source-document corpus.
- The official evaluation code formats closed-book prompts with course context
  and general legal expertise instructions; it does not retrieve from a
  provided corpus.
- Some prompts ask about specific cases or legal provisions, but the dataset
  does not ship the underlying case text/provision text needed to build a
  deterministic gold-passage RAG setup.
- Practical consequence: LexAM is usable as a reasoning/stress side benchmark,
  not as a retrieval-accuracy benchmark unless we separately build a Swiss/EU
  law corpus and accept that gold retrieval labels are absent.

Useful local stats:

- MC questions are long enough to stress context handling: median question
  length is about 674 chars for MC4, 667 for MC8, 714 for MC16, and 808 for
  MC32.
- Open questions are much longer and harder to score: test median question
  length about 700 chars, median reference answer about 959 chars, with p95
  reference answers around 5.8k chars.

Harness state:

- Not wired as a dataset yet.
- MC rows can be exact-scored but the harness currently assumes A-D or A-E
  closed-set formats; LEXam needs variable-N answer formatting and extraction.
- Open rows would require an LLM judge or rubric scorer.
- No retrieval corpus ships with the benchmark, so it is primarily a legal
  reasoning benchmark unless we attach external Swiss/EU/international legal
  materials.

Recommended use:

- Good candidate for an LLM-only / reasoning-capability side analysis.
- Not a direct replacement for HousingQA if the goal is to make retrieval
  accuracy the headline.
- If used in the main grid, prefer the `mcq_4_choices` split first because it
  is exact-scored and closest to the existing BarExamQA format.

## Other Retrieval-Fit Candidates

### MLEB / Isaacus Retrieval Sets

The Massive Legal Embedding Benchmark collection looks useful for retrieval
exposure claims because its datasets generally use the MTEB IR shape:
`corpus.jsonl`, `queries.jsonl`, and `default.jsonl` qrels.

Remote files inspected:

- `queries.jsonl`: question, issue statement, summary, keyword, or clause-type
  query text.
- `corpus.jsonl`: candidate passages.
- `default.jsonl`: qrels with `query-id`, `corpus-id`, and `score`.

The important distinction is that several MLEB tasks have natural-language
questions, but they still evaluate retrieval labels rather than final answer
labels.

Promising candidates checked:

| Dataset | Query rows | Corpus rows | Why it may fit |
|---|---:|---:|---|
| `isaacus/australian-tax-guidance-retrieval` | 112 | 105 | real taxpayer questions paired with Australian government tax guidance; closest to legal QA retrieval |
| `isaacus/mleb-consumer-contracts-qa` | 198 | 82 | questions about online terms of service paired with relevant clauses |
| `isaacus/mleb-scalr` | 120 | 523 | SCALR question-to-holding retrieval subset; overlaps conceptually with current SCALR |
| `isaacus/gdpr-holdings-retrieval` | 500 | 500 | fact patterns paired with GDPR/court holdings; retrieval over holdings |
| `isaacus/irish-legislative-summaries` | 500 | 500 | legislative long-title query to act text retrieval |
| `isaacus/singaporean-judicial-keywords` | 500 | 500 | catchword query to judgment text retrieval |
| `isaacus/uk-legislative-long-titles` | 78 | 78 | smaller legislative long-title query to act text retrieval |
| `isaacus/contractual-clause-retrieval` | 45 | 90 | clause-definition query to clause examples |
| `isaacus/license-tldr-retrieval` | 65 | 65 | license-summary query to full license text |

Fit assessment:

- These are excellent for showing retrieval lift: fixed corpus, fixed query
  set, explicit qrels, no LLM judge required for Hit/MRR.
- Several have question-like queries, especially Australian Tax Guidance and
  Consumer Contracts QA, but they do not ship downstream answer labels separate
  from retrieval labels.
- `australian-tax-guidance-retrieval` is the best next candidate if we want a
  Legal RAG Bench-like retrieval story with real user questions and government
  guidance.
- `mleb-scalr` is useful as a clean retrieval-only SCALR variant, but it should
  not be counted as a new fourth benchmark because it overlaps with our current
  SCALR story.

Recommended next step:

1. Pull `isaacus/australian-tax-guidance-retrieval`.
2. Convert MTEB IR files into the repo's `questions.csv` / `passages.csv`
   shape.
3. Embed the 105-passage corpus.
4. Run raw vs HyDE vs Snap-HyRE retrieval curves only.

### Legal RAG QA

Source: `https://huggingface.co/datasets/isaacus/legal-rag-qa`

Fit assessment:

- Small end-to-end legal RAG dataset released by Isaacus alongside Legal RAG
  Bench.
- Corpus: 190 passages and external materials.
- QA: 138 question-answer-relevant-passage triplets from LibreTexts'
  Introduction to Criminal Law.
- Good for quick open-answer retrieval/grounding checks because it has a fixed
  corpus and relevant document IDs.
- Not MC/exact-scored; downstream evaluation still needs an LLM judge or
  exact/semantic answer scorer.

Recommended use:

- Useful appendix or smoke benchmark if Legal RAG Bench's 100 rows are too few.
- Not a clean replacement for HousingQA if the main matrix must stay
  exact-scored.

### Legal-Link-EU

Source: `https://huggingface.co/datasets/disi-unibo-nlp/legal-link-eu`

Local/harness status:

- Local corpus: `datasets/legal_link_eu/passages.csv`, 3,688 non-empty
  deduped EUR-Lex evidence contexts.
- Local questions: `datasets/legal_link_eu/questions.csv`, 1,127 four-way MC
  rows.
- Local adversarial file: `datasets/legal_link_eu/perturbed_contexts.jsonl`,
  5,635 perturbed context rows preserved outside the clean retrieval corpus.
- Chroma collection: `legal_link_eu_passages`, 3,688 embedded documents.
- Harness wiring is present in `eval/eval_config.py`, `eval/eval_harness.py`,
  `eval/eval_metrics.py`, `utils/fast_embed.py`,
  `scripts/build_generation_cache.py`, `scripts/build_retrieval_cache.py`,
  and `scripts/audit_retrieval_id_alignment.py`.
- Question IDs are unique local row IDs. The original repeated upstream
  document-pair/relation ID is preserved as `example_id`.
- Gold retrieval labels are the row's original non-empty context IDs. Most
  rows have five gold contexts; nine rows have four because the fifth upstream
  original context slot is blank.

Fit assessment:

- 1,127 four-way MC questions over EUR-Lex authority relationships.
- Each row contains `question`, four `options`, `correct_label`,
  `contexts`, `context_titles`, and `perturbed_contexts`.
- Each row has five original evidence contexts. These are provided evidence
  passages, not answer-option passages. The usual title pattern is repeated
  source-document chunks plus repeated target-document chunks, e.g.
  `Source (CELEX)` and `Target (CELEX)`.
- The task is explicitly about reasoning over changing legal authority:
  repeal, correction, obsolescence, extension of validity/application, and
  related document-link relationships.
- Strong exact-scored legal reasoning candidate. It also has valid evidence
  contexts and misleading perturbed contexts, which could support a sharp
  "retrieval helps vs misleading authority hurts" analysis.

Corpus/qrels notes:

- It does not ship as a separate global retrieval corpus file. It ships
  row-level context fields. The local conversion builds a fixed corpus from
  the original `contexts` field and treats each row's original non-empty
  context IDs as relevant evidence.
- Deduplication is not intended to put answer choices into the corpus. It is
  only a corpus-normalization step because the same EUR-Lex chunks recur across
  examples. In a remote inspection, 5,635 row-level context slots collapsed to
  3,688 unique non-empty title-text pairs, with no duplicate context text
  inside a single row.
- The `perturbed_contexts` field should stay out of the main retrieval corpus.
  It is an adversarial/misleading-authority condition, not the clean evidence
  pool.

Alignment and smoke status:

- Gold-id alignment: 3,688/3,688 unique gold IDs found in
  `legal_link_eu_passages`; no metadata fallback.
- The CELEX parser was hardened for parenthesized identifiers such as
  `32021D0506(01)`, and final evidence prompts now include retrieved passage
  metadata headers: `passage_id`, `legal_link_role`, `title`, and `citation`.
- Legal-Link full retrieval caches should use `CROSS_ENCODER_MAX_CHARS=22000`.
  A full raw-question rebuild at CE12000 still left 75 rows with at least one
  char-truncated cross-encoder document; CE22000 eliminates char-level
  cross-encoder truncation on the local corpus.
- Raw-question retrieval full cache, N=1,127, k=10, CE22000:
  `caches/retrieval/full/legal_link_eu_qfull_seed42_raw_question_ce22000_k10.jsonl`.
  It has 1,127/1,127 nonempty rows, zero duplicate/missing IDs, zero short
  rows, zero rows without gold IDs, zero cross-encoder query/doc truncation,
  Hit@1 0.6637, Hit@5 0.9059, Hit@10 0.9556, Recall@5 0.4260, and MRR@10
  0.7689. Source-document evidence appears in 966/1,127 rows; target-document
  evidence appears in 781/1,127 rows; at least one gold context appears in
  1,077/1,127 rows.
- Generated retrieval smoke, N=20, k=5, preliminary CE12000 caches:
  `groq-llama70b` `rag_hyde` Hit@5 0.6500 / MRR@5 0.5625 and `snap_hyre`
  Hit@5 0.8500 / MRR@5 0.6617; `or-gemma4-26b` `rag_hyde` Hit@5 0.5000 /
  MRR@5 0.4292 and `snap_hyre` Hit@5 0.8000 / MRR@5 0.6308;
  `or-ministral-8b` `rag_hyde` Hit@5 0.5500 / MRR@5 0.4667 and `snap_hyre`
  Hit@5 0.5500 / MRR@5 0.4725. These should be rebuilt with CE22000 before
  promotion. Snap-HyRE improves over HyDE on Groq and Gemma in this small
  smoke, but raw-question retrieval remains strongest.
- Historical `or-ministral-8b`, N=3 smoke runs passed under
  `NO_SILENT_FALLBACK=1` for `llm_only`, `rag_simple`, `rag_hyde`,
  `snap_hyre`, `rag_rewrite`, `golden_passage`, and
  `golden_plus_neighbors`. The active small-model row is now
  `groq-llama8b`; rerun these smokes before launching full Legal-Link rows.
- Tiny N=3 answer smoke: `llm_only` 0/3, `rag_simple` 2/3, `rag_hyde` 3/3,
  `snap_hyre` 3/3, `rag_rewrite` 2/3, `golden_passage` 2/3, and
  `golden_plus_neighbors` 2/3. This is only a wiring sanity check, not a
  promoted result.
- Tiny generated-retrieval smoke: `rag_hyde` Hit@5 0.6667 and MRR@5 0.6667;
  `snap_hyre` Hit@5 1.0000 and MRR@5 0.8333 on the same first-three sampled
  rows. This is too small for claims but confirms the cache/replay path.
- Detail-log health scan over the N=3 smoke logs found zero errors, missing
  predictions, parse failures, long final answers, or HyDE/report artifacts.
- Post-hardening answer smoke found clean cache replay and audit fields:
  Groq q20 `rag_simple` 16/20 and `snap_hyre` 17/20; Gemma Cloudflare q5
  `rag_simple` 5/5 and `snap_hyre` 4/5. Field-check detail logs show
  `provider_route`, `cross_encoder_max_chars=12000`, zero CE truncation,
  source/target document hit flags, and evidence metadata in `evidence_store`.

Recommended use:

- Best newly found exact-scored candidate if we want to replace or supplement
  HousingQA with an MC benchmark.
- Next step before promotion: build full Legal-Link generation and generated
  retrieval caches under CE22000, audit source/target and gold-retrieval
  exposure, then
  start deliberate full answer cells.
- Keep `perturbed_contexts` for a separate robustness analysis, not for the
  main clean-corpus retrieval result.

### MASLegalBench

Source: `https://github.com/HKUST-KnowComp/MASLegalBench`

Local/harness status:

- Dataset is a Hugging Face Arrow export committed in the GitHub repo and is
  mirrored locally under `datasets/mas_legal_bench/raw/`.
- Local corpus: `datasets/mas_legal_bench/passages.csv`, 3,950 deduped context
  rows.
- Local main question file: `datasets/mas_legal_bench/questions.csv`, 303
  four-way MC rows.
- Local all-question file: `datasets/mas_legal_bench/questions_all.csv`, 950
  total rows; 647 Yes/No rows are preserved but not used by the current
  exact-scored MC harness.
- Chroma collection: `mas_legal_bench_passages`, 3,950 embedded documents.
- Non-question context types include `background`, `legal framework`,
  `entity`, `relation`, and `inferred alignment`.
- The official runner retrieves BM25/embedding context chunks within each case
  source and evaluates exact answer accuracy.

Fit assessment:

- Strong candidate for downstream answer accuracy under a RAG-style context
  setup.
- Legal domain is GDPR penalty notices/cases, which is coherent and less weird
  than HousingQA's yes/no-only state-law setup.
- It does not provide explicit per-question gold passage IDs. Retrieval
  exposure can be evaluated as source/context selection or context-type
  retrieval, but not as clean Hit@k/MRR over gold passage IDs unless we derive
  labels.
- The local harness intentionally leaves `gold_idx` empty rather than treating
  all same-source context rows as official gold passages. Detail logs report
  `same_source_retrieved` as an operational exposure check.
- Because no per-question gold passage is supplied, `golden_passage` and
  `golden_plus_neighbors` are not meaningful canonical rows for this benchmark
  unless we later derive and explicitly justify pseudo-gold evidence labels.

Smoke status:

- Raw-question retrieval cache, N=20, k=5: 20/20 nonempty retrieval rows;
  same-source evidence was retrieved on 15/20 rows.
- Raw-question retrieval full cache, N=303, k=10:
  `caches/retrieval/full/mas_legal_bench_qfull_seed42_raw_question_k10.jsonl`.
  It has 303/303 nonempty rows, zero duplicate/missing IDs, zero short rows,
  and zero cross-encoder query/doc truncation. It has no official gold qrels,
  so Hit/MRR are not meaningful; same-source evidence was retrieved on
  239/303 rows at k=10.
- Historical `or-ministral-8b`, N=3 smoke runs passed under
  `NO_SILENT_FALLBACK=1` for `llm_only`, `rag_simple`, `rag_hyde`,
  `snap_hyre`, and `rag_rewrite`.
- Active small-model replacement `groq-llama8b` passed a one-row
  MASLegalBench `llm_only` provider smoke through the harness with
  `NO_SILENT_FALLBACK=1`.
- HyDE and Snap-HyRE generation-cache plus retrieval-cache replay paths both
  work on the N=3 smoke. Snap-HyRE required one same-model generation format
  retry in the cache build, then replayed cleanly.

Recommended use:

- Good answer-accuracy side benchmark, especially if we want exact-scored legal
  reasoning with retrieved evidence.
- Less ideal than Legal-Link-EU for the retrieval-accuracy headline unless we
  are comfortable with derived retrieval labels.

### LegalBench-RAG

Source: `https://github.com/zeroentropy-ai/legalbenchrag`

Fit assessment:

- Strong retrieval-only benchmark: the README describes query/test cases with
  ground-truth snippets referencing corpus files and character ranges.
- Larger than Legal RAG Bench: paper/search metadata reports 6,858 query-answer
  pairs over a corpus above 79M characters.
- It is contracts-heavy: source datasets are ContractNLI, CUAD, MAUD, and
  PrivacyQA.
- Data download is external rather than a simple Hugging Face dataset, so it is
  a little more operationally annoying than the Isaacus/MLEB JSONL datasets.

Recommended use:

- Good appendix retrieval benchmark if we want a broader retrieval-only result.
- Less ideal as a main replacement because it is not end-to-end QA with final
  answer accuracy and is less aligned with the current legal-exam/RAG narrative.
