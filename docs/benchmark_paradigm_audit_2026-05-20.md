# Benchmark Paradigm Audit - 2026-05-20

Purpose: prevent a repeat of the CaseHOLD / LegalBench-SCALR mistake where the
searchable "corpus" was effectively a large pool of answer-option holdings,
including wrong displayed options. The active Snap-HyRE matrix should instead
use question + choices, retrieve from a reference corpus, and let retrieval add
supporting legal evidence rather than recover the answer option itself from a
giant answer-option pool.

## Decision

Keep the active exact-scored matrix as:

- BarExamQA
- HousingQA, with jurisdiction state filtering for all retrieval rows
- Legal-Link-EU
- MASLegalBench

Keep CaseHOLD and LegalBench-SCALR out of the active main matrix unless they are
explicitly reframed. Their current local corpora are holdings built from the
answer choices, so they do not match the intended retrieval-over-reference-corpus
paradigm.

## Method Comparability Ledger

The active claim is not that every benchmark has identical data structure. The
active claim is that each method is evaluated against the same benchmark
interface within a dataset, and that dataset-specific differences are limited to
the minimum needed to express the benchmark's real task.

| Axis | Same across active benchmarks | Dataset-specific difference | Justification |
|---|---|---|---|
| Canonical methods | Same method ladder: `llm_only`, `rag_simple`, `golden_passage`, `golden_plus_neighbors`, `rag_hyde`, `snap_hyre`, `rag_rewrite`. | MAS marks `golden_passage` and `golden_plus_neighbors` not applicable. | MAS has no official passage-level gold evidence id in the active MC file; forcing an oracle row would fabricate a control. |
| Final answer stage | Retrieval methods answer from retrieved passages plus the formatted question using the `rag` answer prompt. Snap-HyRE does not show the snap answer to the final answerer. | Dataset formatting changes the required final answer type: MC letter for BarExamQA, Legal-Link-EU, MAS; Yes/No for HousingQA. | This is benchmark scoring format, not a method change. |
| Retrieval depth | Main answer rows use `RETRIEVAL_K=5`; retrieval caches keep k up to 10 for exposure curves. | `golden_plus_neighbors` reserves Source 1 for the gold passage and fills the rest from nearest neighbors. | This is an oracle-neighbor diagnostic, reported separately from retrieved-only methods. |
| Retrieval corpus | Each retrieval method searches a real reference/support corpus, not displayed answer options. | Collections differ by benchmark: BarExamQA support passages, Housing statutes, EUR-Lex contexts, MAS penalty-notice contexts. | Corpus identity is the benchmark itself. The fairness requirement is no option-pool corpus. |
| Metadata filters | No hidden metadata filter by default. | HousingQA retrieval rows require the question-state filter for all retrieval methods. | Housing questions name a jurisdiction and the national corpus contains every jurisdiction. The filter removes wrong-jurisdiction search space uniformly and is not answer leakage. |
| Gold/retrieval metrics | Where official gold ids exist, report Hit@k/MRR over retrieved ids. | MAS retrieval exposure is a same-source proxy, not official passage-level gold retrieval. | MAS exposes source context grouping, not precise per-question evidence ids. Claims must use the proxy label. |
| Generated queries | HyDE, Snap-HyRE, and rewrite spend the same logical generation steps on each applicable dataset. | Legal-Link-EU generated-query methods see the MC choices; raw RAG is question-text retrieval. | The final task is MC. This is acceptable if described as choice-conditioned generated retrieval, not option-blind retrieval. |
| Cache/replay | Retrieval caches store passage ids and are keyed by dataset, method label, collection, embedding model, and metadata filter. | Housing state-filtered caches are separate from unfiltered caches; Legal-Link full caches use the long CE input setting. | These are execution guards to preserve the intended retrieval scope and avoid truncating important benchmark anchors. |

The main non-negotiable fairness rule is within-dataset symmetry: if HousingQA
uses the state filter, every HousingQA retrieval method gets it; if a dataset
lacks gold passage ids, every model/method row treats the oracle modes as not
applicable rather than substituting a proxy. Cross-dataset averages should
therefore be described as averages over benchmark-normalized method cells, not
as a single identical-corpus experiment.

## Choice-Corpus Check

A bounded local scan compared displayed answer choices against the searchable
corpus files.

| Benchmark | Questions | Corpus docs | Choices checked | Exact choice docs | Interpretation |
|---|---:|---:|---:|---:|---|
| BarExamQA | 1,195 | 856,835 | 4,779 | 0 | Support corpus, not option pool. |
| Legal-Link-EU | 1,127 | 3,688 | 4,508 | 0 | EUR-Lex context corpus, not option pool. |
| MASLegalBench | 303 | 3,950 | 1,212 | 0 | Penalty-notice context corpus, not option pool. |
| CaseHOLD | 3,600 | 51,296 | 18,000 | 18,000 | Every displayed option is embedded as a corpus doc. |
| LegalBench-SCALR | 571 | 1,733 | 2,855 | 2,855 | Every displayed option is embedded as a corpus doc. |

MAS had a few short option-string containments inside real passages, such as
legal article names or agency names. Those are not injected distractor rows.

## BarExamQA

BarExamQA is usable for the active paradigm. Questions and choices come from
`datasets/barexam_qa/qa/qa.csv`, while the search corpus is
`datasets/barexam_qa/passages/passages.tsv` embedded into `legal_passages`.
The corpus contains support/explanation/legal-reference text, not displayed
answer options. All 1,195 active `gold_idx` values resolve to corpus passages.

Caveat: BarExamQA gold references are MBE/bar-explanation style support
passages, not purely independent primary legal authority. The current good state
also depends on the augmented qrel-complete collection path, because the older
training-only source misses some full-QA gold ids.

## HousingQA

HousingQA is a clean reference-corpus benchmark when filtered by jurisdiction.
Questions are state-specific Yes/No legal questions in
`datasets/housing_qa/questions.csv`; the searchable corpus is statutory text in
`datasets/housing_qa/statutes.csv`, embedded into `housing_statutes` with
`state`, `citation`, `source`, and `idx` metadata.

The national unfiltered corpus is not a fair main-matrix setting because
retrieval is dominated by wrong-jurisdiction statutes. The state-filtered
process is therefore required for all Housing retrieval rows. The filter uses
the state named in the question and applies uniformly to:

- `rag_simple`
- `golden_plus_neighbors`
- `rag_hyde`
- `snap_hyre`
- `rag_rewrite`

`llm_only` and `golden_passage` do not perform corpus retrieval and are
unaffected.

Current caveat: state-filtered Housing full retrieval caches and answer rows are
still in progress. Existing unfiltered Housing retrieval rows are provenance /
ablation rows only.

## Legal-Link-EU

Legal-Link-EU is usable for the active paradigm. The corpus in
`datasets/legal_link_eu/passages.csv` is deduped EUR-Lex context text with
`source`, `citation`, `role`, and `context_title` metadata. Answer choices live
only in `datasets/legal_link_eu/questions.csv`, and `perturbed_contexts.jsonl`
is not embedded in the active clean corpus.

Caveat: generated-query methods are choice-conditioned, not option-blind.
Raw RAG retrieves from question text only, but HyDE, Snap-HyRE, and rewrite
generation see the multiple-choice framing. This is acceptable as long as the
paper describes the task accurately: retrieval over original EUR-Lex evidence
contexts with exact relation-answer scoring, where generated methods can use
the same choice information the final answerer receives.

Second caveat: Legal-Link is anchor-rich. Raw retrieval often preserves
source/target CELEX anchors better than generated queries; generated methods can
lose source/target anchors even though the corpus itself is clean.

## MASLegalBench

MASLegalBench should remain in the active matrix, but retrieval claims must be
framed as same-source proxy evidence rather than official passage-level gold
retrieval. The active exact-scored subset has 303 four-way MC questions in
`datasets/mas_legal_bench/questions.csv`; the corpus is 3,950 non-question
penalty-notice context rows in `datasets/mas_legal_bench/passages.csv`.

The searchable corpus is not a pool of wrong options. The conversion keeps
questions and choices separate from non-question context rows. MAS has no
official per-question gold evidence ids in the active MC file, so
`golden_passage` and `golden_plus_neighbors` remain not applicable.

Caveat: the same-source proxy is broad. A question may have hundreds of context
rows from the same penalty notice, so same-source retrieval is useful as an
operational proxy but not a precise evidence-hit metric.

## Excluded Benchmarks

CaseHOLD and LegalBench-SCALR are excluded from the current main exact-scored
retrieval matrix because the local corpora are constructed from holdings/options:

- `datasets/casehold/holdings_corpus.csv`
- `datasets/legalbench_scalr/holdings_corpus.csv`

In the local check, every displayed option appeared exactly as a corpus document.
This turns retrieval into finding the correct option among a large pool of
correct and incorrect option texts, rather than retrieving supporting legal
evidence from an external reference corpus. These datasets may still be useful
for reasoning-only or specially reframed analyses, but not for the active
Snap-HyRE retrieval-over-reference-corpus claim.
