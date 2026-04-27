# Results Tables

## Table A. BarExam Tier 3 method matrix: Gemma 4 26B-A4B, N=1195

| Mode | EM | Delta vs `rag_simple` | Audit | Sign-off | Slide read |
|---|---:|---:|---|---|---|
| `rag_simple` | 78.08% | baseline | MINOR | APPROVED-WITH-CAVEAT | Baseline, but 2/15 sampled records had null pred plus empty retrieval. |
| **`rag_snap_hyde`** | **81.17%** | **+3.09pp** | MINOR | **APPROVED** | **Winner.** Snap plus HyDE is the full-corpus legal-MC headline. |
| `snap_only_in_final` | 80.59% | +2.51pp | CLEAN | APPROVED | Strong second without HyDE. |
| `rag_hyde` | 78.91% | +0.83pp | MINOR | APPROVED | HyDE alone is small. |
| `subagent_rag` | 78.16% | +0.08pp | MINOR | APPROVED | Essentially flat. |
| `subagent_hybrid` | 74.23% | -3.85pp | MINOR | APPROVED-WITH-CAVEAT | Re-scored from raw 74.14%; materially worse. |
| `llm_only` | 79.75% | +1.67pp | CLEAN | APPROVED | Strong no-retrieval point. |
| `golden_passage` | 78.66% | +0.58pp | CLEAN | APPROVED | Golden passage does not dominate this setup. |

What this means in plain English: On full-corpus BarExam, Gemma 4 26B-A4B is strongest when its snap legal reasoning is paired with HyDE retrieval.
Source: `docs/signoff_log.md:Section A.1`; detail paths indexed in `docs/presentation/05_logs_index.md:BarExam Tier 3`.
Caveat: `rag_snap_hyde` often agrees with the snap answer because Gemma 4 has strong legal priors; this is BY DESIGN architecture and mechanism understanding.

## Table B. BarExam Tier 3 method matrix: Gemma 4 E4B, N=1195

| Mode | EM | Delta vs `rag_simple` | Audit | Sign-off | Slide read |
|---|---:|---:|---|---|---|
| `rag_simple` | 58.49% | baseline | MINOR | APPROVED-WITH-CAVEAT | Baseline with low exact-gold retrieval caveat. |
| `rag_hyde` | 60.59% | +2.10pp | MINOR | APPROVED | HyDE helps at E4B scale. |
| **`rag_snap_hyde`** | **62.18%** | **+3.69pp** | MINOR | **APPROVED** | **Winner.** Same method wins the smaller Gemma 4 size. |
| `snap_hyde_report` | 60.75% | +2.26pp | MINOR | APPROVED | Strong but below snap plus HyDE. |
| `snap_only_in_final` | 57.82% | -0.67pp | CLEAN | APPROVED | Snap-only does not carry at E4B scale. |
| `subagent_hybrid` | 58.83% | +0.34pp | MINOR | APPROVED | Flat. |
| `subagent_hyde` | 60.17% | +1.68pp | MINOR | APPROVED | Helpful but not best. |
| `subagent_rag` | 60.92% | +2.43pp | MINOR | APPROVED | Best non-snap-HyDE subagent row. |

What this means in plain English: The smaller Gemma 4 model repeats the same legal-MC winner, so the BarExam result is not a one-size artifact.
Source: `docs/signoff_log.md:Section A.2`; detail paths indexed in `docs/presentation/05_logs_index.md:BarExam Tier 3`.
Caveat: Full scan found one raw null parsed prediction for `rag_snap_hyde`; the sampled audit remained clean.

## Table C. BarExam cross-size headline: `rag_snap_hyde` lift

| Model size | Baseline `rag_simple` EM | `rag_snap_hyde` EM | Lift | Tier | Sign-off |
|---|---:|---:|---:|---|---|
| Gemma 4 26B-A4B | 78.08% | 81.17% | +3.09pp | Tier 3, N=1195 | APPROVED |
| Gemma 4 E4B | 58.49% | 62.18% | +3.69pp | Tier 3, N=1195 | APPROVED |

What this means in plain English: Snap+HyDE is the signed BarExam headline because it improves both Gemma 4 sizes at full corpus.
Source: `docs/signoff_log.md:Section A.3`; supporting audit summary in `docs/compiled_results.md:Section 1.1`.
Caveat: The lift is mostly snap-dominated; HyDE is marginal and can hurt when it contradicts a strong snap answer.
Figure: `docs/figures/01*.png`.

## Table D. Llama 70b MuSiQue Tier 2 method matrix, N=200

| Mode | EM | Delta vs `rag_simple` | McNemar p | Audit | Sign-off | Slide read |
|---|---:|---:|---:|---|---|---|
| `rag_simple` | 27.5% | baseline | n/a | CLEAN | APPROVED baseline | Control. |
| **`iterative_planning_table`** | **36.0%** | **+8.5pp** | **0.0533** | McNemar 12:30 | **APPROVED, TRENDING-SIG** | Best EM, just outside conventional 0.05. |
| **`multi_hyde_diverse`** | **35.5%** | **+8pp** | **0.0195** | CLEAN | **APPROVED, paper headline** | **Winner for significant positive lift.** |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | CLEAN | APPROVED mechanism decomposition | Query diversity alone is not enough. |
| `rag_snap_hyde` | 24.0% | -3.5pp | 0.36 | CLEAN | APPROVED cross-domain negative evidence | BarExam method does not transfer cleanly. |
| `iter_hyde` | 24.5% | -3.0pp | 0.47 | CLEAN | APPROVED multi-round neutral at large | More HyDE rounds do not help here. |
| `advisor_planning_table` | 23.0% | -4.5pp | 0.222 | McNemar 12:30 | APPROVED NS but informative negative | Planning form is not sufficient. |
| **`subagent_rag`** | **15.5%** | **-12.0pp** | **0.0007** | CLEAN | **APPROVED, sig negative** | Current gap-routing over-abstains. |

What this means in plain English: Llama 70b multi-hop improves with pooled HyDE or structured planning, while most other variants are neutral-to-negative.
Source: `docs/signoff_log.md:Section B.1`; paired statistics in `docs/mcnemar_2026-04-27.md:Update 2026-04-27 ~12:30 CDT`.
Caveat: `subagent_rag` is a real negative result for this gap-routing implementation, not a verdict against all subagent methods.
Figure: `docs/figures/02*.png`.

## Table E. Mechanism decomposition: Llama 70b MuSiQue, N=200

| Method | EM | Delta vs `rag_simple` | McNemar p | Mechanism read |
|---|---:|---:|---:|---|
| `rag_simple` | 27.5% | baseline | n/a | Retrieve on original question, then answer. |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | Query diversity alone; statistically non-significant. |
| `multi_hyde_diverse` | 35.5% | +8pp | 0.0195 | Diverse answer-bearing HyDE-style passages plus raw question. |
| Split: `mhd` minus `rag_multi_query` | n/a | +6.5pp | n/a | HyDE-style passages explain about 80% of the +8pp lift. |

What this means in plain English: The useful piece is answer-bearing HyDE retrieval, not generic query diversity.
Source: `docs/signoff_log.md:Sections B.1 and B.2`; decomposition details in `docs/mcnemar_2026-04-27.md:Update 2026-04-27 ~11:13 CDT`.
Caveat: The +6.5pp split is a decomposition against method EMs, not a separate paired significance test.
Figure: `docs/figures/03*.png`.

## Table F. Cross-domain method specificity

| Method family tested off-domain | Source-domain winner | Off-domain task/model | Comparator | Method result | Delta | McNemar p | Read |
|---|---|---|---:|---:|---:|---:|---|
| `rag_snap_hyde` | BarExam | MuSiQue, Llama 70b, N=200 | `rag_simple` 27.5% | 24.0% | -3.5pp | 0.36 | Negative paired McNemar; BarExam method does not carry to MuSiQue. |
| `multi_hyde_diverse` | MuSiQue | BarExam, Gemma 4 26B-A4B, paired first 200 | `rag_simple` 84.5% | 82.0% | -2.5pp | 0.499 | Provisional/source-status caveat; if confirmed, mhd does not lift BarExam. |

What this means in plain English: The signed story is task-specific methods, not one universal RAG trick.
Source: `docs/signoff_log.md:Sections B.1 and B.4`; source-status caveat in `docs/mcnemar_2026-04-27.md:BarExam cross-domain mhd test`.
Caveat: The BarExam mhd row has conflicting local source status across docs; keep it source-pending until the SLURM 55107 detail-log status is reconciled.
Figure: `docs/figures/04*.png`.

## Table G. Cross-family check, preliminary

| Dense model | Task | Baseline `rag_simple` | `multi_hyde_diverse` | Delta | McNemar p | Sign-off | Meeting read |
|---|---|---:|---:|---:|---:|---|---|
| Llama 70b | MuSiQue N=200 | 27.5% | 35.5% | +8pp | 0.0195 | APPROVED, significant | Strong Llama-specific headline. |
| Gemma 3 27B | MuSiQue N=200 | 28.5% | 31.0% | +2.5pp | 0.5901 NULL | APPROVED | Not yet a clean cross-family lift; in-flight Tier 3 will tell. |

What this means in plain English: The mhd lift is real on Llama 70b, but the dense-model cross-family claim is not established.
Source: `docs/signoff_log.md:Sections B.1, B.3, and F`; paired Gemma/Llama details in `docs/mcnemar_2026-04-27.md`.
Caveat: Qwen3 full-corpus evidence is still running, so cross-family language should stay provisional.
Figure: `docs/figures/05*.png`.
