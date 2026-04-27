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

Source: docs/signoff_log.md Section A.1.

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

Source: docs/signoff_log.md Section A.2.

## Table C. BarExam cross-size headline: `rag_snap_hyde` lift

| Model size | Baseline `rag_simple` EM | `rag_snap_hyde` EM | Lift | Tier | Sign-off |
|---|---:|---:|---:|---|---|
| Gemma 4 26B-A4B | 78.08% | 81.17% | +3.09pp | Tier 3, N=1195 | APPROVED |
| Gemma 4 E4B | 58.49% | 62.18% | +3.69pp | Tier 3, N=1195 | APPROVED |

Source: docs/signoff_log.md Section A.3.

## Table D. Llama 70b MuSiQue Tier 2 method matrix, N=200

| Mode | EM | Delta vs `rag_simple` | McNemar p | Audit | Sign-off | Slide read |
|---|---:|---:|---:|---|---|---|
| `rag_simple` | 27.5% | baseline | n/a | CLEAN | APPROVED baseline | Control. |
| **`iterative_planning_table`** | **36.0%** | **+8.5pp** | **0.0533** | McNemar 12:30 | **APPROVED, TRENDING-SIG** | Best EM, just outside conventional 0.05. |
| **`multi_hyde_diverse`** | **35.5%** | **+8.0pp** | **0.0195** | CLEAN | **APPROVED, paper headline** | **Winner for significant positive lift.** |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | CLEAN | APPROVED mechanism decomposition | Query diversity alone is not enough. |
| `rag_snap_hyde` | 24.0% | -3.5pp | 0.36 | CLEAN | APPROVED cross-domain negative evidence | BarExam method does not transfer. |
| `iter_hyde` | 24.5% | -3.0pp | 0.47 | CLEAN | APPROVED multi-round neutral at large | More rounds do not help here. |
| `advisor_planning_table` | 23.0% | -4.5pp | 0.222 | McNemar 12:30 | APPROVED NS but informative negative | Planning form is not sufficient. |
| **`subagent_rag`** | **15.5%** | **-12.0pp** | **0.0007** | CLEAN | **APPROVED, sig negative** | **Actively hurts.** |

Source: docs/signoff_log.md Section B.1.

## Table E. Mechanism decomposition: Llama 70b MuSiQue, N=200

| Method | EM | Delta vs `rag_simple` | McNemar p | Mechanism read |
|---|---:|---:|---:|---|
| `rag_simple` | 27.5% | baseline | n/a | Retrieve on original question, then answer. |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | Query diversity alone; statistically non-significant. |
| `multi_hyde_diverse` | 35.5% | +8.0pp | 0.0195 | Diverse answer-bearing HyDE-style passages plus raw question. |
| Split: `mhd` minus `rag_multi_query` | n/a | +6.5pp | n/a | HyDE-style passages explain about 80% of the +8.0pp lift. |

Source: docs/signoff_log.md Sections B.1 and B.2.

## Table F. Cross-domain method specificity

| Method family tested off-domain | Source-domain winner | Off-domain task/model | Comparator | Method result | Delta | McNemar p | Read |
|---|---|---|---:|---:|---:|---:|---|
| `rag_snap_hyde` | BarExam | MuSiQue, Llama 70b, N=200 | `rag_simple` 27.5% | 24.0% | -3.5pp | 0.36 | Negative paired McNemar; BarExam method does not carry to MuSiQue. |
| `multi_hyde_diverse` | MuSiQue | BarExam, Gemma 4 26B-A4B, paired first 200 | `rag_simple` 84.5% | 82.0% | -2.5pp | 0.499 | Negative paired McNemar; multi-hop method does not carry to BarExam. |

Source: docs/signoff_log.md Sections B.1 and B.4.

## Table G. Cross-family check, preliminary

| Dense model | Task | Baseline `rag_simple` | `multi_hyde_diverse` | Delta | McNemar p | Sign-off | Meeting read |
|---|---|---:|---:|---:|---:|---|---|
| Llama 70b | MuSiQue N=200 | 27.5% | 35.5% | +8.0pp | 0.0195 | APPROVED, significant | Strong Llama-specific headline. |
| Gemma 3 27B | MuSiQue N=200 | 28.5% | 31.0% | +2.5pp | 0.5901 | APPROVED, NULL | Not yet a clean cross-family lift; in-flight Tier 3 will tell. |

Source: docs/signoff_log.md Sections B.1, B.3, and F.
