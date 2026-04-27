# 2026-04-27 Meeting Brief: Phase 12-14

## Section 1 — Headline

multi_hyde_diverse is the FIRST multi-hop method to lift cross-FAMILY at N=100: Llama 3.3 70b dense reaches 33% vs 21% `rag_simple` (+12pp, McNemar p=0.023), while Gemma 3 27B dense reaches 30% vs 22% (+8pp, p=0.134 trending). Cross-family direction is consistent: MHD beats RAG on both families, with b > c and matching gold-retrieval lift signatures. The cross-family pattern is the story, not either single p-value. Audit trail: `docs/audit_log.md` and live matrix: `docs/validation_log_2026-04-25.md`.

## Section 2 — Paper Core (BarExam)

**Gemma 4 26B-A4B BarExam, N=1195, 8 modes**

| Mode | EM | Delta vs `rag_simple` |
|---|---:|---:|
| `rag_snap_hyde` | **81.17%** | **+3.09pp** |
| `snap_only_in_final` | 80.59% | +2.51pp |
| `llm_only` | 79.75% | +1.67pp |
| `rag_hyde` | 78.91% | +0.83pp |
| `golden_passage` | 78.66% | +0.58pp |
| `subagent_rag` | 78.16% | +0.08pp |
| `rag_simple` | 78.08% | baseline |
| `subagent_hybrid` | 74.23% | -3.85pp |

**Gemma 4 E4B BarExam, N=1195, 8 modes**

| Mode | EM | Delta vs `rag_simple` |
|---|---:|---:|
| `rag_snap_hyde` | **62.18%** | **+3.69pp** |
| `subagent_rag` | 60.92% | +2.43pp |
| `snap_hyde_report` | 60.75% | +2.26pp |
| `rag_hyde` | 60.59% | +2.10pp |
| `subagent_hyde` | 60.17% | +1.68pp |
| `subagent_hybrid` | 58.83% | +0.34pp |
| `rag_simple` | 58.49% | baseline |
| `snap_only_in_final` | 57.82% | -0.67pp |

Takeaway: `rag_snap_hyde` wins on both sizes. The cross-size lift is real, not noise, but the domain is narrow: legal MC, mostly single-hop doctrinal lookup.

## Section 3 — Phase 13.5 Cross-Family MuSiQue

| Model | `rag_simple` | `multi_hyde_diverse` | Delta | b / c | McNemar p | gold_ret MHD/RAG |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.3 70b dense | 21/100 | **33/100** | **+12pp** | 18 / 6 | **0.0227** | 86 / 83 |
| Gemma 3 27B dense | 22/100 | **30/100** | **+8pp** | 15 / 7 | 0.1338 | 91 / 83 |

Mechanism: 3 diverse HyDE candidates pooled with the raw question prevent single-entity commitment bias, raising effective retrieval coverage across candidate paths. Composition still happens at synthesis, so this is a retrieval-diversity intervention, not an oracle decomposition. Cost: about 2 LLM calls per question, same class as `rag_simple` plus 1 HyDE-generation call.

## Section 4 — Phase 14 iter_hyde Negative on Gemma 3 27B

`iter_hyde` on Gemma 3 27B landed at 2/30 = 6.7%, -20pp vs `rag_simple` at 26.7%, despite 93% gold retrieval. Audit `task-mogsjzg4-5okahl` (`codex` REAL_FINDING) traced the mechanism: `iter_hyde` serially conditions each round on prior findings, so early drift can narrow the synthesizer's option set instead of preserving alternatives. Concrete audit example: `idx=2hop__622308_61845` with gold "Mido"; `iter_hyde` round 2 found "Ray Stewart and Mido", but round 3 conditioned only on Ray Stewart and answered wrong, while MHD kept both candidates and answered "Mido". Llama 70b `iter_hyde` N=30 is pending after Groq TPD exhaustion and OpenRouter 429 route limits.

## Section 5 — Methodology Hardening Shipped Today

| Change | Why it matters |
|---|---|
| Pre-flight ChromaDB collection check: `SystemExit(4)` if required collection is empty | Caught local-Mac empty `legal_passages` that produced a bogus 72% advisor BarExam N=50; tagged FAILED-EMPTY-RETRIEVAL, DO NOT CITE |
| Empty-retrieval summary guard | Tags any RAG run with >50% empty retrieval as `_FAILED-EMPTY-RETRIEVAL` |
| MuSiQue answer placeholder fix: `<your answer here>` to `[your answer here]` | Stops Gemma echo failures; post-fix audit shows 0 placeholder echoes |
| Qwen3-32b 68% caveat | 13/100 records truncated mid-`<think>` at the 2048-token cap; measured score likely understates true accuracy |
| Codex CLI 0.124 -> 0.125 -> 0.126.0-alpha.4 | 13 zombie tasks cancelled; codex defaults now usable again |

## Section 6 — Open / Pending

- `iter_hyde` Llama 70b N=30: route exhaustion; Groq daily TPD reset target was 19:00 CDT on 2026-04-27, OpenRouter free route returned 429s.
- `friend_foe_attribution`: built and smoke verified in commit `71bcdc4`; attribution-bias N=30 data not yet collected.
- Cross-domain MHD/`iter_hyde` x Gemma 4 26B BarExam run: requested as pending, but the job ID/ETA was not present in the five required sources, so no job number or result is cited here.
- HotpotQA dataset setup deferred; MuSiQue remains the lead multi-hop dataset for this meeting.

## Section 7 — Reproducibility Footer

Source state at brief-write time: branch `hpc-setup`, commit `bfcacc7`. `docs/audit_log.md` is the single source of truth for every cited result number; `docs/validation_log_2026-04-25.md` is the live execution log. No old pre-prompt-fix BarExam numbers or N=30 MHD mirror-symmetry claims are cited.
