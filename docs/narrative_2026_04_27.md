# Narrative — Legal/Multi-hop RAG Method Investigation (2026-04-27)

## 1. Why this work
Legal RAG has two distinct evaluation regimes in this project. BarExam is legal multiple choice: the system must retrieve or reason over legal doctrine and choose one of the answer options. MuSiQue is multi-hop open QA: the system must compose facts across several linked entities or passages. The important empirical split is that BarExam is now mostly a solved harness problem for the stronger Gemma 4 cluster models, while MuSiQue remains unstable enough that method choice changes the answer.

That split matters for legal practitioners. A real legal assistant needs both single-fact lookup and multi-step composition: finding a rule, connecting it to a fact pattern, tracking an exception, and then producing a concise answer. A method that improves legal MC can still fail on multi-hop QA, and a multi-hop method can add cost or complexity without improving reliability.

The question for this investigation is therefore practical: which retrieval and prompting modifications actually help, at what scale, and across which model families? The answer is not "more agentic steps." The current audited evidence says simple pooled HyDE-style retrieval is the credible multi-hop lead, while the BarExam winner is a different snap-plus-HyDE family.

## 2. What's been tried in the literature (brief lit review)
The surrounding literature gives several plausible levers. HyDE, associated with Gao et al., generates a hypothetical document or answer-bearing passage and embeds that text for retrieval; the goal is to move the query into the semantic neighborhood of the evidence. Multi-query retrieval instead generates several alternate queries from the same question, pools retrieved passages, and asks the model once.

Planning-style methods such as Plan-and-Solve and ReAct decompose a problem before answering. In RAG settings, this becomes iterative retrieve-and-refine: ask a sub-question, retrieve, summarize the finding, decide whether another step is needed, and then synthesize. Subagent and multi-step retrieval methods follow the same instinct, assigning intermediate reasoning or evidence gathering to smaller internal calls before the final answer.

This project also tested snap-then-RAG: let the model think first, then retrieve against the snap hypothesis or its gaps. Self-RAG and RAFT point toward training or self-reflection variants, but the present work is an evaluation-first method investigation, not a new training run. The core comparison is between simple retrieval-shaping methods and more complex decomposition or agentic methods under the same harness.

## 3. What we built (this work — the setup)
The harness lives in `eval/eval_harness.py` with mode registration driven by `EVAL_MODES`; the current code exposes 61 configured method modes, including the BarExam and MuSiQue modes used here. The target set for the meeting is four models after dropping non-target families: Gemma 4 26B-A4B as the cluster headline MoE, Gemma 4 E4B as the smaller Gemma cluster model, Llama 3.3 70b dense as the paper-headline MuSiQue model, and Qwen3 30B MoE as the third model class for cross-family checking.

The two datasets are deliberately different. BarExam is legal multiple choice, with the Tier 3 full corpus at N=1195. MuSiQue is multi-hop QA, with full corpus N=2400. The statistics now match that split: paired exact two-sided McNemar tests for method deltas, plus paired bootstrap 95% confidence intervals where reported in `docs/mcnemar_2026-04-27.md`.

The post-calibration tier discipline is: N=100 is directional, N=200+ is citeable if paired and audited, and full corpus is preferred. The guardrails are now as important as the raw methods: pre-flight gates, empty-retrieval protection, paired tests, and per-entry audits in `docs/compiled_results.md` and `docs/signoff_log.md`.

## 4. What we found (confirmed)
### 4.1 BarExam (legal MC, Tier 3 N=1195)
The BarExam result is the cleanest legal-domain finding. `rag_snap_hyde` wins across the two Gemma 4 sizes: Gemma 4 26B-A4B improves from 78.08% to 81.17%, a +3.09pp lift, and Gemma 4 E4B improves from 58.49% to 62.18%, a +3.69pp lift. The method matrices across both Gemma 4 sizes point to the same snap+HyDE family as the consistent winner. The sign-off gate calls the cross-size headline approved, with normal BarExam caveats around low exact-gold retrieval and a few already-counted parser misses.

### 4.2 MuSiQue (multi-hop QA, Tier 2 N=200, paired McNemar)
The MuSiQue headline is narrower but stronger statistically. On Llama 3.3 70b dense, `multi_hyde_diverse` reaches 35.5% versus `rag_simple` at 27.5%, a +8.0pp lift with McNemar p=0.0195 SIG. The same comparison on Gemma 3 27B dense is 31.0% versus 28.5%, +2.5pp with p=0.5901 NS, so this is not universal across dense model families. Qwen3 30B MoE has only N=100 directional evidence so far: 28.0% versus 24.0%, +4pp, with full corpus running. Other Llama 70b methods do not beat baseline: `rag_multi_query` is +1.5pp p=0.728 NS, `rag_snap_hyde` is -3.5pp p=0.36 NS, `iter_hyde` is -3.0pp p=0.47 NS, and `subagent_rag` actively hurts at -12.0pp p=0.0007.

### 4.3 Mechanism decomposition (Tier 2 confirmed)
The Tier 2 mechanism story is simple: the +8.0pp Llama mhd gain decomposes into +1.5pp from multi-query-style diversity alone and about +6.5pp from HyDE-style answer-bearing passages. Diversity alone is not significant; answer-bearing passages do roughly 80% of the observed lift.

## 5. The story we want to tell
The narrative arc is that multi-hop QA is hard in a specific way. The failure is not just "need more steps"; several complex methods either do nothing or make performance worse. On the signed Llama 70b MuSiQue matrix, `iter_hyde` and `rag_snap_hyde` are neutral-to-negative and non-significant, while `subagent_rag` is significantly negative. That makes the result more interesting: the winning multi-hop method is not a planner, agent, or serial refinement loop.

`multi_hyde_diverse` is simple. It generates three diverse HyDE-style hypothetical answer passages, pools retrieval over those passages plus the raw question, and then answers once. The evidence says the lift comes mostly from those answer-bearing passages, not from query diversity by itself. This is the core mechanism claim worth making.

The second story is domain specificity. BarExam likes snap+HyDE, and the cross-size Gemma 4 legal-MC result supports that. But on MuSiQue Llama 70b, `rag_snap_hyde` trends below baseline and is not significant. So the paper should avoid claiming a universal RAG trick. The stronger claim is sharper: different tasks expose different bottlenecks, and multi-hop composition rewards parallel answer-bearing retrieval more than serial agentic decomposition.

## 6. Models — keep / drop / problematic
Gemma 4 26B-A4B: KEEP. This is the BarExam cluster headline model. `compiled_results.md` gives the cross-size `rag_snap_hyde` row an audit verdict of MINOR: the `rag_simple` baseline has 2 sampled null predictions with empty retrieval, but the evidence is otherwise present. Its 8-mode BarExam matrix is a useful Tier 3 anchor, and the caveats do not compromise the method comparison.

Gemma 4 E4B: KEEP-WITH-CAVEAT. The smaller Gemma is useful because it reproduces the same BarExam winner. The cross-size row is MINOR: sampled records are parse-clean, but the full scan found 1 raw null `rag_snap_hyde` prediction and low exact-gold retrieval. Keep it as a scale check, but cite with the compiled-results caveat and the sign-off status in mind.

Llama 3.3 70b dense: KEEP. This is the paper-headline MuSiQue model. The mhd row is MINOR because of 1/20 sampled `predicted_answer='None'`, full-scan 3/200, and 1 routed fallback; the baseline has 6/200 `None` predictions. These are already counted as misses, and the paired matrix is signed.

Qwen3 30B MoE: KEEP-WITH-CAVEAT. It is the third model class, but current MuSiQue mhd evidence is only N=100. `compiled_results.md` marks the mechanism row MINOR because 2/20 sampled outputs are empty/null and 1 sampled retrieval is empty. Do not drop it now, but do not promote it beyond directional until the full N=2400 run lands and passes audit.

## 7. What's still open / next steps
The Llama 70b N=200 confirmation is done, and it is the current paper-grade MuSiQue result. No Llama full-corpus N=2400 MuSiQue run is listed in the in-flight table; the in-flight Tier 3 disambiguation is instead on Gemma 4 26B-A4B and Qwen3 30B MoE. Those full-corpus runs will matter because the Gemma 3 27B N=200 null prevents a clean cross-family claim today.

The current in-flight list has `gemma4_full` mhd-pair on Gemma 4 26B-A4B at N=2400 MuSiQue and `qwen_full` mhd-pair on Qwen3 30B MoE at N=2400 MuSiQue, both pending full-run audit. For BarExam, SLURM 55107 is running mhd plus `iter_hyde` on Gemma 4 26B-A4B at N=200. That is important because it tests whether the MuSiQue mhd idea transfers back to legal MC, where `rag_snap_hyde` is already the signed winner.

The main open scientific question is cross-domain transfer. If mhd helps BarExam too, it becomes a broader retrieval default. If it does not, the story strengthens around task-specific retrieval shaping: snap+HyDE for legal MC, pooled answer-bearing HyDE for multi-hop QA.

## 8. Reproducibility / where to find the data
Use `docs/signoff_log.md` as the cite-or-not gate. It is the top-level approval file for which rows are paper-grade, caveated, pending, or rejected. Use `docs/compiled_results.md` for per-entry audited details, direct log paths, and model-specific audit verdicts. Use `docs/mcnemar_2026-04-27.md` for the paired MuSiQue exact McNemar tests and bootstrap intervals. Use `docs/mhd_mechanism_2026-04-27.md` as historical mechanism context, but prefer the updated Tier 2 mechanism rows in the McNemar and compiled-results docs for paper claims. Use `docs/audit_log.md` for the BarExam Tier 3 source-of-truth and historical hardening record.

Raw run summaries are in `logs/experiments.jsonl`; per-question records are in `logs/eval_*_detail.jsonl`. The vector store is `chroma_db/`, which is gitignored and environment-dependent, so cite the audit docs rather than assuming local retrieval state is portable.

Branch: `hpc-setup`. Current HEAD from `git rev-parse --short HEAD`: `d3fdd87`.

Footnote: `CLAUDE.md` and `compiled_results.md` agree on the headline numbers used here. One metadata discrepancy remains: `CLAUDE.md` labels the E4B table as "6 modes landed," while `compiled_results.md` lists 8 audited E4B BarExam modes; this narrative follows `compiled_results.md`.
