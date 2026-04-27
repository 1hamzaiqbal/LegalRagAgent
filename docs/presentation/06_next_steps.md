# Next Steps / Open Work

## Immediate (before Tier 3 full-corpus claims)

- [ ] Llama 70b MuSiQue mhd full-corpus N=2400 - why: promote the signed N=200 `multi_hyde_diverse` headline from Tier 2 to Tier 3 - owner / dataset / model / mode: eval/HPC operator / MuSiQue / Llama 70b / `rag_simple` + `multi_hyde_diverse`.

- [ ] Wait for `qwen_full` to land and audit it - why: Qwen3 30B MoE is the live cross-family full-corpus check, currently Tier 2.5 partial only - owner / dataset / model / mode: eval/HPC operator / MuSiQue / Qwen3 30B MoE / `rag_simple` + `multi_hyde_diverse`.

- [ ] Re-run `gemma4_full` and Gemma planning on cluster vLLM after SLURM 55107 finishes - why: OR-served Gemma had runaway-loop serving failures, so cluster vLLM is the usable serving path - owner / dataset / model / mode: cluster operator / MuSiQue / Gemma 4 26B-A4B and Gemma 3/4 planning target / `multi_hyde_diverse` + `iterative_planning_table`.

- [ ] Cross-model check `iterative_planning_table` on Gemma 4 and Qwen - why: Llama 70b is TRENDING-SIG at +8.5pp, but the method needs non-Llama confirmation - owner / dataset / model / mode: eval/HPC operator / MuSiQue / Gemma 4 26B-A4B + Qwen3 30B MoE / `iterative_planning_table`.

## Methodology improvements

- [ ] Run mechanism decomposition with `rag_multi_query` on Gemma 4 26B and Qwen at Tier 2 - compare diversity-only vs HyDE-style passages outside Llama 70b.

- [ ] Scale friend/foe attribution-bias to Tier 2 - current N=30 directional rows show changed outcomes, but paper claims need N=100+ and paired audit.

- [ ] Resolve the BarExam mhd source-status conflict - `signoff_log.md` has a Section B.4 row, while `mcnemar_2026-04-27.md` and the narrative still mark the pair source-pending.

## Story validation

- [ ] BarExam snap-vs-HyDE mechanism - what would refute / strengthen the claim: refute if pred!=snap cases can be fixed without losing the snap-dominated win; strengthen if examples show HyDE helps only on identifiable low-confidence snap cases.

- [ ] Cross-family mhd claim - what would refute / strengthen the claim: refute if Qwen/Gemma full-corpus lands null like Gemma 3 27B N=200; strengthen if Qwen or cluster Gemma shows a signed positive lift.

- [ ] subagent_rag over-abstention story - what would refute / strengthen the claim: refute if a prompt-only rerun still triggers 100% gaps and loses 12pp; strengthen if lower abstention recovers part of the lost EM.

## Implementation fixes

- [ ] Reframe the `subagent_rag` gap-routing prompt - code change needed: make gap detection selective and prevent gap reports from overriding a plausible answer with "Unknown/Not found" unless evidence is genuinely absent.

- [ ] Add run-source reconciliation before presentation export - code change needed: flag rows where `docs/signoff_log.md`, `docs/mcnemar_2026-04-27.md`, and `docs/presentation/05_logs_index.md` disagree on source status.

- [ ] Add an example audit helper for presentation snippets - code change needed: given a detail log plus `idx`, emit the compact JSON fields used in `02_methods_explained.md` so examples stay ground-truthed.

Sources: `docs/signoff_log.md:Sections B-D`, `docs/audits/2026-04-27_llama70b_musique_audit.md:Section 8`, `docs/audits/2026-04-27_barexam_26b_audit.md:Summary Table`, `docs/narrative_2026_04_27.md:Sections 7-8`, `docs/friend_foe_bias_analysis_2026-04-27.md:Summary`.
