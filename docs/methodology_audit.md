# Methodology Audit

Date: 2026-04-17

Status: this is a pre-fix audit record based on the April 2026 logs and the harness behavior observed before the prompt-cleanup patch landed on 2026-04-17. Use it to understand what was wrong in those runs, not as a claim about the patched local code after that point.

This note audits whether the evaluation methods are:

1. described accurately in [docs/method_index.md](/Users/hamzaiqbal/grad/LegalRagAgent/docs/method_index.md)
2. implemented that way in [eval/eval_harness.py](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py)
3. auditable from the detail logs in `logs/*.jsonl`

The goal is not to rank performance. The goal is to catch places where a method's claimed behavior and its actual runtime behavior diverge, or where the logs are too weak to validate the claim.

## Scope

Primary code references:

- [_fmt()` / prompt formatting in eval_harness.py](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:33)
- [`run_rag_hyde()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:415)
- [`run_rag_snap_hyde()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:493)
- [`_gap_analysis()` / `_run_gap()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:566)
- [`run_subagent_rag()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:1038)
- [`run_snap_hyde_report()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:1084)
- [`run_ce_threshold()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:1890)
- [detail-log record builder in `run_eval()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:2895)

Primary logs checked:

- [logs/eval_rag_hyde_cluster-vllm_20260417_1223_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_rag_hyde_cluster-vllm_20260417_1223_detail.jsonl)
- [logs/eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl)
- [logs/eval_rag_snap_hyde_cluster-vllm_20260413_1102_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_rag_snap_hyde_cluster-vllm_20260413_1102_detail.jsonl)
- [logs/eval_subagent_rag_cluster-vllm_20260416_1720_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_subagent_rag_cluster-vllm_20260416_1720_detail.jsonl)
- [logs/eval_gap_rag_nosnap_cluster-vllm_20260416_0544_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_gap_rag_nosnap_cluster-vllm_20260416_0544_detail.jsonl)
- [logs/eval_ce_threshold_cluster-vllm_20260415_2022_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_ce_threshold_cluster-vllm_20260415_2022_detail.jsonl)
- [logs/eval_snap_hyde_report_cluster-vllm_20260417_1418_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_snap_hyde_report_cluster-vllm_20260417_1418_detail.jsonl)

## Summary

The current method index is mostly directionally correct, but there are several methodology-critical mismatches:

- HyDE generation is contaminated by answer-formatting and often emits explicit multiple-choice answers.
- "Report-only" modes often leak answer labels back into the final call through the report text.
- Gap-family runs are effectively one-gap runs in current practice, even if the descriptions read more generally.
- The detail logs are not rich enough to prove what the final model actually saw.
- `rag_hyde` currently points to a broken older trace in the method index.

## Main Findings

### 1. HyDE queries are not clean retrieval passages

This is the biggest methodology issue.

`run_rag_hyde()` and `run_rag_snap_hyde()` both build HyDE inputs from the formatted eval question, not from a retrieval-specific question view. For BarExam, the formatted question includes the multiple-choice choices and the instruction `Provide your answer as: Answer: (X)` in [format_question_prompt()`](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_config.py:112).

That means the HyDE generator is not being asked from a neutral retrieval interface. It is being shown an answer-oriented multiple-choice prompt. In `rag_snap_hyde`, the model is additionally shown the full snap answer before HyDE generation [eval_harness.py](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:501).

Observed in logs:

- `rag_hyde` 2026-04-17 N=200: `200/200` `hyde_passage` values contain `Answer:`
- `rag_hyde` 2026-04-17 N=200: `15/200` `hyde_passage` values are exactly `Answer: (X)` with no useful passage text
- `rag_hyde` 2026-04-15 full N=1195: `1195/1195` `hyde_passage` values contain `Answer:`
- `rag_hyde` 2026-04-15 full N=1195: `1182/1195` `hyde_passage` values are exactly `Answer: (X)`
- `rag_snap_hyde` 2026-04-13 full N=1195: `890/1195` `hyde_passage` values contain `Answer:`
- `rag_snap_hyde` 2026-04-13 full N=1195: `258/1195` `hyde_passage` values are exactly `Answer: (X)`

Representative examples from the logs:

- `rag_hyde`, `mbe_339`: `Answer: (D)`
- `rag_hyde`, `mbe_809`: `Answer: (B)` followed by a passage
- `rag_snap_hyde`, `mbe_20`: `Answer: (C)`
- `rag_snap_hyde`, `mbe_0`: `Answer: (B)` followed by a passage

Why this does not make methodological sense:

- HyDE is supposed to generate a passage-like semantic query, not a committed answer label.
- The answer letter `(A)/(B)/(C)/(D)` is not a corpus feature. It is an eval artifact.
- If the retrieval query contains the model's candidate answer, the method is no longer "question -> hypothetical passage -> retrieve". It is closer to "candidate answer -> retrieve evidence that matches the candidate answer."
- This makes interpretation muddy. A gain could come from answer anchoring or answer-conditioned retrieval, not from a genuinely better legal-reference query.
- Exact `Answer: (X)` outputs are especially bad because the retrieval query becomes almost content-free.

Impact on the method index:

- The `rag_hyde` description in [method_index.md](/Users/hamzaiqbal/grad/LegalRagAgent/docs/method_index.md:174) currently reads like a clean question-only HyDE setup.
- That is not what the logs show in practice.
- The example trace currently points to [logs/eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl), which is the worst possible trace to use because its HyDE outputs are almost entirely just answer labels.

Recommended fix:

- Separate answer-time question formatting from retrieval-time question formatting.
- Add a retrieval-specific formatter that removes the final answer instruction.
- For MC tasks, do not allow `Answer: (X)` lines in HyDE output; strip them before retrieval.
- Log the exact `retrieval_queries` used so this is auditable per run.

### 2. "Report-only" modes are not actually answer-free

The docs describe `snap_hyde_report` as a report-only variant in which the final call sees one synthesized report and not the snap answer or raw passages [method_index.md](/Users/hamzaiqbal/grad/LegalRagAgent/docs/method_index.md:581).

The code tries to enforce this. The report writer is instructed: `No answer letters` [eval_harness.py](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:1112).

But the logs show this instruction fails often:

- `snap_hyde_report` 2026-04-17 N=200: `102/200` top-level `report` fields contain `Answer:`
- `subagent_rag` 2026-04-16 full N=1195: `60/1195` questions contain at least one sub-report with `Answer:`

This matters because the final model is then effectively being shown an answer proposal through the report text, even when the method is described as report-only or de-anchored.

Representative `snap_hyde_report` examples:

- `mbe_809`: report ends with `Answer: (A)`
- `mbe_73`: report ends with `Answer: (C)`
- `mbe_118`: report ends with `Answer: Irrelevant`

Recommended fix:

- Sanitize reports before the final call by stripping `Answer:` lines and answer-letter patterns.
- Add a `report_sanitized` field or `report_contains_answer` flag to the detail logs.
- Update descriptions of report-only methods to say they are intended to be answer-free, but current runs are not reliably so unless sanitization is applied.

### 3. Gap-family methods are effectively one-gap methods

The current gap analyzer is written to identify the single most important gap and only emit a second gap if it is truly independent [eval_harness.py](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:573).

In the recent full runs checked here:

- `subagent_rag` 2026-04-16 full N=1195: every row had exactly `1` gap
- `gap_rag_nosnap` 2026-04-16 full N=1195: every row had exactly `1` gap

So the current operational behavior is not "multi-gap investigation" in practice. It is a one-gap architecture.

The method index is not outright false here, but it is underspecified. The current implementation should be described as:

- snap
- one-gap analysis in practice
- one targeted retrieval/report step
- final answer

Recommended fix:

- Update the method descriptions to say the current prompt is effectively a one-gap prompt.
- If multiple gaps are truly desired, change the analyzer prompt and log `n_gaps_requested` and `n_gaps_returned`.

### 4. The detail logs are not rich enough to audit visibility claims

This is the root auditability problem.

The detail record builder in `run_eval()` stores the raw question, choices, gold answer, and whatever each mode returns [eval_harness.py](/Users/hamzaiqbal/grad/LegalRagAgent/eval/eval_harness.py:2895). That is useful, but insufficient for methodology validation.

What is missing:

- the exact formatted question shown to the model
- the exact retrieval query or query list
- the rerank query
- the final user prompt or a structured summary of it
- a machine-readable record of which artifacts were visible to the final model
- sanitation flags such as `hyde_contains_answer` or `report_contains_answer`

As a result, claims like "the final call did not see the snap answer" are only provable by reading code, not by reading logs.

Recommended fix:

- Add per-row fields:
  - `formatted_question`
  - `retrieval_queries`
  - `rerank_query`
  - `final_context_fields`
  - `final_prompt_preview`
  - `hyde_contains_answer`
  - `report_contains_answer`
- Keep them truncated if needed, but log enough to verify the method.

### 5. The method index currently points to at least one misleading trace

`rag_hyde` in [method_index.md](/Users/hamzaiqbal/grad/LegalRagAgent/docs/method_index.md:191) points to the April 15 full run.

That trace is not a good canonical example of the method because:

- `1195/1195` HyDE passages contain `Answer:`
- `1182/1195` HyDE passages are exactly `Answer: (X)`

If the method index is meant to document actual methodology, that trace should be replaced with the newer April 17 run or marked as historically broken.

There is also minor doc drift for `snap_hyde_report`: the method index still says `No detail log found` [method_index.md](/Users/hamzaiqbal/grad/LegalRagAgent/docs/method_index.md:600), but the log now exists at [logs/eval_snap_hyde_report_cluster-vllm_20260417_1418_detail.jsonl](/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_snap_hyde_report_cluster-vllm_20260417_1418_detail.jsonl).

## Methods That Looked Basically Aligned

Not everything is broken. A few recent methods looked consistent between docs, code, and logs:

- `ce_threshold`: the logs show the documented branch behavior clearly. In the April 15 full run, `735/1195` rows routed to `snap_only` with `2` calls, and `460/1195` routed to `rag` with `3` calls.
- `gap_rag_nosnap`: the code does hide snap from the final prompt path, and the logs show the expected `3`-call pattern with one gap per row. The audit problem is not that the method is wrong; it is that the logs do not explicitly encode final visibility.
- `subagent_rag`: the code and logs match on the high-level call pattern for the current one-gap regime. The main issue is report leakage, not a total mismatch in control flow.

## Immediate Patch List

If the goal is "the logs should prove the methodology," these are the highest-value fixes:

1. Split answer formatting from retrieval formatting.
2. Strip `Answer:` lines from HyDE outputs before retrieval.
3. Strip `Answer:` lines from reports before the final call.
4. Log `retrieval_queries`, `rerank_query`, and `final_context_fields`.
5. Replace broken/historical example traces in `method_index.md`.
6. Mark gap-family methods as effectively one-gap unless and until the analyzer is widened.

## Bottom Line

The current repo has a real methodology-audit problem:

- some methods are cleaner in prose than they are in runtime behavior
- some logs are cleaner in appearance than the actual prompts that produced them
- at least one documented example trace is a broken historical run

The HyDE answer-label contamination is the most serious issue. If that is not cleaned up, then `rag_hyde` and `rag_snap_hyde` should not be described as clean passage-generation retrieval methods.
