# Log trustworthiness audit — 2026-04-26 (round 3, deep)

Subagent-produced audit specifically targeting "are our logs trustworthy" —
the user's worry that we might launch evals producing useless data.

**Verdict:** all cluster runs trustworthy + most API runs trustworthy. **7 silent-failure runs identified for deletion** + **3 hardening fixes applied this commit** to prevent recurrence.

## Trustworthy runs (use freely)

All 12 cluster `cluster-vllm` runs from 2026-04-26 → CLEAN (0 errors, no mode collapse, full schema). Need to be appended to experiments.jsonl (see recommendation #5).

API runs:
- `gemma-4-26b-a4b-it` (or-gemma4-26b) MuSiQue + BarExam — CLEAN
- `groq-llama70b` 81% (BarExam N=100) — CLEAN
- `groq-scout` 67% (BarExam N=100) — CLEAN

## SILENT FAILURES — 7 detail logs to delete

All have 100% records with `error: "Error code: 401"` or similar — accuracy=0.0 is meaningless.

| Detail log | Provider | Cause |
|---|---|---|
| eval_llm_only_groq-llama70b_20260426_1917_detail.jsonl | groq-llama70b | GROQ_API_KEY not set |
| eval_llm_only_deepseek_20260426_1917_detail.jsonl | deepseek | DEEPSEEK_API_KEY missing |
| eval_llm_only_groq-{qwen,llama70b,kimi,scout}_20260426_1923_detail.jsonl | groq-* | First key write didn't take |
| eval_llm_only_groq-kimi_20260426_1925_detail.jsonl | groq-kimi | 404 model not found |

## Output sanity

- gemma-4-26b-a4b: clean, 0 think-tags, no mode collapse
- cluster runs: clean — repeated stylistic prefixes are within-mode regularity, not collapse
- **groq-qwen `qwen3-32b`: 100% records have raw `<think>...</think>` blocks**; 2/5 truncated mid-think (no close tag, no `Answer:` extracted)
- subagent_hybrid: 2 records hit max_new_tokens mid-reasoning (~30k tokens)
- One record had `Answer: (Source 5)` instead of letter — extraction returned None correctly

## Hardening fixes applied this commit

### Fix 1: Pre-flight smoke gate (main loop)
Before iterating questions, fire ONE test call per provider. On 401/403/404 abort with `SystemExit(2)` BEFORE any per-question work. Skipped for `custom`/`cluster-vllm`.

### Fix 2: Per-question circuit breaker
Track consecutive errors. After 5 consecutive errors, abort with `SystemExit(3)` rather than logging garbage for all 100 questions. Resets on any successful question.

### Fix 3: Block summary write on high error rate
If `error_rate > 50%` at end of run, tag experiments.jsonl row as `<tag>_FAILED-do-not-use` + add `error_count`/`error_rate` fields. Detail log still written for forensics; post-hoc analysis won't cite as legitimate.

### Fix 4: Strip `<think>...</think>` tags before answer extraction
`_strip_think_tags()` removes closed think-blocks before `extract_answer_*`. Open (truncated) think-blocks left alone so failure is detectable. Unlocks Qwen3 family.

## Deferred recommendations (low priority)

- Sync 12 cluster detail logs into experiments.jsonl
- Bump max_new_tokens for subagent_hybrid (2/1195 record loss = 0.17%)
- Add `model` field to per-record detail output (forensic robustness)
