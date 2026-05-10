# CaseHOLD Option-Table Held-Out Probe - 2026-05-10

## Purpose

Test whether the existing compact option-conversion route,
`adaptive_snap_hyre_option_table`, improves the remaining CaseHOLD bottleneck on
the same held-out rows 200-249 used by the controller validation.

This is a targeted follow-up to the completion audit caveat: CaseHOLD retrieval
and candidate exposure can improve without reliably improving final answer
accuracy, so the unresolved question is whether an option-level selector can
convert retrieved evidence into the right displayed holding.

## Submission

- Active job: `67530`
- Superseded job: `67519` failed in preflight before method execution because
  the cluster launch did not see `adaptive_snap_hyre_option_table` in
  `EVAL_MODES`; the mode was confirmed present in the checkout and importable
  inside `.venv`, then resubmitted with explicit `sbatch --export`.
- Superseded job: `67520` failed in preflight for the same reason because the
  Slurm script defaulted `REPO` to the non-adaptive cluster checkout. Job
  `67521` explicitly exports `REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre`
  and `DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent`.
- Superseded job: `67521` passed mode and Chroma preflight but failed during
  vLLM startup on an A40 with CUDA OOM: Gemma 4 26B used 44.39 GiB of a
  44.42 GiB device and failed on a final 22 MiB allocation. Job `67527` pins
  the run to `h100-2405`.
- Superseded job: `67527` was pinned to `h100-2405` but stayed pending because
  the requested node was unavailable/reserved; it was cancelled after the API
  fallback started.
- Job `67528` is the active OpenRouter fallback: it uses the same cluster
  checkout and Chroma collection, but sets `USE_VLLM=0` and
  `PROVIDER=or-gemma4-26b` to avoid the H100 queue and A40 vLLM OOM.
- Superseded job: `67528` entered the eval loop with OpenRouter but failed
  after four rows with `CUDA error: device-side assert triggered`, likely from
  the local embedding/reranking path on the allocated GPU.
- Superseded job: `67530` kept the same OpenRouter LLM path and cluster Chroma
  collection,
  but forces local retrieval/reranking models to CPU with
  `EMBEDDING_DEVICE=cpu` and `CROSS_ENCODER_DEVICE=cpu`; it still failed after
  five rows with a cross-encoder index error.
- Next fallback should set `DISABLE_CROSS_ENCODER=1` so the option-table prompt
  can be tested over dense candidate tables without the brittle cross-encoder
  reranking path.
- Dataset: `casehold`
- Mode: `adaptive_snap_hyre_option_table`
- Provider path: OpenRouter Gemma 4 26B API with CPU local retrieval/reranking
- Slice: `--questions 250 --sample-start 200 --sample-end 250`
- Effective evaluated rows: 50
- Retrieval: `k=5`
- Tag: `casehold-option-table-heldout-or-gemma4-26b-api-cpu-casehold-q250-start200-end250-k5`

## Integration Gate

Before promoting results:

1. Confirm `sacct` completion and exit code for job `67530`.
2. Inspect `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/67530.out` for
   Tracebacks, API/rate-limit errors, parsing failures, empty retrieval, or
   timeout; also confirm the CUDA assert from `67528` does not recur.
3. Run `scripts/analyze_detail_flags.py` on the landed detail JSONL.
4. Run `scripts/audit_adaptive_hyre_logs.py` on the landed detail JSONL.
5. Compare against the held-out CaseHOLD rows already validated:
   - `rag_simple`: 34/50 = 68.0%
   - `rag_rewrite`: 38/50 = 76.0%
   - `adaptive_snap_hyre_diverse`: 39/50 = 78.0%
