# Lessons learned — 2026-04-25/26 coverage wave + MuSiQue API

Failures hit during the run + principles to internalize so they don't repeat.

## Failure 1: OpenRouter slow-route hangs

**What happened:** API jobs (especially RAG modes with longer context) would stall at 1-9 questions for 30+ min while one specific OpenRouter downstream provider hung the request. With the default 300s OpenAI client timeout + 0 retries, a single slow route blocked the entire run.

**Root cause:** OpenRouter routes the same model (`google/gemma-4-26b-a4b-it`) to multiple downstream providers; some are slow. The harness has no per-call timeout floor.

**Fix applied:** llm_config.py timeout 300s → 90s, max_retries 0 → 1.

**Principle:** Whenever using a multi-provider router (OpenRouter, Together fallbacks, etc.), set per-call timeouts to a small multiple of expected latency, not the safety-default. Slow routes WILL happen; design for fail-fast, not patient wait.

## Failure 2: Detail log only written at end-of-mode

**What happened:** When MuSiQue rag_snap_hyde N=30 hung at q30 and was killed, the harness had no detail log because `eval_harness.py` only writes the JSONL after the mode loop completes. Lost all 29 questions of intermediate data.

**Root cause:** Single end-of-run write pattern in `eval_harness.py`.

**Fix not yet applied.** Worked around by computing partials from the stdout log.

**Principle:** Long-running iterations should write checkpoint files incrementally (every 10-50 items). For evals: append per-question JSONL row immediately, not at end-of-mode. Crash recovery is a basic robustness property.

## Failure 3: Cluster wallclock killed mode mid-flight (54173)

**What happened:** E4B-1 was given 28h walltime for 4 modes. Modes 1+2 finished, mode 3 (llm_only) died at 1155/1195 due to wallclock, mode 4 (golden_passage) never started. Two cells lost.

**Root cause:** Bundled multi-mode SLURM jobs share a single walltime budget. When one mode runs slower than estimated, downstream modes don't get a chance.

**Principle:** Either (a) generously over-budget walltime (1.5-2x estimate) for bundled multi-mode jobs OR (b) split into one job per mode so wallclock failure only loses one cell. For meeting-critical runs, prefer (b).

## Failure 4: ChromaDB compaction crash on concurrent NFS access

**What happened:** Embed-musique 54260 ran while 7 vLLM eval jobs held read locks on chroma_db on engrfs (NFS). chromadb compaction failed on the first batch add: `chromadb.errors.InternalError: Error in compaction: Failed to apply logs to the metadata segment`.

**Root cause:** chromadb's WAL/metadata writes don't coexist well with concurrent reads on NFS-backed storage.

**Workaround applied:** Built in-row BM25 retrieval for MuSiQue — bypasses chroma entirely.

**Principle:** Treat NFS-backed sqlite/chroma as single-writer. Schedule write operations after read-heavy phases finish, OR use isolated dirs per write task and merge later.

## Failure 5: Pipe-buffering on `tail`/`grep`

**What happened:** Initial smoke run via `... | tail -25` looked completely hung — 0 bytes for 5+ min. The harness was actually running fine; `tail` buffered until pipe close.

**Fix applied:** Run with `PYTHONUNBUFFERED=1 ... > /tmp/x.log 2>&1` and view via `tail -F` separately. Don't pipe through `tail` or `grep` for live monitoring.

**Principle:** For background eval runs, ALWAYS redirect to a file with `PYTHONUNBUFFERED=1`. Tail the file separately. Pipe-buffering is invisible until you check what was lost.

(This was already noted in CLAUDE.md but I forgot it. The principle: re-read CLAUDE.md launch patterns before firing the first eval of a session.)

## Failure 6: Path mismatch between two repo clones

**What happened:** Embed job 54190 ran from `LegalRagAgent` (data repo) but my `fast_embed.py` musique-corpus addition was pushed to `LegalRagAgent-clean` (code repo). Result: "Unknown corpus: musique" silent failure.

**Root cause:** Two cluster clones diverged: chroma_db lives in one path, latest code in another, SLURM scripts hard-code the old path.

**Fix applied:** scp'd `fast_embed.py` to the data repo path manually.

**Principle:** Don't maintain two clones of the same repo unless absolutely required. If you must, use symlinks for canonical files (utils/, eval/) so they stay in sync.

## Failure 7: Bare `--gpus 1` defaulted to a100 GRES

**What happened:** Embed-musique 54218 with `#SBATCH --gpus 1` got `gres/gpu:a100=1` from SLURM — but a40 nodes don't have a100 GPUs, so the job sat PENDING for hours with StartTime tomorrow.

**Fix applied:** Changed to `#SBATCH --gres=gpu:a40:1`.

**Principle:** Always specify GPU TYPE explicitly via `--gres=gpu:<type>:N`. The bare `--gpus N` is partition-config-dependent and breaks when partition defaults change.

## Failure 8: Stale wakeup notifications stacking

**What happened:** Multiple ScheduleWakeup calls firing the same babysit prompt ~15 min apart, even when state hadn't changed enough to warrant action. Plus stale Monitor timeouts triggering after the actual completion event already fired.

**Principle:** ScheduleWakeup creates a NEW pending wake; old ones still fire. To avoid stale wake floods: (a) only call ScheduleWakeup once per turn, (b) treat each wake's prompt as authoritative for "what should I check now" not "what was true 15 min ago", (c) on stale Monitor timeouts, just acknowledge — don't re-arm if event already fired.

## Meta-principles

1. **Fail-fast over fail-patient.** Default timeouts are usually too patient. Cut them to 1.5x expected latency.
2. **Incremental checkpoints over end-of-run writes.** Crash recovery is a feature.
3. **Explicit > implicit** for cluster GRES, paths, env vars. Defaults bite.
4. **Re-read CLAUDE.md launch-pattern guidance before firing the first eval of a session.** Many of the above are already documented there but I forgot.
5. **One repo path per machine.** Two clones = two truths = silent divergence.
