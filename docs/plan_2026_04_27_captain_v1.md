# Captain's Plan 2026-04-27 (5-hour horizon to meeting)

## 1. Coverage matrix as of 2026-04-27 09:45 CDT

Scope: only the four target models, only BarExam and MuSiQue, only the listed method modes. Cells below are selected from `logs/experiments.jsonl` by highest clean `n_questions`, with ties broken by latest timestamp. Excluded: `N<100`, tags/run IDs containing `smoke`, `_FAILED`, `FAILED-EMPTY-RETRIEVAL`, `invalid`, or zero-call operational failures (`avg_llm_calls=0` with zero tokens). Tier key: Tier 1 = N=100 directional, Tier 2 = N=200, Tier 3 = N>=800/full.

Note: some audit docs cite post-fix re-score values that are not present as matching summary rows in `logs/experiments.jsonl`. Per this plan's grounding rule, the matrix quotes only the JSONL summary rows.

| Family | Mode | Gemma 4 26B-A4B BarExam | Gemma 4 26B-A4B MuSiQue | Gemma 4 E4B BarExam | Gemma 4 E4B MuSiQue | Llama 3.3 70B BarExam | Llama 3.3 70B MuSiQue | Qwen3 30B MoE BarExam | Qwen3 30B MoE MuSiQue |
|---|---|---|---|---|---|---|---|---|---|
| snap+HyDE | `rag_snap_hyde` | Tier 3: 75.40% (N=1195) | EMPTY | Tier 3: 58.41% (N=1195) | EMPTY | Tier 2: 76.50% (N=200) | Tier-1 only @ N=100: 21.00% | EMPTY | EMPTY |
| snap+HyDE | `snap_only_in_final` | Tier 3: 75.15% (N=1195) | EMPTY | Tier 3: 54.81% (N=1195) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| snap+HyDE | `snap_hyde_report` | Tier 3: 76.57% (N=1195) | EMPTY | Tier 2: 66.00% (N=200) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| HyDE / diversity | `rag_hyde` | Tier 3: 74.64% (N=1195) | EMPTY | Tier 3: 57.74% (N=1195) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| HyDE / diversity | `multi_hyde_diverse` | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | Tier-1 only @ N=100: 33.00% | EMPTY | Tier-1 only @ N=100: 28.00% |
| HyDE / diversity | `iter_hyde` | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| Subagent | `subagent_rag` | Tier 3: 75.73% (N=1195) | EMPTY | Tier 3: 57.24% (N=1195) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| Subagent | `subagent_hyde` | Tier 3: 76.57% (N=1195) | EMPTY | Tier 2: 62.50% (N=200) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| Subagent | `subagent_hybrid` | Tier 3: 73.39% (N=1195) | EMPTY | Tier 2: 63.50% (N=200) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| Iterative / planning | `iterative_planning_table` | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |
| Iterative / planning | `advisor_planning_table` | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY | Tier-1 only @ N=100: 23.00% | EMPTY | EMPTY |
| Baselines | `rag_simple` | Tier 3: 71.80% (N=1195) | EMPTY | Tier 3: 55.73% (N=1195) | EMPTY | EMPTY | Tier-1 only @ N=100: 21.00% | EMPTY | Tier-1 only @ N=100: 24.00% |
| Baselines | `llm_only` | Tier 3: 74.31% (N=1195) | EMPTY | Tier 3: 55.48% (N=1195) | EMPTY | Tier-1 only @ N=100: 81.00% | EMPTY | Tier-1 only @ N=100: 70.00% | EMPTY |
| Baselines | `golden_passage` | Tier 3: 74.98% (N=1195) | EMPTY | Tier 3: 62.18% (N=1195) | EMPTY | EMPTY | EMPTY | EMPTY | EMPTY |

The failed BarExam `advisor_planning_table` Llama row is intentionally excluded: `logs/experiments.jsonl` tags it `api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL`, with `empty_retrieval_rate=1.0`.

## 2. Critical gaps to fill in 5 hours

Highest-value gaps:

1. MuSiQue is not yet paper-grade in the JSONL matrix. Existing landed target-model MuSiQue cells are only Tier 1: Llama `rag_simple` 21.00% (N=100), Llama `multi_hyde_diverse` 33.00% (N=100), Llama `rag_snap_hyde` 21.00% (N=100), Llama `advisor_planning_table` 23.00% (N=100), Qwen `rag_simple` 24.00% (N=100), and Qwen `multi_hyde_diverse` 28.00% (N=100).
2. Gemma 4 26B-A4B and Qwen3 30B MoE MuSiQue full-corpus `multi_hyde_diverse` pairs are already running, so the key gap is not to duplicate them; it is to wait for those Tier 3 rows to land.
3. Llama 70B MuSiQue needs Tier 2 confirmation for the `multi_hyde_diverse` signal. Today that is rate-limit constrained: Groq Llama 70B likely hit TPD from `snap_hyde`, `friend_foe`, and `multi_query`; OpenRouter `or-llama70b` is a Venice 429 route.
4. Gemma 4 E4B has no MuSiQue target-cell coverage at N>=100. If the cluster slot is not spent on the timed-out BarExam job, an E4B near-full MuSiQue run is the cleanest way to add a fourth-model size point.
5. Planning/iterative cells are mostly empty. Do not chase them broadly inside 5 hours; one 26B BarExam `iter_hyde` rerun is enough because SLURM 55040 already attempted it and timed out.

## 3. Recommended launches (pick 3-5)

1. Requeue the timed-out cluster BarExam 26B `multi_hyde_diverse` / `iter_hyde` Tier 2 run with an 8-hour walltime.

```bash
ssh wustl 'cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean && sbatch -t 8:00:00 --export=ALL,MODEL=google/gemma-4-26B-A4B-it,MODES="multi_hyde_diverse iter_hyde",N_QUESTIONS=200,SEED=42,TAG_SUFFIX=bar-26b-mhd-iter-n200-rerun scripts/hpc/slurm_mhd_iterhyde_barexam.sh'
```

Slot: one free cluster slot. Dataset cap: `N_QUESTIONS=200` BarExam. ETA: risky 4-6 hours; previous SLURM 55040 timed out at 3 hours, so this may still miss the meeting, but 8 hours avoids wasting the rerun. Rationale: fills the biggest non-MuSiQue 26B method-family holes without touching locked API providers.

2. Probe Groq Llama 70B MuSiQue baseline at N=200, but stop immediately if quota errors appear.

```bash
PYTHONUNBUFFERED=1 uv run python eval/eval_harness.py \
  --mode rag_simple \
  --provider groq-llama70b \
  --questions 200 \
  --dataset musique \
  --tag captain-llama70b-musique-rag-simple-n200 \
  > /tmp/captain_llama70b_musique_rag_simple_n200.log 2>&1
```

Slot: Groq Llama 70B. Dataset cap: `--questions 200` MuSiQue. ETA if quota is open: about 5 minutes from the N=100 observed latency; risk: high, because Groq Llama 70B TPD is likely capped until the 19:00 CDT reset.

3. If launch 2 finishes cleanly and Groq quota still works, immediately run the paired Llama 70B `multi_hyde_diverse` N=200.

```bash
PYTHONUNBUFFERED=1 uv run python eval/eval_harness.py \
  --mode multi_hyde_diverse \
  --provider groq-llama70b \
  --questions 200 \
  --dataset musique \
  --tag captain-llama70b-musique-mhd-n200 \
  > /tmp/captain_llama70b_musique_mhd_n200.log 2>&1
```

Slot: same Groq Llama 70B slot. Dataset cap: `--questions 200` MuSiQue. ETA if quota is open: about 10 minutes from the N=100 observed latency; risk: high TPD. This is the most valuable possible Llama result because the current landed pair is only `rag_simple` 21.00% (N=100) vs `multi_hyde_diverse` 33.00% (N=100).

4. Conditional cluster alternative: if the BarExam rerun cannot be submitted, use the cluster slot for Gemma 4 E4B MuSiQue near-full baseline + `multi_hyde_diverse` at N=800.

```bash
ssh wustl 'cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean && sbatch -p general-gpu -A engr-lab-jacobsn --gpus 1 -c 8 --mem=64G -t 8:00:00 -J e4b-musique-n800 -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out --wrap '"'"'
set -euo pipefail
MODEL=google/gemma-4-E4B-it
PORT=8025
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv
cd "$REPO"
"$GEMMA_VENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization 0.85 --max-model-len 8192 > "$LOG_DIR/vllm_e4b_musique_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break; sleep 5; done
source "$EVAL_VENV/bin/activate"
for mode in rag_simple multi_hyde_diverse; do
  LLM_PROVIDER=cluster-vllm LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" LLM_API_KEY=DUMMY_KEY LLM_MODEL="$MODEL" \
    python eval/eval_harness.py --mode "$mode" --provider cluster-vllm --questions 800 --dataset musique --tag captain-e4b-musique-n800
done
'"'"''
```

Slot: cluster, mutually exclusive with launch 1. Dataset cap: `--questions 800` MuSiQue, which reaches Tier 3 by the matrix rule. ETA: 3-5 hours if E4B throughput is healthy; risk: one-off wrapper, less proven than the BarExam script. Rationale: fills the totally empty E4B MuSiQue column without double-booking `or-gemma4-26b` or `or-qwen3-30b-moe`.

Do not launch new `or-gemma4-26b` or `or-qwen3-30b-moe` jobs during this window; those providers are already occupied by the full-corpus MuSiQue jobs below. Do not spend the 5-hour window on `or-llama70b` unless Groq fails instantly and someone is willing to babysit 429 retries; the route is known rate-limited.

Cluster status check from this sandbox was not verifiable. The attempted command was `ssh wustl 'squeue -u hiqbal'`, and the exact failure was:

```text
Control socket connect(/Users/hamzaiqbal/.ssh/sockets/hiqbal@shell.engr.wustl.edu-22): Operation not permitted
ssh: Could not resolve hostname shell.engr.wustl.edu: -65563
```

## 4. Already-running (don't kill)

- `gemma4_full` `multi_hyde_diverse` pair on MuSiQue full corpus, N=2400, `or-gemma4-26b`, PID 2487, approximately 15% per the human status note. Do not duplicate `rag_simple` or `multi_hyde_diverse` on `or-gemma4-26b`.
- `qwen_full` `multi_hyde_diverse` pair on MuSiQue full corpus, N=2400, `or-qwen3-30b-moe`, PID 2675, approximately 27% per the human status note. Do not duplicate `rag_simple` or `multi_hyde_diverse` on `or-qwen3-30b-moe`.

## 5. Meeting story

We have a paper-grade BarExam story and only a directional MuSiQue story so far. In `logs/experiments.jsonl`, Gemma 4 26B-A4B has full BarExam `rag_snap_hyde` at 75.40% (N=1195) versus `rag_simple` at 71.80% (N=1195), and Gemma 4 E4B has `rag_snap_hyde` at 58.41% (N=1195) versus `rag_simple` at 55.73% (N=1195), so snap+HyDE is the clean cross-size legal-MC signal. On MuSiQue, `multi_hyde_diverse` is directionally better on two target families but only at Tier 1: Llama 70B is 33.00% (N=100) versus 21.00% (N=100) `rag_simple`, and Qwen3 30B MoE is 28.00% (N=100) versus 24.00% (N=100). The next landings should decide whether that multi-hop diversity signal survives at paper scale: Gemma 4 26B-A4B and Qwen3 30B MoE full-corpus MuSiQue pairs are already running, while Llama 70B Tier 2 is the main blocked-but-high-value confirmation.

## 6. Contingency if Llama 70b N=200 stays blocked

Fallback narrative: Llama 70B remains a Tier 1 mechanism check, not a headline full-validation claim. Say plainly that Groq Llama 70B was rate-limited before the meeting and the OpenRouter free route was 429-limited, so the Llama result is queued for the 19:00 CDT reset rather than absent for scientific reasons. The meeting claim should then lean on two safer pillars: full BarExam Gemma cross-size snap+HyDE evidence from the JSONL matrix, and the imminent full-corpus MuSiQue runs for Gemma 4 26B-A4B plus Qwen3 30B MoE. If those full MuSiQue jobs do not land before the meeting, present MuSiQue as an active validation frontier: directional `multi_hyde_diverse` lift at N=100, full/near-full confirmation in flight, no overclaim.

## 7. What to ignore from now on

- Scout, Maverick, and any other non-target model. The four-model frame is Gemma 4 26B-A4B, Gemma 4 E4B, Llama 3.3 70B, and Qwen3 30B MoE.
- New Gemma 3 27B launches. It can stay as a historical anchor if already in docs, but it is outside this sprint's four-model frame.
- N=30, N=5, smoke, and `_FAILED-EMPTY-RETRIEVAL` rows. They can diagnose code, not support the paper.
- Local-Mac BarExam RAG runs unless the populated `legal_passages` collection is verified first; the audit log already caught the empty-collection failure mode.
- More method families. The sprint is not short of ideas; it is short of clean full/near-full cells in the existing matrix.
- Any launch that double-books `or-gemma4-26b` or `or-qwen3-30b-moe` while the full-corpus MuSiQue jobs are running.
