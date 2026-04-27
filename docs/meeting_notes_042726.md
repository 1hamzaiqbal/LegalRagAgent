# Meeting notes / HPC handoff — 2026-04-27

Temporary handoff note from Hamza's 2026-04-27 meeting notes plus a quick repo-state pass on branch `hpc-setup`.

- Branch pulled locally: `hpc-setup`
- Local worktree used for this pass: `/home/techguy227/grad/LegalRagAgent-hpc-setup`
- Pulled HEAD at handoff creation: `880c851` (`docs: codex archive pass — 10 stale docs + duplicate figures/ folder moved to archive_2026-04-27/`)
- No new evals were launched in this pass.
- Main source docs to read first: `docs/signoff_log.md`, `docs/narrative_2026_04_27.md`, `docs/mcnemar_2026-04-27.md`, `docs/compiled_results.md`, `docs/verification_2026-04-27.md`.

## 0. Current repo-state snapshot

What `hpc-setup` currently says:

1. **BarExam / legal MC headline**: `rag_snap_hyde` is the current signed Tier 3 winner on Gemma 4 cluster runs.
   - Gemma 4 26B-A4B: `rag_simple` 78.08% → `rag_snap_hyde` 81.17% (`+3.09pp`, N=1195)
   - Gemma 4 E4B: `rag_simple` 58.49% → `rag_snap_hyde` 62.18% (`+3.69pp`, N=1195)
   - Caveat: `golden_passage` is **not** a near-100% oracle in the current BarExam logs; see sanity-check item below.

2. **MuSiQue / multi-hop headline**: Llama 70B `multi_hyde_diverse` is the current signed Tier 2 result.
   - `rag_simple`: 55/200 = 27.5%
   - `multi_hyde_diverse`: 71/200 = 35.5%, `+8.0pp`, McNemar `p=0.0195`
   - `iterative_planning_table`: 72/200 = 36.0%, `+8.5pp`, McNemar `p=0.0533` (trending, not cleanly significant)
   - `subagent_rag`: 31/200 = 15.5%, `-12pp`, McNemar `p=0.0007`; this is a real negative for the current over-abstaining gap-routing implementation.

3. **Cross-family status is not settled**.
   - Gemma 3 27B `multi_hyde_diverse` at N=200 is NULL: 28.5% → 31.0%, `+2.5pp`, `p=0.5901`.
   - Qwen3 30B MoE has N=100 directional evidence only in local logs; the full run is/was in flight per operator snapshots.
   - OR-served Gemma is problematic for iterative/multi-call methods due to runaway-loop generations. Cluster vLLM should be preferred for Gemma.

4. **Source-pending / don't overcite**.
   - `qwen_full` full MuSiQue mhd-pair snapshot: source log not present locally; treat as operator snapshot until synced.
   - SLURM `55107` BarExam `multi_hyde_diverse` / `iter_hyde` × Gemma 4 26B N=200: source detail log not present locally; do not cite as landed.
   - `gemma4_full` OR API MuSiQue run was killed at q431/2400; use only as serving-caveated partial context.

5. **Old local qwen3-8b Snap-HyDE note**.
   - A separate older local worktree had completed qwen3-8b BarExam n100 Snap-HyDE probes: default GTE `rag_snap_hyde` 58%, legal-BGE `rag_snap_hyde` 64%.
   - These are not the current `hpc-setup` paper headline and should stay historical/directional unless deliberately merged into the current branch narrative.

## 1. Immediate HPC operator checklist

For the agent/operator with cluster access:

```bash
ssh wustl
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
git fetch origin
git checkout hpc-setup
git pull --ff-only origin hpc-setup

# Before citing or launching new work:
python tests/test_formatter.py
python tests/test_sanitizer.py

# Job status / log triage:
squeue -u hiqbal
# logs usually under:
# /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/
```

Cluster reminders from `docs/hpc_setup_log.md`:
- Use `python` directly for vLLM jobs; avoid `uv run` after vLLM deps are pinned.
- Always set `XDG_CACHE_HOME` to `/engrfs/tmp/...` for vLLM torch compile cache.
- Avoid known-bad nodes: `a100-2207`, `a100s-2307`; avoid `r28-1801` for insufficient VRAM.
- vLLM startup can take 5-20 min depending on model/cache; health checks need long timeouts.

## 2. Golden-passage sanity check — highest-priority audit

Meeting note: **"Sanity check golden passage vs llm_only vs rag_snap_hyde. Where does lift come from?"**

Why this matters:
- A true strong-model + true-gold-passage condition should be close to ceiling.
- Current audited BarExam Gemma 4 26B numbers are surprising: `golden_passage` 78.66% is below `llm_only` 79.75%, while `rag_snap_hyde` is 81.17%.
- If gold context is correct and sufficient, this should not happen often. So either the "gold" passage is not sufficient, the prompt is not using it correctly, the benchmark labels/options/fact pattern require more than the cited passage, or extraction/formatting is still hiding an issue.

Specific requested artifact to inspect if available on Hamza's Mac / synced machine:

```text
/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_llm_only_deepseek_20260426_1917_detail.jsonl
```

Concrete checks:
1. Build a paired table for the same question IDs:
   - `llm_only`
   - `golden_passage`
   - `rag_simple`
   - `rag_snap_hyde`
   - optionally `snap_only_in_final`
2. Count paired transitions:
   - llm wrong → golden correct
   - llm correct → golden wrong
   - llm wrong → rag_snap_hyde correct
   - rag_snap_hyde correct with/without exact `gold_retrieved`
3. Manually audit at least 20 golden failures:
   - Is `gold_passage` actually relevant and sufficient?
   - Does the question include the full fact pattern (`prompt` + `question`) in the final prompt?
   - Does the model ignore or misread the passage?
   - Does extraction parse the correct final option?
   - Is the gold label itself questionable?
4. If strong-model `golden_passage` is not near-ceiling, do **not** call it an oracle ceiling. Rename in paper language to "single gold-passage control" or "provided gold snippet control" unless the audit proves sufficiency.

## 3. Lift-source decomposition

Meeting note: **"Where does lift come from? More reasoning steps? Diverse set of evidences?"**

For each candidate lift, log and analyze:
- Number of LLM calls (`llm_calls`)
- Retrieved IDs and `gold_retrieved`
- Number/diversity of evidence passages
- Whether the final answer matches the snap answer (`snap_letter`) when available
- Whether the final evidence changed a previously wrong snap
- Token/cost/latency per question

High-signal paired comparisons:

| Dataset | Model | Baseline | Treatment | Question |
|---|---|---|---|---|
| BarExam | Gemma 4 E4B / 26B | `llm_only` | `golden_passage` | Does true evidence help, or does it distract? |
| BarExam | Gemma 4 E4B / 26B | `rag_simple` | `rag_snap_hyde` | Is snap+HyDE doing more than retrieval? |
| BarExam | Gemma 4 E4B / 26B | `snap_only_in_final` | `rag_snap_hyde` | How much marginal value does retrieval add over snap reasoning? |
| MuSiQue | Llama 70B | `rag_simple` | `rag_multi_query` | Diversity-only component |
| MuSiQue | Llama 70B | `rag_multi_query` | `multi_hyde_diverse` | HyDE-style answer-passage component |
| MuSiQue | Llama 70B | `rag_simple` | `iterative_planning_table` | Structured reasoning/planning component |

Current repo interpretation: on MuSiQue Llama 70B, most `multi_hyde_diverse` lift appears to come from HyDE-style answer-bearing passages (~+6.5pp of +8pp), not diversity-only (+1.5pp NS). On BarExam, snap+HyDE wins, but snap dominance must be framed as architecture/mechanism, not hidden leakage.

## 4. Top-1 vs top-5 retrieval-depth ablation

Meeting note: **"Check lift between passing in top-1 retrieved vs top-5 retrieved, see if more passages boost or hurt performance."**

Current code status:
- Most RAG modes call `_retrieve_and_format(..., k=5, ...)` directly in `eval/eval_harness.py`.
- `rag_utils.retrieve_documents(..., k=5)` and `retrieve_documents_multi_query(..., k=5)` fetch `k*3` first-stage candidates, then cross-encoder rerank to top `k`.
- There is not yet a clean CLI-level `--retrieval-k` switch for all modes.

Suggested implementation:
1. Add `retrieval_k: int = 5` to `EvalConfig` and CLI arg `--retrieval-k`.
2. Replace hardcoded `k=5` in primary modes with `k=config.retrieval_k`.
3. Preserve default behavior as `k=5`.
4. Run paired top-1 vs top-5 ablations first at N=200, not full corpus:
   - BarExam: `rag_simple`, `rag_snap_hyde`, maybe `golden_passage` is not retrieval-k relevant.
   - MuSiQue: `rag_simple`, `multi_hyde_diverse`, `rag_multi_query`.
5. Report both EM and evidence properties: exact gold retrieved, supporting IDs covered, token cost, answer changes.

## 5. Multi-HyDE / diverse-HyDE literature and naming

Meeting notes:
- **"Does multi hyde diverse exist in literature? Compare to snap hyde and think about narrative if so."**
- **"HyRe → vs HyDE: Hypothetical Reasoning Embeddings."**

Literature sweep targets:
- HyDE / hypothetical document embeddings
- Query2Doc / pseudo-document expansion
- RAG-Fusion and multi-query retrieval
- Chain-of-Note / self-reflective RAG variants
- Multi-hop RAG with decomposed retrieval
- HyRe / hypothetical reasoning embeddings if it exists as a named method

Narrative to test:
- `rag_snap_hyde`: snap first, then one targeted HyDE passage. Works on BarExam legal MC, fails/doesn't help on MuSiQue multi-hop.
- `multi_hyde_diverse`: no single snap commitment; generate multiple plausible answer-bearing passages and pool retrieval. Helps Llama 70B on MuSiQue.
- Possible contribution frame: **task-specific retrieval shaping** rather than "new universal RAG trick."

## 6. One-call optimization for Snap-HyDE

Meeting note: **"Try rag_snap_hyde (3 API calls rn) → snap + hyde into 1 call, then retrieve, then final synthesize call (only 2 calls). snap_hyde_rag_answer try snap+hyde_rag_answer."**

Current `rag_snap_hyde` call structure in `eval/eval_harness.py`:
1. `snap_hyde/snap`
2. `snap_hyde/generate`
3. `snap_hyde/answer`

Proposed ablation/mode names:
- `rag_snap_hyde_2call`
- `snap_hyde_rag_answer`
- `snap_hyde_compact`

Success criteria:
- Preserve most of the BarExam `rag_snap_hyde` lift with one fewer LLM call.
- Do not expose the raw snap answer letter to the final answer unless it is an intentional ablation.
- Record cost and latency reduction.

## 7. Dataset × model × method matrix

Meeting note: **"Dataset x model x method."**

Keep the matrix focused; avoid a giant uncontrolled sweep.

### Datasets
- BarExam — current legal MC Tier 3 anchor (N=1195).
- MuSiQue — current multi-hop open QA anchor (N=2400 full; N=200 citeable Tier 2).
- FRAMES: <https://huggingface.co/datasets/google/frames-benchmark>
- SCALR / MLEB-SCALR: <https://huggingface.co/datasets/isaacus/mleb-scalr>
- Other legal retrieval/multi-hop candidates, especially Supreme Court / legal-document MC retrieval datasets.

### Models
Meeting note model list:
- Gemma 4 E4B (~8B effective / small cluster workhorse)
- Gemma 4 26B-A4B / "27B" family (branch docs use `google/gemma-4-26B-A4B-it`)
- Llama 3.3 70B
- DeepSeek as upper bound
- Qwen3 if rule-following is consistent; ignore/drop if truncation or `<think>` parsing remains unstable

### Methods
Minimum high-signal set:
- `llm_only`
- `golden_passage` / sufficient-evidence control
- `rag_simple`
- `rag_snap_hyde`
- `rag_multi_query`
- `multi_hyde_diverse`
- `iterative_planning_table` where relevant

## 8. Paper/deadline/narrative tasks

Meeting notes:
- **ICML AI4Law deadline: May 22** → about 3 weeks from the meeting.
- **EMNLP/ENMLP note says May 25**, while `docs/action_items.md` currently says EMNLP 2026 due May 20. Verify exact deadline before planning backwards.
- Workshop vs conference:
  - Workshop: can be a focused empirical/narrative contribution if tight and honest.
  - Conference: needs a stronger main idea + contribution, not just composing existing methods.

Narrative warning:
- Current state is not "we invented a giant agent and it worked."
- Stronger arc: **method lift is domain-specific; simple retrieval shaping can beat more agentic structures, but only when matched to the task bottleneck.**
- Need a clean paper flow:
  1. Problem: retrieval can help or hurt depending on task/model; agentic complexity is not automatically useful.
  2. Harness: fixed datasets, fixed modes, paired tests, audit gate.
  3. Legal MC finding: BarExam likes snap+HyDE across Gemma 4 sizes.
  4. Multi-hop finding: MuSiQue Llama likes diverse answer-bearing HyDE; serial gap/subagent over-abstains.
  5. Mechanism: diversity-only vs HyDE-answer passages vs planning.
  6. Limits: Gemma 3 null, Qwen pending, golden-passage weirdness, OR-Gemma serving caveat.

## 9. Authorship / team coordination

Meeting note: **"On authorship: figure out time budgets and author order."**

Suggested next conversation with teammates:
- Who is owning cluster ops / run sync?
- Who is owning paper narrative + related work?
- Who is owning audit/statistics tables?
- Who is owning figures and final writing?
- What time budget can each person commit before May 22 / exact EMNLP deadline?
- Decide authorship/order after ownership and concrete contributions are explicit.

## 10. Concrete next actions for the HPC-access agent

1. Pull this branch and read this note plus `docs/verification_2026-04-27.md`.
2. Sync any currently running/pending cluster artifacts back into `logs/`:
   - `qwen_full` MuSiQue mhd-pair full run
   - SLURM `55107` BarExam `multi_hyde_diverse` / `iter_hyde`
   - any cluster-vLLM Gemma reruns that replaced OR-served Gemma failures
3. Commit source detail logs or explicit status notes with job ID, node, branch, commit, and exact command.
4. Run the golden-passage audit before launching a large new sweep.
5. Implement retrieval-k ablation (`--retrieval-k`) before top-1 vs top-5 tests.
6. Implement a 2-call Snap-HyDE ablation if the BarExam story still needs efficiency/cost support.
7. Run only small paired N=200 tests first; promote to full corpus only when the mechanism question is clear.
