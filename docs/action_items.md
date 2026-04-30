# Action Items — Paper Sprint

## Update 2026-04-27 evening — meeting follow-ups

Source: [`docs/meeting_notes_042726.md`](meeting_notes_042726.md). Key new asks:

1. **Golden-passage paradox — DONE** (audit landed). See [`docs/golden_paradox_audit_2026-04-27.md`](golden_paradox_audit_2026-04-27.md). Headline: gold-passage injection is roughly symmetric (96 hurt vs 83 helped, net -1.09pp); the dominant failure is anchoring, not insufficient evidence; gold-passage length distribution is identical across paradox / win / match buckets. **Action**: rename `golden_passage` from "oracle ceiling" to "single gold-passage control" in paper tables.
2. **Top-1 vs top-5 retrieval-depth ablation** — `--retrieval-k` CLI flag landed (commit `b286279`). Suggested paired N=200 launch list (do not launch without ops sync, mind Groq RPD = 1000 / TPD = 100K):

   ```bash
   # BarExam, Llama 70b, top-1 vs top-5 paired (rag_simple + rag_snap_hyde)
   uv run python eval/eval_harness.py --mode rag_simple    --provider groq-llama70b --questions 200 --dataset barexam --retrieval-k 1
   uv run python eval/eval_harness.py --mode rag_simple    --provider groq-llama70b --questions 200 --dataset barexam --retrieval-k 5
   uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset barexam --retrieval-k 1
   uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset barexam --retrieval-k 5

   # MuSiQue, Llama 70b, top-1 vs top-5 paired (rag_simple + multi_hyde_diverse + rag_multi_query)
   uv run python eval/eval_harness.py --mode rag_simple         --provider groq-llama70b --questions 200 --dataset musique --retrieval-k 1
   uv run python eval/eval_harness.py --mode rag_simple         --provider groq-llama70b --questions 200 --dataset musique --retrieval-k 5
   uv run python eval/eval_harness.py --mode multi_hyde_diverse --provider groq-llama70b --questions 200 --dataset musique --retrieval-k 1
   uv run python eval/eval_harness.py --mode multi_hyde_diverse --provider groq-llama70b --questions 200 --dataset musique --retrieval-k 5
   uv run python eval/eval_harness.py --mode rag_multi_query    --provider groq-llama70b --questions 200 --dataset musique --retrieval-k 1
   uv run python eval/eval_harness.py --mode rag_multi_query    --provider groq-llama70b --questions 200 --dataset musique --retrieval-k 5

   # Cluster vLLM Gemma 4 26B-A4B BarExam top-1 paired (rag_simple + rag_snap_hyde) — re-uses existing top-5 Tier 3 baselines
   # (run on cluster via SLURM script under scripts/hpc/; do NOT launch via OR for Gemma due to runaway-loop serving issue)
   ```

   Pair the `--retrieval-k 1` runs against the existing N=200 / Tier 3 top-5 baselines using `scripts/compute_mcnemar.py`. Report: EM, paired delta, McNemar p, gold_retrieved rate, mean input tokens, mean retrieved-passage chars.

3. **`rag_snap_hyde_2call`** — landed. 2-call efficiency variant: single LLM call producing snap reasoning + HyDE passage in one response (parsed apart with `## Passage` marker), then retrieve + final synth. Goal: preserve the +3pp BarExam lift with 33% fewer LLM calls. Same final-context contract as `rag_snap_hyde` (snap letter NOT shown to final agent). Failed-parse cases marked with `routed_to: snap_hyde_2call_parse_failed_fallback_to_question`. Initial paired N=200 runs are now summarized in `docs/snap_hyde_2call_2026-04-28.md`: Llama 70B MuSiQue is significant at +9.5pp, while Gemma 4 26B BarExam N=200 is directional only at +3.0pp, p=0.377. Reference rerun pattern:

   ```bash
   uv run python eval/eval_harness.py --mode rag_snap_hyde       --provider groq-llama70b --questions 200 --dataset barexam
   uv run python eval/eval_harness.py --mode rag_snap_hyde_2call --provider groq-llama70b --questions 200 --dataset barexam
   ```

   Compare via paired McNemar; check `snap_hyde_2call_parse_ok` rate before citing efficiency claim.
4. **Re-run llm_only / golden_passage on the same 4 models** if any methodology has changed since the cited Tier 3 logs (currently 2026-04-26 cluster-vllm runs are the source of truth — no rerun needed unless a fix lands).
5. **Lift-source decomposition** — for each candidate lift method, log llm_calls + retrieved_ids + evidence diversity + snap-vs-final agreement + token/latency. Already partially in place.
6. **Dataset × model × method matrix** — keep tight: BarExam + MuSiQue (current), add FRAMES (https://huggingface.co/datasets/google/frames-benchmark) and SCALR / MLEB-SCALR (https://huggingface.co/datasets/isaacus/mleb-scalr) when bandwidth allows.
7. **HyRe / multi-hyde-diverse literature sweep** — does the diverse-HyDE pattern exist under another name (Query2Doc, RAG-Fusion, Chain-of-Note, HyRe)? Frame the contribution as "task-specific retrieval shaping" rather than "new universal trick".
8. **Authorship / time-budget coordination** — async with team, separate from code work.

Live status of cluster work (do not rerun without ops sync):
- `qwen_full` MuSiQue mhd-pair × Qwen3 30B MoE × N=2400 (task #63) — operator-side.
- `SLURM 55107` BarExam mhd+iter_hyde × Gemma 4 26B-A4B × N=200 (task #70) — operator-side.

## Update 2026-04-27 ~12:30 CDT

Change reason: housekeeping sweep after Tier 2 MuSiQue sign-off. Completed: Llama 70b N=200 method matrix, exact McNemar infrastructure (`scripts/compute_mcnemar.py`), friend/foe analysis script (`scripts/analyze_friend_foe_bias.py`), and N<200 citation gating. Current follow-ups: audit the running Tier 3/full MuSiQue jobs when they land; do not cite Gemma 3 27B MHD as confirmed (+2.5pp, p=0.5901 NULL); keep `subagent_rag` on MuSiQue as a significant negative (-12.0pp, p=0.0007), not a candidate improvement.

Target venues:
- **EMNLP 2026** — due May 20, 2026 (8 pages)
- **ICML AI4Law Workshop** — due May 22, 2026 (8 pages)
- **ACL Rolling Review** — rolling deadlines
- Fallback: NAACL 2027 (August deadline)

---

## Update 2026-04-27

Current meeting-facing numbers are superseded by `docs/audit_log.md`, `CLAUDE.md`, `RESEARCH.md`, and `docs/experiment_overview.md`. Key changes since the 2026-04-20/21 snapshot below: Phase 12 post-prompt BarExam full-N winner is Gemma 4 26B `rag_snap_hyde` **81.17%**; E4B `rag_snap_hyde` is **62.18%**; Phase 13.5 `multi_hyde_diverse` is the first MuSiQue cross-family lift at N=100 (Llama +12pp p=0.023, Gemma 3 27B +8pp p=0.134); Phase 14 `iter_hyde` hurts Gemma 3 27B (-20pp). The historical job statuses and pre-fix HyDE narrative below are retained for audit continuity, not current citation.

## Update 2026-04-20/21

- Leakage audit: every canonical HyDE-family leaderboard number in `logs/experiments.jsonl` is currently a pre-leak-fix reference. Historical `rag_hyde` passages leaked `Answer: (X)` in **100%** of samples and historical `rag_snap_hyde` passages in **74%**; `_sanitize_intermediate_text` landed in `02edbb7` on 2026-04-17 after those runs were logged and still had regex bugs.
- Hardening landed on `hpc-setup` through `dfb6a9b`: `e508765`, `951729d`, `bf89b78`, `baef4d8`, `0b4e35d`, `71533fd`, `c85fe70`, `a377867`, `6118161`, `bab7cf5`, `a493491`. Smoke job `50812` removed generation-time HyDE leakage (`top_level_hyde_artifacts=0`) with `rag_hyde` **19/30 (63.3%)** and `rag_snap_hyde` **21/30 (70.0%)**.
- 2026-04-21 status: `50835` mini-eval landed clean — E4B N=200 `rag_simple` 60.5%, `rag_hyde` 59.5%, `rag_snap_hyde` 66.5%, `snap_only_in_final` 64.0%, all 0% leak. **Narrative flip**: snap adds **+7pp** over plain HyDE post-fix; the old "0pp" reading was a leak artifact. `50836` downloads done (E2B + 31B cached). `50822`/`50838` superseded by full-corpus reruns.
- **Historical 2026-04-21 status:** 12-job cross-scale full-N=1195 wave was in flight. This snapshot is superseded by the 2026-04-27 audit; current BarExam full-N claims should cite `docs/audit_log.md`.
- Historical landed full N=1195 rows as of 2026-04-21: E2B `rag_simple` **45.4%**, E4B `rag_simple` **55.7%**, 26B-A4B `rag_simple` **70.8%** / `rag_hyde` **74.2%**, 31B `rag_simple` **79.6%**. These E4B/26B values are superseded for current citation by the 2026-04-27 audit (E4B rag_simple **58.49%**, 26B rag_simple **78.08%**, 26B rag_hyde **78.91%**). Do not treat the historical `57.9%` / `58.6%` HyDE numbers below as clean current leaderboard results.

---

## Paper Narrative

**Historical core claim (revised April 17; superseded for current citations by the 2026-04-27 audit):** HyDE passage generation is the primary driver of retrieval quality for legal QA — it bridges the genre gap between question-form queries and doctrinal corpus passages. Snap (letting the LLM reason first) helps plain RAG (+5pp) and parametric reasoning (+5pp), but adds zero to HyDE in the pre-prompt-fix comparison. Current post-prompt full-N claims should cite `docs/audit_log.md`.

**Supporting evidence:**
- Snap ablation across three families on the pre-leak-fix canonical runs: HyDE **0pp** (`rag_hyde` fixed 57.9% = `snap_hyde` 57.9% at N=1195), plain RAG **+5pp**, parametric reasoning **+5pp**; superseded for current citation by the 2026-04-27 audit
- The previous HyDE snap lift (+3pp) was a bug artifact from a broken Gemma prompt
- Showing snap to the final agent always hurts: `snap_hyde_report_snap` 64% < `snap_hyde_report` 66%, `subagent_rag_snap` 63% < `subagent_rag` 66%, `subagent_rag_full` 62% < `subagent_rag` 66%
- Three identified failure modes: noise, anchoring, genre mismatch
- Cross-dataset: snap lift is BarExam-specific (flat on HousingQA, negative on CaseHOLD)

---

## Priority 1: Critical Experiments (MUST DO for paper)

### P1.1: Snap vs No-Snap Ablation (the paper's core comparison)
- [x] **Pure HyDE (no snap)** — fixed full `rag_hyde` completed at **57.9%** on Gemma 4 E4B N=1195 (**pre-leak-fix canonical run; superseded by 2026-04-27 audit**)
- [x] **Compare:** paired full `snap_hyde` (**57.9%**) vs fixed full `rag_hyde` (**57.9%**) = **0pp** snap contribution for HyDE on the pre-leak-fix canonical pair
- [x] **snap_rag (62.0%) vs rag_simple (57.0%)** = already done, +5pp ✓
- [x] **`vectorless_direct` vs `vectorless_nosnap`** — completed: **64.5% vs 59.5%**, another **+5pp** snap lift
- [x] **Core table complete:** snap adds **0pp to HyDE**, **+5pp to plain RAG**, and **+5pp to parametric reasoning**
- Data: `logs/experiments.jsonl`, detail logs in `logs/eval_*_detail.jsonl`

### P1.2: Cross-Dataset Validation
Completed on 2026-04-14. Main take-away: the BarExam snap / parametric lift did not transfer cleanly to HousingQA or CaseHOLD.
→ see `docs/archive/action_items_completed.md` for the completed checklist and recorded numbers.

### P1.3: Full-Scale N=1195 Validation
Historical pre-2026-04-17 full-scale validation is complete and now serves as audit context for the post-fix reruns above.
→ see `docs/archive/action_items_completed.md` for the completed checklist and archived results.

---

## Priority 2: Important Experiments (SHOULD DO)

### P2.1: Fix Historical Vectorless / Parametric Reasoning and Test Snap Contribution
Completed. The historical `vectorless_*` snap-ablation block is archived to keep the active queue focused.
→ see `docs/archive/action_items_completed.md` for the finished checklist and recorded deltas.

### P2.2: Subagent Variants
- [x] **subagent_hyde** — completed at **62.5%** (`125/200`), below `subagent_rag` **66.0%**
- [ ] **subagent_vectorless** — subagent generates knowledge instead of retrieving (no corpus)
- [ ] **subagent_panel** — multiple subagents with different roles (textbook/barprep/casebook)
- [x] Results from subagent_hybrid and subagent_rag_evidence: **DONE** — `subagent_hybrid` 63.5%, `subagent_rag_evidence` 61.0%
- [x] Update 2026-04-15: `subagent_hyde` used **5.2 avg** calls and still trailed `subagent_rag`
- Code: `eval/eval_harness.py`, subagent runners in the gap-family section

### P2.3: Combo-Mode Anchoring Controls
Completed on 2026-04-17. Visible-snap combo controls are archived; the active docs now keep only the top-line finding that showing snap to the final agent hurts.
→ see `docs/archive/action_items_completed.md` for the completed checklist and recorded numbers.

### P2.4: Corpus Structure / Metadata Approaches
- [ ] **Proximity RAG** — use RAG to find a passage, then expand context by pulling the full case/document it came from (using `case_id` + `relative_paragraph_id`), plus neighboring passages. Subagent reads the expanded context and summarizes. Addresses the 95-word avg passage length problem — answers often span multiple paragraphs from the same source.
- [ ] **Topic-filtered retrieval** — classify passages by bar exam subject (7 topics), retrieve only from matching topic
- [ ] **PageIndex-style ToC** — build a table of contents from the corpus, let LLM navigate. NOTE: PageIndex is designed for single documents, our corpus is 686K flat passages. May need adaptation.
- [ ] How good is existing metadata? Source: 98.9% caselaw, 0.3% mbe, 0.8% wex. Gold passages are ALL from the 2,318 mbe passages (0.3% of corpus). Filtering to mbe-only would be trivially better but defeats the purpose.
- [x] Update 2026-04-15: `snap_entity_informed` completed at **59.5%** (`119/200`) — below `entity_search` **60.0%**, suggesting snap terms add noise to entity matching
- [x] Update 2026-04-15: full `entity_search` completed at **53.2%** (`636/1195`) and finished below vector `rag_simple` **54.2%**
- Data: `datasets/barexam_qa/barexam_qa_train.csv` (columns: idx, source, case_id, opinion_id, text)

---

## Priority 3: Lower Priority (NICE TO HAVE)

### P3.1: PageIndex Implementation
- [ ] Study PageIndex ToC building: https://pageindex.ai/blog/pageindex-intro
- [ ] Build a ToC from the barexam corpus (cluster passages by topic, generate summaries)
- [ ] Test LLM-navigated retrieval via ToC vs vector search
- [ ] Limitation: PageIndex max 520K, our corpus is 686K — may need to subset
- Note: research agent found PageIndex is designed for single-doc, not multi-doc corpus. Needs adaptation.

### P3.2: Strong vs Weak Model Experiments
- [ ] Test same modes on larger model (Qwen3-32B or Gemma-27B via API)
- [ ] Does snap lift scale with model capability?
- [ ] Meeting note: "interesting but not necessarily paper direction-worthy"

### P3.3: Literature Review
- [ ] **Is subagent RAG new?** Research existing work on multi-agent retrieval, subagent summarization
- [ ] **Is snap-then-retrieve new?** Research HyDE paper, Chain-of-Note, Self-RAG, CRAG
- [ ] **Can we combine subagent_rag and snap_hyde?** Think about this architecturally
- [ ] Related: RAG for retrieving metadata, embeddings to inform topic structure

---

## What's Done (reference)
Completed pre-2026-04-17 sprint checklists and the longer finished-results table were moved out of the main action list.
→ see `docs/archive/action_items_completed.md` for the full archived reference block.

---

## Latest Job Status (2026-04-20/21 historical snapshot; superseded)

| Job | What | Status |
|---|---|---|
| 50835 | clean mini-eval E4B N=200 × 4 modes | ✅ DONE — rag_simple 60.5%, rag_hyde 59.5%, rag_snap_hyde 66.5%, snap_only_in_final 64.0%, 0% leak |
| 50836 | Gemma checkpoint downloads | ✅ DONE — E2B + 31B cached on engrfs |
| 50812 | Mini smoke N=30 (rag_hyde, rag_snap_hyde) | ✅ DONE — 0% leak confirmed at generation |
| 50857 | 31B smoke N=5 on H100 | ✅ DONE — 5/5, proves 31B works unquantized |
| 50864 | 31B N=200 × 4 modes | ✅ DONE — rag_simple 79%, rag_hyde 83%, rag_snap_hyde 85%, snap_only_in_final 84% |
| 50822/50838 | 26B-A4B smoke N=5 | SUPERSEDED by full-N=1195 50868 |
| 50867 | E2B full N=1195 × 4 modes | ⚠ WALLCLOCKED at q 1000/1195 in rag_hyde; rag_simple landed 45.4%; resubmitted as 50986 |
| 50986 | E2B redo (3 remaining modes) | Running on a40-2205 |
| 50858 | E4B full P1a (rag_simple, rag_hyde, snap_only_in_final) | Running on a40-2206 — rag_simple 55.7% landed |
| 50859 | E4B full P1b (rag_snap_hyde) | Running on a40-2206 — q ~1130/1195 |
| 50865 | 31B full N=1195 × 4 modes | Running on h100-2405 — rag_simple 79.6% landed |
| 50868 | 26B-A4B full N=1195 × 4 modes | Running on a100s-2306 — rag_simple 70.8%, rag_hyde 74.2% landed |
| 50990 | 26B-A4B llm_only + golden_passage | Running on a100s-2305 (parallel) |
| 50991 | 26B-A4B subagent_rag + subagent_hyde | Running on a100s-2305 (parallel) |
| 50992 | 26B-A4B subagent_hybrid + snap_hyde_report | Running on a100s-2305 (parallel) |
| 50993 | 31B llm_only + golden_passage | PENDING on h100-2405 |
| 50994 | 31B subagent_rag + subagent_hyde | PENDING on h100-2405 |
| 50995 | 31B subagent_hybrid + snap_hyde_report | PENDING on h100-2405 |
| 44371 | case summaries build | Completed — 22K summaries built |
| 44394 | snap ablations | Completed — `rag_hyde` 62.5%, `vectorless_nosnap` 59.5% |
| 44395 | cross-dataset jobs | Completed — HousingQA and CaseHOLD follow-ups logged |
| 44520 | entity graph rebuild | Status unverified — last noted at 74% on 2026-04-14 |
| 45350 | rag_hyde + ce_threshold full | ✅ Completed — ce_threshold 55.9%; the interim rag_hyde rerun was later superseded |
| 45735 | gap_rag_nosnap + subagent_rag (1-gap) full | ✅ Completed — 55.9%, 57.2% |
| 48393 | combo modes N=200 | ✅ Completed — snap_hyde_report 66.0%, snap_hyde_report_snap 64.0%, subagent_rag_snap 63.0%, subagent_rag_full 62.0% |
| 48555 | rag_hyde fixed full N=1195 | ✅ Completed — **57.9%** (matches snap_hyde; snap lift = 0pp) |
| 43471 | vectorless_direct + hybrid full N=1195 | Cancelled — fake vectorless / not real corpus search |

---

## Key Files

| What | Where |
|---|---|
| All results | `logs/experiments.jsonl` |
| Meeting prep | `docs/meeting_2026_04_17.md` |
| Experiment overview | `docs/experiment_overview.md` |
| This action list | `docs/action_items.md` |
| Experiment narratives | `EXPERIMENTS.md` |
| Research state | `RESEARCH.md` |
| Detail logs | `logs/eval_*_detail.jsonl` |
| Formatted readable logs | `/tmp/*_readable.md` (local only) |
