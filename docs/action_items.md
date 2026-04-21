# Action Items — Paper Sprint

Target venues:
- **EMNLP 2026** — due May 20, 2026 (8 pages)
- **ICML AI4Law Workshop** — due May 22, 2026 (8 pages)
- **ACL Rolling Review** — rolling deadlines
- Fallback: NAACL 2027 (August deadline)

---

## Update 2026-04-20/21

- Leakage audit: every canonical HyDE-family leaderboard number in `logs/experiments.jsonl` is currently a pre-leak-fix reference. Historical `rag_hyde` passages leaked `Answer: (X)` in **100%** of samples and historical `rag_snap_hyde` passages in **74%**; `_sanitize_intermediate_text` landed in `02edbb7` on 2026-04-17 after those runs were logged and still had regex bugs.
- Hardening landed on `hpc-setup` through `dfb6a9b`: `e508765`, `951729d`, `bf89b78`, `baef4d8`, `0b4e35d`, `71533fd`, `c85fe70`, `a377867`, `6118161`, `bab7cf5`, `a493491`. Smoke job `50812` removed generation-time HyDE leakage (`top_level_hyde_artifacts=0`) with `rag_hyde` **19/30 (63.3%)** and `rag_snap_hyde` **21/30 (70.0%)**.
- 2026-04-21 status: `50835` mini-eval landed clean — E4B N=200 `rag_simple` 60.5%, `rag_hyde` 59.5%, `rag_snap_hyde` 66.5%, `snap_only_in_final` 64.0%, all 0% leak. **Narrative flip**: snap adds **+7pp** over plain HyDE post-fix; the old "0pp" reading was a leak artifact. `50836` downloads done (E2B + 31B cached). `50822`/`50838` superseded by full-corpus reruns.
- **12-job cross-scale full-N=1195 wave in flight**: E2B (50986 redo after 50867 wallclocked), E4B (50858/50859), 26B-A4B (50868 core + 50990/50991/50992 expansion on parallel a100s-2305 slots), 31B (50865 core + 50993/50994/50995 queued for H100). Covers 10 modes per size: llm_only, golden_passage, rag_simple, rag_hyde, rag_snap_hyde, snap_only_in_final, subagent_rag, subagent_hyde, subagent_hybrid, snap_hyde_report. Plan + live snapshot in `docs/size_comparison_matrix.md`.
- Landed full N=1195 rows so far (all post-fix, 0% leak): E2B `rag_simple` **45.4%**, E4B `rag_simple` **55.7%**, 26B-A4B `rag_simple` **70.8%** / `rag_hyde` **74.2%**, 31B `rag_simple` **79.6%**. Monotonic scaling. 31B N=200 matrix already landed at 79-85% across 4 modes. Do not treat the historical `57.9%` / `58.6%` HyDE numbers below as clean post-fix leaderboard results.

---

## Paper Narrative

**Core claim (revised April 17):** HyDE passage generation is the primary driver of retrieval quality for legal QA — it bridges the genre gap between question-form queries and doctrinal corpus passages. Snap (letting the LLM reason first) helps plain RAG (+5pp) and parametric reasoning (+5pp), but adds zero to HyDE. Showing snap to the final decision-maker always hurts (-2 to -4pp).

**Supporting evidence:**
- Snap ablation across three families on the pre-leak-fix canonical runs: HyDE **0pp** (`rag_hyde` fixed 57.9% = `snap_hyde` 57.9% at N=1195), plain RAG **+5pp**, parametric reasoning **+5pp**; clean reruns pending
- The previous HyDE snap lift (+3pp) was a bug artifact from a broken Gemma prompt
- Showing snap to the final agent always hurts: `snap_hyde_report_snap` 64% < `snap_hyde_report` 66%, `subagent_rag_snap` 63% < `subagent_rag` 66%, `subagent_rag_full` 62% < `subagent_rag` 66%
- Three identified failure modes: noise, anchoring, genre mismatch
- Cross-dataset: snap lift is BarExam-specific (flat on HousingQA, negative on CaseHOLD)

---

## Priority 1: Critical Experiments (MUST DO for paper)

### P1.1: Snap vs No-Snap Ablation (the paper's core comparison)
- [x] **Pure HyDE (no snap)** — fixed full `rag_hyde` completed at **57.9%** on Gemma 4 E4B N=1195 (**pre-leak-fix canonical run; clean rerun pending**)
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

## Latest Job Status

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
