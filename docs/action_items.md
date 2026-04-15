# Action Items — Paper Sprint

Target venues:
- **EMNLP 2026** — due May 20, 2026 (8 pages)
- **ICML AI4Law Workshop** — due May 22, 2026 (8 pages)
- **ACL Rolling Review** — rolling deadlines
- Fallback: NAACL 2027 (August deadline)

---

## Paper Narrative

**Core claim:** Letting the LLM reason first ("snap") before retrieval is the single biggest contributor to legal QA accuracy. This lift generalizes across retrieval methods (RAG, HyDE, historical `vectorless_*` / parametric reasoning) and architectures (simple, gap-informed, subagent).

**Supporting evidence:**
- Snap ablation is now complete across the paper's three main families: HyDE +3pp (`snap_hyde` 65.5% vs `rag_hyde` 62.5%), plain RAG +5pp (`snap_rag` 62.0% vs `rag_simple` 57.0%), and parametric reasoning +5pp (`vectorless_direct` 64.5% vs `vectorless_nosnap` 59.5%)
- HyDE leverages snap reasoning for passage-form retrieval (+3.5pp)
- "Vectorless" baselines (really multi-turn parametric reasoning, not corpus search) match vector retrieval on BarExam
- Subagent architecture (snap → gap analysis → subagent reports) achieves new best (66.0%)
- Three identified failure modes: noise, anchoring, genre mismatch

**Open question for paper:** Does the snap lift generalize to other datasets/corpora? Same universal lift?
Latest answer: not universally. The April 14 follow-up is flat on HousingQA and negative on CaseHOLD for the new `vectorless_*` / snap-style controls.

---

## Priority 1: Critical Experiments (MUST DO for paper)

### P1.1: Snap vs No-Snap Ablation (the paper's core comparison)
- [x] **Pure HyDE (no snap)** — `rag_hyde` on Gemma 4 E4B N=200 completed at **62.5%**
- [x] **Compare:** `snap_hyde` (65.5%) vs pure HyDE (62.5%) = **+3pp** snap contribution for HyDE
- [x] **snap_rag (62.0%) vs rag_simple (57.0%)** = already done, +5pp ✓
- [x] **`vectorless_direct` vs `vectorless_nosnap`** — completed: **64.5% vs 59.5%**, another **+5pp** snap lift
- [x] **Core table complete:** snap adds **+3pp to HyDE**, **+5pp to plain RAG**, and **+5pp to parametric reasoning**
- Data: `logs/experiments.jsonl`, detail logs in `logs/eval_*_detail.jsonl`

### P1.2: Cross-Dataset Validation
- [ ] Copy housing QA data to cluster: `scp datasets/housing_qa/ wustl:...`
- [ ] Run on **HousingQA**: llm_only, rag_simple, snap_hyde, vectorless_direct, subagent_rag (N=200 each)
- [ ] Run on **CaseHOLD**: same modes (already have corpus on cluster?)
- [ ] Key question: does snap lift transfer to domains where model lacks knowledge?
- [x] Status: resubmitted cross-dataset block `44395` completed
- [x] HousingQA follow-up completed: `llm_only` **50.5%**, `vectorless_direct` **50.0%**, `vectorless_nosnap` **52.5%**, `snap_hyde` **50.0%**
- [x] CaseHOLD follow-up completed: `llm_only` **69.5%**, `vectorless_direct` **68.0%**, `vectorless_nosnap` **67.5%**
- [x] Key finding from the April 14 block: parametric reasoning does **not** help on unknown-domain HousingQA or citation-matching CaseHOLD
- [x] Supporting infra update: case-summary build job `44371` completed with **22K summaries**; entity-graph rebuild moved to job `44520` and is **74% done**
- Data: HousingQA at `datasets/housing_qa/`, CaseHOLD at `datasets/casehold/`

### P1.3: Full-Scale N=1195 Validation
- [x] rag_snap_hyde full: **57.9% ✓ DONE**
- [x] vectorless_direct full: **CANCELLED** (job `43471`) — misnamed parametric reasoning, not real corpus search
- [x] vectorless_hybrid full: **CANCELLED** (job `43471`) — same issue
- [x] `subagent_rag` full N=1195: **56.9%** (`680/1195`) — the N=200 edge did **not** hold at scale vs `snap_hyde` **57.9%**
- [x] Update 2026-04-15: `entity_search` full N=1195: **53.2%** (`636/1195`) — real NLP entity-graph corpus search, zero embeddings, 1 LLM call; below full `rag_simple` **54.2%**
- [x] Update 2026-04-15: full `rag_hyde` attempt was **BROKEN** — 100% 11-char HyDE outputs from the same terse generic `hyde` prompt; fixed and resubmitted as job `45350`
- [x] Update 2026-04-15: scale robustness note — `entity_search` fell **6.8pp** (`60.0% -> 53.2%`) while vector `rag_simple` fell only **2.8pp** (`57.0% -> 54.2%`), so NLP entity matching is less robust than vector search at scale
- Data: `logs/experiments.jsonl`

---

## Priority 2: Important Experiments (SHOULD DO)

### P2.1: Fix Historical Vectorless / Parametric Reasoning and Test Snap Contribution
- [x] **vectorless_nosnap** — completed at **59.5%** on Gemma 4 E4B BarExam N=200
- [x] Compare: `vectorless_direct` **64.5%** (with snap) vs `vectorless_nosnap` **59.5%** (without snap) = **+5pp**
- [x] This now directly measures whether snap helps vectorless, mirroring `snap_hyde` vs `rag_hyde`

### P2.2: Subagent Variants
- [ ] **subagent_hyde** — subagent uses HyDE retrieval per gap instead of raw sub-question
- [ ] **subagent_vectorless** — subagent generates knowledge instead of retrieving (no corpus)
- [ ] **subagent_panel** — multiple subagents with different roles (textbook/barprep/casebook)
- [x] Results from subagent_hybrid and subagent_rag_evidence: **DONE** — `subagent_hybrid` 63.5%, `subagent_rag_evidence` 61.0%
- [x] Update 2026-04-15: `subagent_hyde` completed at **62.5%** (`125/200`) with **5.2 avg** calls — below `subagent_rag` **66.0%**
- Code: `eval/eval_harness.py`, subagent runners in the gap-family section

### P2.3: Corpus Structure / Metadata Approaches
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

| Experiment | Result | Status |
|---|---|---|
| Snap vs no-snap ablation | `rag_hyde` 62.5%, `vectorless_nosnap` 59.5%; core lift = +3 / +5 / +5 | ✅ Done |
| Cross-dataset follow-up | HousingQA flat, CaseHOLD negative for new parametric controls | ✅ Done |
| Embedding comparison (7 models × 3 modes) | Cross-encoder dominates | ✅ Done |
| Gap architecture + GAP_MIN_CE fix | gap_rag 63.5%, gap_hyde 62.0% | ✅ Done |
| Anchoring hypothesis | gap_rag_nosnap 64.5% > gap_rag 63.5% | ✅ Done |
| Historical "vectorless" / parametric-reasoning baselines (5 modes) | hybrid 65.0%, direct 64.5% | ✅ Done |
| Subagent RAG | **66.0% NEW BEST** | ✅ Done |
| Subagent follow-ups | hybrid 63.5%, rag_evidence 61.0% | ✅ Done |
| subagent_hyde | 62.5%, below subagent_rag 66.0% | ✅ Done |
| snap_entity_informed | 59.5%, below entity_search 60.0% | ✅ Done |
| snap_hyde full N=1195 | 57.9% | ✅ Done |
| subagent_rag full N=1195 | 56.9%, below snap_hyde 57.9% | ✅ Done |
| entity_search full N=1195 | 53.2%, below rag_simple 54.2% | ✅ Done |
| rag_hyde full N=1195 rerun | broken original attempt; fixed and resubmitted as job `45350` | ⚠️ Resubmitted |
| Case-summary build | 22K summaries built (job `44371`) | ✅ Done |
| Phase 1 alignment (10 modes) | snap_hyde 65.5% best | ✅ Done |
| 180 total experiments | — | ✅ Logged |
| 185 total experiments (as of 2026-04-15) | current count in `logs/experiments.jsonl` | ✅ Logged |

---

## Latest Job Status

| Job | What | Status |
|---|---|---|
| 44371 | case summaries build | Completed — 22K summaries built |
| 44394 | snap ablations | Completed — `rag_hyde` 62.5%, `vectorless_nosnap` 59.5% |
| 44395 | cross-dataset jobs | Completed — HousingQA and CaseHOLD follow-ups logged |
| 44520 | entity graph rebuild | Running — 74% done |
| 45350 | rag_hyde full rerun | Resubmitted — original full run was broken (100% 11-char HyDE outputs) |
| 43471 | vectorless_direct + hybrid full N=1195 | Cancelled — fake vectorless / not real corpus search |

---

## Key Files

| What | Where |
|---|---|
| All results | `logs/experiments.jsonl` |
| Meeting prep | `docs/meeting_2026_04_13.md` |
| Experiment overview | `docs/experiment_overview.md` |
| This action list | `docs/action_items.md` |
| Experiment narratives | `EXPERIMENTS.md` |
| Research state | `RESEARCH.md` |
| Detail logs | `logs/eval_*_detail.jsonl` |
| Formatted readable logs | `/tmp/*_readable.md` (local only) |
