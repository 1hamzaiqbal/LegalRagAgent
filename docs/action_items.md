# Action Items — Paper Sprint

Target venues:
- **EMNLP 2026** — due May 20, 2026 (8 pages)
- **ICML AI4Law Workshop** — due May 22, 2026 (8 pages)
- **ACL Rolling Review** — rolling deadlines
- Fallback: NAACL 2027 (August deadline)

---

## Paper Narrative

**Core claim:** Letting the LLM reason first ("snap") before retrieval is the single biggest contributor to legal QA accuracy. This lift generalizes across retrieval methods (RAG, HyDE, vectorless) and architectures (simple, gap-informed, subagent).

**Supporting evidence:**
- Snap reasoning adds +5pp universally (snap_rag 62% vs rag_simple 57%)
- HyDE leverages snap reasoning for passage-form retrieval (+3.5pp)
- Vectorless RAG (LLM generates knowledge from snap reasoning) matches vector retrieval
- Subagent architecture (snap → gap analysis → subagent reports) achieves new best (66.0%)
- Three identified failure modes: noise, anchoring, genre mismatch

**Open question for paper:** Does the snap lift generalize to other datasets/corpora? Same universal lift?

---

## Priority 1: Critical Experiments (MUST DO for paper)

### P1.1: Snap vs No-Snap Ablation (the paper's core comparison)
- [ ] **Pure HyDE (no snap)** — test `rag_hyde` on Gemma 4 E4B N=200. We have this mode already but never ran it on Gemma. This is snap_hyde minus the snap.
- [ ] **Compare:** snap_hyde (65.5%) vs pure hyde (?%) = the snap contribution for HyDE
- [ ] **snap_rag (62.0%) vs rag_simple (57.0%)** = already done, +5pp ✓
- [ ] **snap_vectorless vs pure vectorless** — need to implement a vectorless mode that doesn't snap first. Tests if snap helps vectorless too.
- Data: `logs/experiments.jsonl`, detail logs in `logs/eval_*_detail.jsonl`

### P1.2: Cross-Dataset Validation
- [ ] Copy housing QA data to cluster: `scp datasets/housing_qa/ wustl:...`
- [ ] Run on **HousingQA**: llm_only, rag_simple, snap_hyde, vectorless_direct, subagent_rag (N=200 each)
- [ ] Run on **CaseHOLD**: same modes (already have corpus on cluster?)
- [ ] Key question: does snap lift transfer to domains where model lacks knowledge?
- Data: HousingQA at `datasets/housing_qa/`, CaseHOLD at `datasets/casehold/`

### P1.3: Full-Scale N=1195 Validation
- [ ] snap_hyde full: **57.9% ✓ DONE**
- [ ] vectorless_direct full: **RUNNING** (job 43471)
- [ ] vectorless_hybrid full: **RUNNING** (job 43471)
- [ ] subagent_rag full N=1195: **QUEUE after vectorless finishes**
- Data: `logs/experiments.jsonl`

---

## Priority 2: Important Experiments (SHOULD DO)

### P2.1: Fix Vectorless and Test Snap Contribution
- [ ] **vectorless_nosnap** — implement a vectorless mode without snap step (just question → generate knowledge → answer). 2 calls instead of 3.
- [ ] Compare: vectorless_direct (64.5%, with snap) vs vectorless_nosnap (?%, without snap)
- [ ] This directly measures whether snap helps vectorless, mirroring the snap_hyde vs pure_hyde comparison

### P2.2: Subagent Variants
- [ ] **subagent_hyde** — subagent uses HyDE retrieval per gap instead of raw sub-question
- [ ] **subagent_vectorless** — subagent generates knowledge instead of retrieving (no corpus)
- [ ] **subagent_panel** — multiple subagents with different roles (textbook/barprep/casebook)
- [ ] Results from subagent_hybrid and subagent_rag_evidence: **RUNNING** (job 43499)
- Code: `eval/eval_harness.py`, runners near line 930

### P2.3: Corpus Structure / Metadata Approaches
- [ ] **Proximity RAG** — retrieve nearby passages from same case_id after finding one good passage
- [ ] **Topic-filtered retrieval** — classify passages by bar exam subject (7 topics), retrieve only from matching topic
- [ ] **PageIndex-style ToC** — build a table of contents from the corpus, let LLM navigate. NOTE: PageIndex is designed for single documents, our corpus is 686K flat passages. May need adaptation.
- [ ] How good is existing metadata? Source: 98.9% caselaw, 0.3% mbe, 0.8% wex. Gold passages are ALL from the 2,318 mbe passages (0.3% of corpus). Filtering to mbe-only would be trivially better but defeats the purpose.
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
| Embedding comparison (7 models × 3 modes) | Cross-encoder dominates | ✅ Done |
| Gap architecture + GAP_MIN_CE fix | gap_rag 63.5%, gap_hyde 62.0% | ✅ Done |
| Anchoring hypothesis | gap_rag_nosnap 64.5% > gap_rag 63.5% | ✅ Done |
| Vectorless RAG (5 modes) | hybrid 65.0%, direct 64.5% | ✅ Done |
| Subagent RAG | **66.0% NEW BEST** | ✅ Done |
| snap_hyde full N=1195 | 57.9% | ✅ Done |
| Phase 1 alignment (10 modes) | snap_hyde 65.5% best | ✅ Done |
| 166 total experiments | — | ✅ Logged |

---

## What's Running

| Job | What | Est. Done |
|---|---|---|
| 43471 | vectorless_direct + hybrid full N=1195 | ~4h |
| 43458 | gap_vectorless + gap_hyde_nosnap N=200 | ~2h |
| 43499 | subagent_hybrid + subagent_rag_evidence N=200 | ~8h |

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
