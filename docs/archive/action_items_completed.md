### P1.2: Cross-Dataset Validation
- [x] Cross-dataset block `44395` completed
- [x] HousingQA follow-up completed at N=200: `llm_only` **50.5%**, `vectorless_direct` **50.0%**, `vectorless_nosnap` **52.5%**, `snap_hyde` **50.0%**
- [x] CaseHOLD follow-up completed at N=200: `llm_only` **69.5%**, `vectorless_direct` **68.0%**, `vectorless_nosnap` **67.5%**
- [x] Supporting infra update: case-summary build job `44371` completed with **22K summaries**
- [x] Key question answered: not universally; the April 14 block is flat on HousingQA and negative on CaseHOLD for the new parametric controls
- [x] Key finding from the April 14 block: parametric reasoning does **not** help on unknown-domain HousingQA or citation-matching CaseHOLD
- [x] Entity-graph rebuild moved to job `44520` and was last noted at **74%** on 2026-04-14
- Data: HousingQA at `datasets/housing_qa/`, CaseHOLD at `datasets/casehold/`

### P1.3: Full-Scale N=1195 Validation
- [x] rag_snap_hyde full: **57.9% paired rerun** (earlier best run: **58.6%**) (**both pre-leak-fix; clean reruns pending**) ✓ DONE
- [x] vectorless_direct full: **CANCELLED** (job `43471`) — misnamed parametric reasoning, not real corpus search
- [x] vectorless_hybrid full: **CANCELLED** (job `43471`) — same issue
- [x] `subagent_rag` full N=1195: **56.9%** (`680/1195`) — the N=200 edge did **not** hold at scale vs the repaired HyDE pair at **57.9%**
- [x] Update 2026-04-15: `entity_search` full N=1195: **53.2%** (`636/1195`) — real NLP entity-graph corpus search, zero embeddings, 1 LLM call; below full `rag_simple` **54.2%**
- [x] Update 2026-04-17: fixed full `rag_hyde` rerun completed at **57.9%** (`692/1195`) after invalidating the broken 11-character-output attempts (**still pre-leak-fix; clean rerun pending**)
- [x] Update 2026-04-16: full `ce_threshold` completed at **55.9%** (`668/1195`)
- [x] Update 2026-04-16: full `gap_rag_nosnap` completed at **55.9%** (`668/1195`)
- [x] Update 2026-04-16: full `subagent_rag` 1-gap rerun completed at **57.2%** (`684/1195`)
- [x] Update 2026-04-15: scale robustness note — `entity_search` fell **6.8pp** (`60.0% -> 53.2%`) while vector `rag_simple` fell only **2.8pp** (`57.0% -> 54.2%`), so NLP entity matching is less robust than vector search at scale
- Data: `logs/experiments.jsonl`

### P2.1: Fix Historical Vectorless / Parametric Reasoning and Test Snap Contribution
- [x] **vectorless_nosnap** — completed at **59.5%** on Gemma 4 E4B BarExam N=200
- [x] Compare: `vectorless_direct` **64.5%** (with snap) vs `vectorless_nosnap` **59.5%** (without snap) = **+5pp**
- [x] This now directly measures whether snap helps vectorless, mirroring `snap_hyde` vs `rag_hyde`

### P2.3: Combo-Mode Anchoring Controls
- [x] **`snap_hyde_report`** — completed at **66.0%**; report-only compression is neutral vs fixed `rag_hyde`
- [x] **`snap_hyde_report_snap`** — completed at **64.0%**; showing snap hurts **-2pp**
- [x] **`subagent_rag_snap`** — completed at **63.0%**; showing snap hurts **-3pp**
- [x] **`subagent_rag_full`** — completed at **62.0%**; max information hurts **-4pp**
- [x] **Combo-mode block `48393`** — completed and closed; visible snap consistently hurts the final decision-maker

## What's Done (reference)

| Experiment | Result | Status |
|---|---|---|
| Snap vs no-snap ablation | fixed `rag_hyde` 57.9 vs paired `snap_hyde` 57.9; core lift = 0 / +5 / +5 | ✅ Done |
| Cross-dataset follow-up | HousingQA flat, CaseHOLD negative for new parametric controls | ✅ Done |
| Embedding comparison (7 models × 3 modes) | Cross-encoder dominates | ✅ Done |
| Gap architecture + GAP_MIN_CE fix | gap_rag 63.5%, gap_hyde 62.0% | ✅ Done |
| Anchoring hypothesis | gap_rag_nosnap 64.5% > gap_rag 63.5% | ✅ Done |
| Historical "vectorless" / parametric-reasoning baselines (5 modes) | hybrid 65.0%, direct 64.5% | ✅ Done |
| Subagent RAG | **66.0% NEW BEST** | ✅ Done |
| Subagent follow-ups | hybrid 63.5%, rag_evidence 61.0% | ✅ Done |
| subagent_hyde | 62.5%, below subagent_rag 66.0% | ✅ Done |
| snap_entity_informed | 59.5%, below entity_search 60.0% | ✅ Done |
| snap_hyde full N=1195 | 57.9% paired rerun; earlier best run 58.6% | ✅ Done |
| subagent_rag full N=1195 | 56.9%, below the repaired HyDE pair at 57.9% | ✅ Done |
| entity_search full N=1195 | 53.2%, below rag_simple 54.2% | ✅ Done |
| rag_hyde full N=1195 rerun | **57.9%** after the repaired prompt fix; ties paired `snap_hyde` | ✅ Done |
| ce_threshold full N=1195 | **55.9%** — barely above llm_only (55.5%) | ✅ Done |
| gap_rag_nosnap full N=1195 | **55.9%** — same as ce_threshold | ✅ Done |
| subagent_rag (1-gap) full N=1195 | **57.2%** — improved prompt, up from 56.9% | ✅ Done |
| Case-summary build | 22K summaries built (job `44371`) | ✅ Done |
| Phase 1 alignment (10 modes) | Historical alignment block completed; later follow-ups raised the current top tier to 66.0% | ✅ Done |
| 195 total experiments (as of 2026-04-19) | current count in `logs/experiments.jsonl` | ✅ Logged |
| New combo modes | snap_hyde_report 66.0%, snap_hyde_report_snap 64.0%, subagent_rag_snap 63.0%, subagent_rag_full 62.0% (job `48393`) | ✅ Done |
