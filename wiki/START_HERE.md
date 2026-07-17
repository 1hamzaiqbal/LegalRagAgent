---
title: Start Here
type: hub
tags: [orientation]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Start here — LegalRagAgent wiki orientation

> **2026-07-17 reset:** read [[research-state-2026-07-17]] first. The active
> science is [[three-dial]]; the parallel gated implementation track is
> [[opd-distillation]]; SCOPE is historical ([[scope-old]]). The material below
> explains how the project arrived here.

**What this project is now.** A legal-RAG research effort that submitted
[[scope-paper-2026]] (SCOPE, a two-call generated-query method) to ICML AI4Law
2026 and was rejected ([[icml-ai4law-2026-rejection]] — read this first; it is
the sharpest external audit of the work). The project's live scientific
assets are no longer the method: they are a **mechanism**
([[geometry-vs-factuality]]: expansion benefit = geometric affinity movement,
not factuality), a **regime law** ([[weak-vs-strong-query-regime]]: expansion
helps ∝ query–corpus gap, hurts strong queries via [[query-drift]]), and an
**operational recipe** ([[regime-routing]]).

**Reading order for a new agent/human:**
1. [[research-state-2026-07-17]] — current evidence-backed snapshot and gates.
2. [[three-dial]] + [[opd-distillation]] — clean development surfaces.
3. [[literature/index]] + [[coverage-audit-2026-07-17]] — primary-source and
   novelty boundary.
4. [[thesis-v2]] — the earlier working thesis with pre-stated predictions and their
   live status (P1 ✓ leakage rejected · P2 ✓ judge wins both regimes ·
   P3 revised: label semantics decide · conversion break-even measured).
5. [[icml-ai4law-2026-rejection]] — what died and why (criticisms C1–C12).
6. [[weak-vs-strong-query-regime]] + [[geometry-vs-factuality]] — the regime
   law and mechanism.
7. [[direction-2026-07]] — the July 2 direction map (historical decision state).
8. [[generated-query-family]] — prior-art landscape (incl. the reviewer-named
   near-twins [[koblex-parser]], [[gure]]).
9. Results pages under `results/` — each records an explicit question, its
   verdict, and evidence paths ([[log]] for chronology).

**House rules.** [[WIKI_GUIDE]] for schema. Cite-or-not gate for any number:
[signoff_log](../docs/signoff_log.md). Skeptic's discipline: verify against
detail logs; the rejection happened partly because framing drifted from
tables. LLM-only is the primary baseline in every comparison, always.

**Repo map (not duplicated here):** eval harness `eval/eval_harness.py`
(65 modes); caches under `caches/`; generated analyses `docs/generated/`;
run ledger `logs/experiments.jsonl`; ideas log [ideas.md](../ideas.md);
cluster runbook [docs/hpc_setup_log.md](../docs/hpc_setup_log.md) + WUSTL EIT.
Persistent literature lives at
`/engrfs/project/jacobsn/hiqbal/literature/legalrag/`; scratch experiment
artifacts remain under `/engrfs/tmp/jacobsn/hiqbal_legalrag/`.
