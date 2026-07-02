---
title: Start Here
type: hub
tags: [orientation]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Start here — LegalRagAgent wiki orientation

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
1. [[thesis-v2]] — the working thesis with pre-stated predictions and their
   live status (P1 ✓ leakage rejected · P2 ✓ judge wins both regimes ·
   P3 revised: label semantics decide · conversion break-even measured).
2. [[icml-ai4law-2026-rejection]] — what died and why (criticisms C1–C12).
3. [[weak-vs-strong-query-regime]] + [[geometry-vs-factuality]] — the regime
   law and mechanism.
4. [[direction-2026-07]] — the direction map (queue statuses updated inline).
5. [[generated-query-family]] — prior-art landscape (incl. the reviewer-named
   near-twins [[koblex-parser]], [[gure]]).
6. Results pages under `results/` — each records an explicit question, its
   verdict, and evidence paths ([[log]] for chronology).

**House rules.** [[WIKI_GUIDE]] for schema. Cite-or-not gate for any number:
[signoff_log](../docs/signoff_log.md). Skeptic's discipline: verify against
detail logs; the rejection happened partly because framing drifted from
tables. LLM-only is the primary baseline in every comparison, always.

**Repo map (not duplicated here):** eval harness `eval/eval_harness.py`
(65 modes); caches under `caches/`; generated analyses `docs/generated/`;
run ledger `logs/experiments.jsonl`; ideas log [ideas.md](../ideas.md);
cluster runbook [docs/hpc_setup_log.md](../docs/hpc_setup_log.md) + WUSTL EIT
(`/engrfs/tmp/jacobsn/hiqbal_legalrag/`, papers archived under
`references/papers/`).
