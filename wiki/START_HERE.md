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
1. [[icml-ai4law-2026-rejection]] — what died and why (criticisms C1–C12).
2. [[weak-vs-strong-query-regime]] + [[geometry-vs-factuality]] — what lives.
3. [[direction-2026-07]] — where we're going (meeting map).
4. [[generated-query-family]] — the prior-art landscape (incl. the reviewer-named
   near-twins [[koblex-parser]], [[gure]]).
5. Results pages under `results/` — each links its evidence artifacts.

**House rules.** [[WIKI_GUIDE]] for schema. Cite-or-not gate for any number:
[signoff_log](../docs/signoff_log.md). Skeptic's discipline: verify against
detail logs; the rejection happened partly because framing drifted from
tables. LLM-only is the primary baseline in every comparison, always.

**Repo map (not duplicated here):** eval harness `eval/eval_harness.py`
(65 modes); caches under `caches/`; generated analyses `docs/generated/`;
run ledger `logs/experiments.jsonl`; ideas log [ideas.md](../ideas.md);
cluster runbook `docs/hpc_setup_log.md` + WUSTL EIT
(`/engrfs/tmp/jacobsn/hiqbal_legalrag/`, papers archived under
`references/papers/`).
