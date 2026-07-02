---
title: Direction Map — July 2026 (post-rejection)
type: hub
tags: [direction, meeting, roadmap]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Direction map for the 2026-07-02 meeting

Purpose: reviewer critiques + literature are now fully mapped
([[icml-ai4law-2026-rejection]], 23 ingested sources). This page turns that
map into a ranked set of paths so we stop re-deciding and start converting.

## 1. Where we actually stand (honest, three sentences)

The **method paper is dead** — pre-empted by [[koblex-parser]] (verdict:
substantial-overlap-differentiable; fatal *to the method framing*
specifically) and by the benchmark authors' own query-expansion baselines
([[zheng-cslaw]]), with the one novel component (snap/discard) unvalidated
(C7/C12). The **science that replaced it is in good shape and mostly
post-submission**: a per-query geometric mechanism replicated across 7
datasets × 3 retrievers × 4 generators, a falsification of the hallucination
account that survived an independent-judge credibility battery, a regime law
with an operational routing recipe, and the first significant SCOPE answer
win over LLM-only (MedQA q200 +5.5pp, p=0.019). The **bottleneck is
conversion, not discovery**: eleven analysis reports and zero new manuscript
text; the cheap reviewer must-dos (cite the twins, run the ablations, redo
token accounting) are all still open.

## 2. What the literature campaign changed (deltas only)

1. **[[koblex-parser]]** (C6): reviewer's mapping is fair; ParSeR is a
   superset on the overlapping mechanism (multi-provision + Selection stage,
   +19–31pt answer gains that *survive the no-retrieval comparator*). But: no
   snap-answer analog, no discard, no mechanism/regime analysis, civil-law
   Korean, 226 questions, and their Limitations concede possible common-law
   non-transfer. Same author group as [[gure]] (POSTECH/KT) — one group owns
   "generated legal queries"; assume expert review from that circle at any
   legal venue.
2. **[[zheng-cslaw]]** re-read (missed by us *and* both reviewers): the
   benchmark authors already ran legal-tailored generative expansion on these
   exact datasets. Two gifts: (a) calibration — their best Historical-MBE
   Recall@10 is **6.95**; SCOPE's Hit@5 is **9.5–12.1** → "marginal gains" (C5)
   inverts on the retrieval side when properly anchored; (b) their
   rollout-as-pseudo-passage *hurts answers* — published support for the
   discard guardrail, and their "10% retrieval → 2% answer in theory" line is
   the answer-conversion gap stated by the benchmark's own authors.
3. **[[yoon2025leakage]]** (new threat, ACL'25): HyDE-family gains may be
   knowledge leakage (LLM regurgitating gold evidence). A rival explanation to
   our geometry mechanism, currently observationally confounded with it on our
   benchmarks. **We must run their NLI audit** — and legal statutes are the
   best place to *beat* the critique (state statutes are corpus-specific;
   if our lift survives on unmatched rows, our result gets stronger than the
   general-domain family's).
4. **The weak-query space is crowding fast**: [[li2026legalmalr]] (RL
   multi-agent reformulation, Jan'26), LexPath (IRAC-guided expansion,
   May'26), Nguyen'24 COLIEE expansion, LEMUR (trained legal embeddings),
   [[afane2026laborbench]] (same Stanford lab, structure-aware retrieval
   beating Westlaw/Lexis). All retrieval-only. **Nobody owns mechanism,
   regime, falsification, or answer-conversion** — our lane is real but the
   window is narrowing.
5. **[[thinking-machines-expert-judgment]]**: frontier models plateau at
   74–78% on expert judgment even with expert prompts; a
   judgment-label-trained 235B hits 84.7% at 1/14th cost. Thesis: expert
   judgment resists prompt articulation → needs labels. Template for C1 and
   for the answer-conversion attack (see Path C).

## 3. The paths (ranked recommendation)

### Path A — the mechanism paper, IR/NLP venue (RECOMMENDED primary)
**Claim**: *When does generative query expansion help? A geometric
(affinity-margin) account that predicts per-query benefit, falsifies the
hallucination explanation, and yields a regime-routing recipe — validated on
legal, medical, and BEIR corpora.*
- **Assets (≈80% run)**: [[affinity-margin-mechanism]] · [[beir-phase1]] ·
  [[three-retriever-generality]] · [[factuality-falsification]] ·
  [[qpp-routing-negative]] · [[pooling-regime]] · [[musique-cross-domain]] ·
  MedQA q200.
- **Why it wins**: C1 evaporates (right venue), C3/C6 evaporate (no method
  claim), C4 becomes a headline *result*, C5 becomes the second pillar
  (answer-conversion), and the near-twins (Emami'26/Tian'25) both *punt*
  exactly what we have (generation-aware, mechanism-level analysis).
- **Missing before writing**: leakage audit (kills or hardens everything);
  retrieval-side bootstrap significance (free, on caches); keep-vs-discard +
  a0 ablations reframed as *margin levers*; held-out regime-routing check;
  MedQA full-N. Claude second judge is polish, not gating.
- **Risk**: BEIR strong-query slice makes expansion look bad-by-default;
  framing must be "when/why", not "expansion is good".

### Path B — legal-venue revision of SCOPE (NOT recommended as primary)
Everything C2/C6/C8 demands: cite+run ParSeR/GuRE baselines, corpus long-tail
+ jurisdiction analysis (the [[afane2026laborbench]] StatReg corpus is the
ready substrate), Zheng-calibrated framing, filled matrix, CIs. **Cost**:
GuRE SFT + ParSeR reimplementation + heavy eval; into a space one expert
group dominates, with our own data showing the method delta is ~null (C7).
Only worth it *as a workshop-scale companion* (e.g. NLLP) reusing Path A's
machinery on the legal slice — not as the flagship.

### Path C — expert-judgment replication for legal RAG (the new bet; pilot now, scale later)
[[expert-judgment-replication]]: learn to replicate *lawyer judgments* (of
passage relevance / answer quality) instead of optimizing exact-match through
a general-domain CE. Attacks the answer-conversion wall at the metric layer
(our CE demonstrably buries gold: median rank 4–5, 35–40% below rank 5), gives
the C1-grade practitioner grounding, and composes with Path A (the geometry
features feed the judge). TM's recipe details (labels not rubrics;
disagreement-routed verification; leave-one-out ablations) are the template.
- **Pilot that costs almost nothing**: our signed logs already hold ~50K
  (question, evidence, answer, correct?) tuples across 6 benchmarks × 3
  models; train/evaluate a small judgment model (does this passage change the
  answer? is this answer legally supported?) against held-out outcomes before
  any human labels. If the learned judge reranks better than the CE on
  weak-query legal, that's a paper-grade result on its own.
- **Open**: real lawyer-label source (Zheng-style law-student annotation?
  disagreement-routing makes it cheap); Bridgewater-scale data is not
  available to us — the pilot must prove signal at academic scale.
- Harness note: this does *not* need the LangGraph agent; a thin
  SFT/LoRA + eval stack (or Tinker-style API if accessible) is the right
  tooling. The `main.py` pipeline stays legacy.

### Kill / keep-parked
- Per-query QPP routing: **killed** (τ 0.05–0.11 OOD vs 0.5 bar) — cite as
  principled negative only.
- 3-candidate diversity, fixed-medoid exemplars, always-pool: killed by data.
- HotpotQA-distractor as weak-query probe: invalid (raw 99.4%).
- Full agentic system (multi-agent, planner): parked until a component
  (judge, router) independently earns its slot — per the atomic-rebuild rule.

## 4. Near-term experiment queue (cheap → expensive, each with its motivation)

Ordering logic: **de-risk before decorate** — items 1–3 can *kill or harden*
the Path A thesis, so they run before anything that merely extends it.

1. **Yoon NLI leakage audit** on BarExam/Housing/MedQA pseudo-docs — ~1
   GPU-day, caches exist. *Motivation*: [[yoon2025leakage]] is the one
   published result that could explain away our entire weak-query lift
   ("the LLM just regurgitates gold evidence it memorized"). Their protocol —
   NLI-match generated docs against gold, split the lift into matched vs
   unmatched rows — is cheap and decisive: if SCOPE's Hit@5 gain survives on
   *unmatched* rows (plausible: state statutes and MBE rule paragraphs are
   corpus-specific), our geometry story beats their leakage story and the
   paper gains a section; if it doesn't, we need to know before writing, not
   from a reviewer. Gates everything. (C9/C4)
2. **Retrieval-side bootstrap CIs + McNemar on all cached Hit@5 deltas** —
   hours, free, pure analysis over existing caches. *Motivation*: V1's audit
   found *zero* retrieval-side significance tests in the repo; every
   Hit@5 comparison we cite (snap-vs-HyDE, SCOPE-vs-raw, pool-vs-single) is
   point-estimate-grade. One afternoon converts every retrieval claim from
   "would re-trigger C7" to publication-grade. (C7/C11)
3. **C12 ablation trio** (q200 first): (a) pass a0 into call 2, (b)
   keep+concat the raw question Query2doc-style, (c) conclusion-banned
   generation ParSeR-style. *Motivation*: these three cells convert SCOPE's
   two asserted design choices (privacy guardrail, query discard) from
   diagram features into findings — and reframed as *margin levers* they feed
   the mechanism paper directly (does conditioning on a draft answer move
   affinity more than banning conclusions?). [[query2doc]]'s Table 4 predicts
   (b) helps on strong queries and hurts on weak; [[zheng-cslaw]]'s
   rollout-as-passage result predicts (a) hurts. Cheap, pre-registered
   predictions, either outcome is a paragraph. (C12/C7)
4. **MedQA full-N answer cells** — 1273×2 calls, caches banked. *Motivation*:
   MedQA q200 is the **only cell anywhere where SCOPE significantly beats
   LLM-only on answers** (+5.5pp, p=0.019) — i.e., the only existing
   counter-example to C9's "answers are parametric" on our stack. If it
   holds at full N it anchors the answer-side story of Path A; if it
   doesn't, we stop citing it. Either way we must know. (C5/C9)
5. **Corpus long-tail + jurisdiction breakdown** — analysis-lane; gold-citation
   frequency over our corpora, jurisdiction split for Housing;
   [[afane2026laborbench]]'s StatReg (4.6M sections, all states) as substrate.
   *Motivation*: the C8 ask, and [[gure]]'s Table 5/Fig 4 give the exact
   template; it also answers a question Path A needs anyway — *is the
   geometric mechanism uniform across passage-frequency strata, or do
   generated queries only reach popular rules?* (C8)
6. **Held-out regime-routing validation** — analysis-lane. *Motivation*: the
   routing recipe ([[regime-routing]]) is currently in-sample; one held-out
   split turns "we suggest routing" into "routing validated," which is the
   deployable-recipe half of Path A's conclusion. (—)
7. **Judge pilot from existing logs (Path C v0, Tinker)** — ~$50–150 API.
   *Motivation*: three birds. (i) The answer-conversion wall is a *selector*
   problem before it is a reader problem — the CE demonstrably buries gold it
   already holds (median rank 4–5; BarExam pool 3.9% vs SCOPE-alone 12.0%);
   a judge trained on our free outcome labels tests whether *learned legal
   relevance* recovers that buried headroom. (ii) It is the
   [[thinking-machines-expert-judgment]] thesis at academic scale — prompted
   judges plateau, trained judges don't; we already saw the prompted-judge
   version maximize exposure but not answers. (iii) It is the C1 answer with
   teeth: a model of legal *judgment*, not another retrieval trick. Running
   NOW on Tinker (see [[judge-pilot-v0]]). (C1 + answer-conversion)
8. **Union+CE-rerank full-N downstream** — 48K calls, Groq Batch.
   *Motivation*: the one deployable recipe from May (Housing q200: union 65.0%
   vs raw 62.0% vs SCOPE 59.0%) was never scaled; it is the answer-side proof
   that regime-aware fusion, not expansion alone, fixes the strong-query
   regression. Also the natural A/B partner for the judge pilot (CE vs judge
   at the same fusion point). (C10)
9. **GuRE SFT baseline on BarExamQA pairs** — ~60 GPU-h, public code.
   *Motivation*: only needed for Path B / legal-venue positioning: it answers
   "why not just train a rewriter?" with data instead of a shrug. Their own
   result (10K pairs suffice) makes it feasible; BarExamQA has ~1.2K
   gold pairs, so this doubles as a data-scarcity stress test of their claim.
   Skip if Path B is shelved. (C8)
10. **Claude second judge + Housing judge coverage** — ~$25–50.
    *Motivation*: upgrades the falsification from "single independent judge"
    to the pre-registered two-judge closeout; pure credibility polish for
    Path A's headline result. (C4)

**And the standing rule: every completed row converts to manuscript text the
same week.** The May sprint proved the lab lane works; the paper lane is the
one that starved.

## 5. Proposed meeting agenda
1. Review scoreboard (10 min): C1–C12 statuses — 2 addressed-with-evidence,
   5 partially, 3 reframed, 2 open ([[icml-ai4law-2026-rejection]]).
2. The two verdicts that matter: KoBLEX adjudication (method dead,
   program alive) and Zheng calibration (our retrieval numbers beat the
   benchmark authors' best — the C5 inversion).
3. Decide the flagship: Path A framing + venue (target list + dates to
   check: NLLP@EMNLP for the legal companion; ECIR/SIGIR/WSDM cycle or ARR
   for the main paper).
4. Greenlight queue items 1–4 (all cheap, all de-risking).
5. Path C pilot: yes/no on the logs-only judge experiment; if yes, where the
   eventual lawyer labels come from.

## Links
[[icml-ai4law-2026-rejection]] · [[snap-vs-hyde-ledger]] ·
[[geometry-vs-factuality]] · [[weak-vs-strong-query-regime]] ·
[[answer-conversion-gap]] · [[regime-routing]] ·
[[expert-judgment-replication]] · [[thinking-machines-expert-judgment]] ·
[[yoon2025leakage]] · [[koblex-parser]] · [[gure]] · [[zheng-cslaw]] ·
[ideas.md](../ideas.md)
