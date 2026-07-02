---
title: ICML AI4Law 2026 Rejection — SCOPE (Submission 97)
type: review
tags: [review, rejection, scope, ai4law, postmortem]
created: 2026-07-02
updated: 2026-07-02
status: draft
---

# ICML AI4Law 2026 rejection — criticism inventory & assessment

**Submission**: #97, *"SCOPE: When Generated Legal Queries Help Legal RAG"*
(Iqbal, Li, Li, Huang, Huang), submitted 22 May 2026. **Decision: Reject**
(12 Jun 2026). Two reviews, both **Rating 2: Strong rejection** — Reviewer 7gGP
(confidence 4) and Reviewer oSUu (confidence 5, clearly expert; the review is
effectively a free roadmap). Source: [feedback.md](../../official_paper_and_review_icml_ai_4_law/feedback.md),
[paper PDF](../../official_paper_and_review_icml_ai_4_law/97_SCOPE_When_Generated_Legal_.pdf).

What we submitted: [[scope]] two-call pipeline evaluated on BarExamQA
(weak-query regime) and HousingQA (strong-query regime) with Llama 3.1 8B /
Gemma 4 26B / Llama 3.3 70B; headline = BarExamQA Hit@5 1.4% → 9.5–12.1% with
answer gains framed vs raw-question RAG. See [[scope-paper-2026]] for the
paper's own content.

**Key timing fact**: our internal repositioning
([RELATED_WORK_GROUNDING](../../paper/submission/RELATED_WORK_GROUNDING.md),
2026-05-26) had *already* concluded — before reviews arrived — that the method
claim was weak/risky, the routing idea was taken, and the mechanism was the
open anchor. The reviews independently confirmed that read from the IR side and
added the **legal-NLP blind spot** (KoBLEX/ParSeR, GuRE) that our IR-focused
scan missed entirely.

## Criticism inventory (C1–C12)

Statuses: `valid-open` (right, nothing done yet) · `valid-addressed-since`
(right, post-submission work answers it — evidence linked) · `valid-reframed`
(right about the paper as framed; the fix is framing/venue, not new
experiments) · `partially-valid`.

### C1 — Not about legal reasoning or practitioner processes (7gGP) — `valid-reframed`
"Focuses primarily on retrieval optimization techniques rather than legal
reasoning principles or legal practitioners' actual analytical processes."
**Assessment**: true as venue-fit critique — AI4Law wanted legal-domain
insight; we submitted an IR-mechanics paper wearing a legal costume. The snap
answer was *motivated* by practitioner intuition (form hypothesis → find
authority → judge on merits) but nothing in the paper engages actual legal
research process literature. Two coherent responses: (a) take the
generalization results ([[beir-phase1]], MedQA, MuSiQue) to their logical
conclusion and target an **IR/NLP venue** where C1 evaporates; (b) genuinely
engage the practitioner dimension — the [[expert-judgment-replication]]
direction (Thinking Machines pattern applied to legal judgment) is the serious
version of this. Half-measures re-trigger this review.

### C2 — Insufficient legal-NLP literature grounding (7gGP + oSUu W2) — `valid-open` → being fixed
Neither benchmarks nor baselines reflect legal-NLP community prior work.
**Assessment**: correct, and our own grounding doc proves the point — it
repositioned against QPP/expansion IR literature but cited *zero* legal-IR
work. The reviewer-named misses ([[koblex-parser]], [[gure]]) are exactly the
sort of thing the wiki's [[legal-rag-benchmarks]] + discovery sweeps now
inventory. A legal-venue revision must cite and ideally *run* these.

### C3 — "Essentially HyDE applied to the legal domain" (7gGP) — `partially-valid`
**Assessment**: mechanically fair — SCOPE is a [[generated-query-family]]
member; the structural delta vs [[hyde]] is (i) snap-answer conditioning,
(ii) embed pseudo-doc *only* (no query-vector mixing), (iii) private-draft
discard guardrail. Our grounding doc already narrowed the novelty claim to
composition + mechanism + regime theory. What survives post-submission is not
"a new method" but the **mechanism account** ([[geometry-vs-factuality]],
CE-affinity margin) and the **[[weak-vs-strong-query-regime]]** theory — those
are contributions HyDE does not contain.

### C4 — Inconsistency: relies on LLM generation while stressing LLM limits; fabrication risk (7gGP) — `valid-addressed-since` (strongest rebuttal we own)
**Assessment**: the concern is coherent, but our post-submission
**factuality falsification** result answers it empirically: SCOPE pseudo-document
*factuality* (LLM-judged, gemma + full-N gpt-4o re-judge) barely predicts
retrieval failure (AUC ≈ 0.55–0.58, marginal lift after geometry ≈ +0.001–0.003)
while the *geometric* affinity margin predicts it strongly (AUC 0.791 legal
pooled, 0.94 BEIR ΔM-target). I.e., **hypothetical documents don't need to be
true to work — they need to move the query toward the gold region of embedding
space; and when they fail it's geometry, not hallucination**. Artifact-verified
2026-07-02 (adversarial pass: all four memory claims CONFIRMED). Carry the
caveats: single independent judge (Claude pending), HousingQA not
judge-covered, and on BarExamQA specifically factuality-vs-geometry are
correlated (raw-top3 factuality AUC 0.757 there). See
[[geometry-vs-factuality]], [[factuality-falsification]].

### C5 — Gains are marginal (7gGP + oSUu W3) — `partially-valid`
**Assessment**: answer-side, yes (that's the [[answer-conversion-gap]] — 8×
retrieval lift moved average accuracy 72.3 → 72.9). Retrieval-side, no — Hit@5
1.4 → 12.1 on BarExamQA is a real, large, replicated effect. The honest paper
says: expansion fixes *retrieval exposure* in weak-query regimes; converting
exposure into answers is a separate open bottleneck that we characterize.

### C6 — Uncited near-twin ParSeR/KoBLEX (oSUu W1) — `valid-open`, severity pending adjudication
Reviewer maps parametric provisions → pseudo-documents, Retrieve-Rerank-Selection
→ retrieve-rerank-answer nearly one-to-one. **Assessment**: we did not know
this paper. Adjudication of true overlap (multi-hop Korean statutory QA +
selection stage, no snap/discard) is running — see [[koblex-parser]]. Whatever
the verdict, *any* legal-venue resubmission must cite and position against it;
if overlap is substantial, the method-novelty framing is dead (consistent with
the grounding doc's independent conclusion) and the mechanism/regime framing is
what remains.

### C7 — Snap-conditioning shows no measurable benefit vs HyDE; no significance test (oSUu W1) — `valid` and confirmed by our own signed rows
**Assessment**: the reviewer inferred this from paper deltas (+1.2/+0.7/+0.5pp
Hit@5; +0.8/+1.7/−0.5pp answers; pooled +0.1pp). The bitter irony: **the tests
existed** — 16 McNemar-tested full-N snap-vs-HyDE answer pairs sit in
[docs/signoff_log.md](../../docs/signoff_log.md); the paper just omitted them. Verified compilation
([[snap-vs-hyde-ledger]]): 13/16 non-significant; the significant cells split
by dataset (pro-snap Legal-Link-EU Gemma +4.17pp p=0.004; pro-HyDE HousingQA
70B −6.45pp p=1.4e-28 unfiltered / −2.57pp p=1.7e-06 state-filtered — worse
than "parity"). Retrieval-side direction flips by dataset and has *no*
significance tests anywhere yet. **Consequence**: stop claiming snap > HyDE.
Reframe snap/discard vs keep/concat as *mechanism probes that vary the
affinity margin*. The defensible snap-side assets are (i) the BEIR
drift-robustness finding (HyDE −31.3pp vs SCOPE −12.2pp pooled Hit@5, 5/5
datasets, 4 generators — a variance claim, not a mean-lift claim) and (ii)
the honest "answer-equivalent, retrieval-regime-dependent" statement drafted
in the ledger page.

### C8 — No trained-rewriter baseline (GuRE); no corpus-level distributional analysis (oSUu W2) — `valid-open`
**Assessment**: correct on both. GuRE defines the obvious legal-domain
supervised comparator; we never ran any trained rewriter. And we have no
passage-frequency/long-tail breakdown of where SCOPE's gains live (our
per-query margin analysis is adjacent but not the corpus-level
characterization asked for). Both are concrete, runnable items — see
[[gure]] for feasibility notes.

### C9 — Headline framed against the weakest baseline (oSUu W3) — `valid`, framing error we must never repeat
Raw-question RAG (69.0 avg) is *below* LLM-only (72.3) on BarExamQA; vs
LLM-only, SCOPE deltas are −0.4/+1.2/+1.0pp. **Assessment**: correct and
self-inflicted. LLM-only is the primary baseline from now on, in every table,
in this wiki and any paper. The deep issue it exposes is real and is our
second pillar: the [[answer-conversion-gap]].

### C10 — HousingQA regression framed as "parity" (oSUu W3) — `valid`, with a post-submission explanation
SCOPE 59.0/59.6 vs raw RAG 62.3/62.1 (pooled −2.9pp). **Assessment**: correct;
"parity" was spin. Post-submission we can *explain* it ([[query-drift]] on
strong queries; expansion is net-negative there across BEIR too) and *avoid*
it ([[regime-routing]]: vanilla SCOPE only on weak-query regimes; raw∪SCOPE
pooling on strong — recovers +2–3pp instead of −3pp). The revision should
present Housing as the predicted-failure half of the regime theory, not as
parity.

### C11 — Rigor gaps: 31/42 cells, no CIs, token accounting excludes stage-1 (oSUu) — `valid-open`, mechanical
**Assessment**: all three correct. Fixes are mechanical: fill the HousingQA
column (Gemma 26B rows exist partially in the comprehensive matrix), report
bootstrap CIs + McNemar on every headline delta, and report total-pipeline
token cost (SCOPE's call-1 emits snap+passage; hiding it flattered us).

### C12 — Confirmation-bias guardrail asserted, never ablated (oSUu) — `valid-open`, cheapest high-value experiment
**Assessment**: correct. Two-cell ablation: (a) pass a0 into the answer call
(tests the guardrail), (b) keep-vs-discard the raw question in the retrieval
query, Query2doc-style concat (tests the discard design against
[[query2doc]]'s Table-4 evidence that discard is usually worse). Note we
already have adjacent evidence: the repo's earlier finding that *showing the
snap answer letter to the final agent always hurts* (regression-tested,
`tests/test_sanitizer.py`) — that is guardrail-supporting but was never framed
as the ablation the reviewer asked for.

## What the reviews demand of any resubmission (checklist)
1. Cite + engage [[koblex-parser]] and [[gure]]; position inside legal-IR (C2, C6, C8).
2. LLM-only as primary baseline; every delta with McNemar + CI (C9, C11).
3. Significance-tested SCOPE-vs-HyDE on retrieval *and* answers — and accept
   the likely null on answers; pivot the claim to mechanism/robustness (C7).
4. Corpus-level breakdown (passage frequency, jurisdiction) of where gains live (C8).
5. Guardrail + keep-vs-discard ablations (C12).
6. Full matrix or a principled inclusion rule; honest HousingQA regression framing (C10, C11).
7. Either a legal-reasoning contribution or an IR venue (C1).

## Post-review discoveries (2026-07-02 literature campaign)

Things the campaign surfaced that *neither we nor the reviewers* had:
- **The benchmark authors already ran legal query expansion** on these exact
  datasets ([[zheng-cslaw]]: GPT-3.5 structured issue+rule rollouts). Their
  best released-set Recall@10 is 6.95; SCOPE's Hit@5 is 9.5–12.1 — properly
  anchored, our retrieval result *beats the benchmark authors' best*, which
  partially inverts C5 on the retrieval side. Their "10% retrieval → 2%
  answer in theory" and rollout-as-passage-hurts findings also pre-figure the
  answer-conversion gap and the discard guardrail.
- **A rival explanation exists and is untested on our data**:
  [[yoon2025leakage]] (ACL'25) — HyDE-family gains may be knowledge leakage.
  Until we run their NLI matched/unmatched audit on our corpora, their story
  and our geometry story are confounded on BarExamQA. Highest-priority cheap
  experiment ([[direction-2026-07]] queue #1).
- **The space is crowding**: [[li2026legalmalr]], LexPath'26, Nguyen'24
  COLIEE, LEMUR'26, [[afane2026laborbench]] — all retrieval-only, none doing
  mechanism/regime/conversion analysis. Lane open; window narrowing.
- **Sociology note**: [[koblex-parser]] and [[gure]] share an author group
  (POSTECH/KT). Expect expert review from that circle at any legal venue.

## Meta-lessons (process)
- **The IR lit scan was necessary but not sufficient** — domain-venue reviews
  come from domain literature. A submission to venue X needs a scan of X's own
  community output (NLLP/JURIX/ICAIL/CSLAW for law), not just the method's home
  field. The wiki's discovery sweeps are the standing fix.
- **Internal honesty beat external honesty to the punch**: the 05-26 grounding
  doc predicted W1/W3 in substance. If we had held the submission to that doc's
  standard ("narrow the novelty claim, name query drift, LLM-only primary"),
  the reviews would have been survivable. Lesson: when an internal audit says
  the anchor claim is weak, *change the anchor before submitting*, not after.
- oSUu's review is a gift: treat it as a co-authored revision plan.

## Links
[[scope]] · [[scope-paper-2026]] · [[koblex-parser]] · [[gure]] · [[hyde]] ·
[[query2doc]] · [[weak-vs-strong-query-regime]] · [[query-drift]] ·
[[answer-conversion-gap]] · [[geometry-vs-factuality]] · [[regime-routing]] ·
[[snap-vs-hyde-ledger]] · [[direction-2026-07]] ·
[RELATED_WORK_GROUNDING](../../paper/submission/RELATED_WORK_GROUNDING.md)
