# SCOPE — Research Ideas (spitball log)

Started 2026-05-25 from an advisor spitball session. Forward-looking ideas,
not committed results. Numbers cited are from the v8 main answer matrix
(`paper/submission/tables/current_answer_matrix.tex`) and are descriptive
context, not claims. Keep this doc honest about what's speculative.

---

## 0. Where the bottleneck actually is (read this first)

The gold-evidence oracle gaps decide which ideas pay off on which benchmark:

- **BarExamQA (weak-query regime).** Raw RAG often *hurts* vs LLM-only
  (54.5<57.3 at 8B, 78.0<80.8 at 26B, 74.6<78.7 at 70B): bad retrieval injects
  distractors. SCOPE wins by retrieving cleaner passages so RAG stops hurting
  (it even beats the gold oracle at 26B: 82.0 vs 78.6). Headroom to gold is
  small at 26B/70B, larger at 8B. → SCOPE's value is *avoiding distractor
  injection*, not climbing to a high oracle ceiling.
- **HousingQA (strong-query regime).** Retrieval *helps a lot* (raw 62.1 vs
  LLM 44.8 at 70B), and **gold is meaningfully above raw** (67.3 vs 62.1,
  +5.2 at 70B). There is real retrieval headroom — but SCOPE *underperforms
  raw* here (59.6 < 62.1) because the raw question is already a good query and
  SCOPE's generated passage drifts.

**Implication:** HousingQA has retrieval headroom that *neither* raw nor SCOPE
captures. That gap is the target for the ideas below. But note the HousingQA
oracle ceiling is only ~+5pp over raw, so part of HousingQA's difficulty is
*answer conversion* (yes/no statutory entailment), not retrieval — better
retrieval has a bounded ceiling there.

---

## 1. `parallel3`: diverse multi-query generation + LLM evidence judge

Pipeline:
1. System prompt carries a small fixed set of held-out corpus passages
   (k examples; their associated questions excluded from the test set — no
   leakage). Extends the N=500 single-exemplar result.
2. Generate **N orthogonal hypothetical passages** (different doctrinal angles),
   conditioned on the question + examples. Precedent: `multi_hyde_diverse` was a
   real winner in this codebase's MuSiQue history.
3. Retrieve with each → pool candidate evidence.
4. **An LLM call selects the evidence** (given question + pooled candidates),
   replacing the cross-encoder reranker.
5. Final answer from the selected evidence + question, with the snap draft kept
   private (same as current SCOPE).

**Key improvement over the raw proposal:** include the **raw question itself**
as one of the retrieval queries in the diverse set (raw + snap-conditioned + a
different angle). On strong-query benchmarks raw is best; on weak-query,
generated is best. Letting the judge pick evidence regardless of which query
found it is how you'd capture the HousingQA gold headroom that pure-SCOPE
misses. This directly attacks Section 0's gap.

> **Validated 2026-05-25 (read-only, full N)** — see
> `docs/generated/raw_scope_pooling_ce_separability_2026-05-25.md`:
> - **raw ∪ SCOPE pooling is complementary on HousingQA, not BarExamQA.**
>   HousingQA union Hit@5 gain over the best single method: +10.4pp (8B),
>   **+13.8pp (Gemma 26B: 38.1→51.9)**, +7.4pp (70B). BarExamQA: ~+1pp (raw
>   retrieval is near-useless there, so the union ≈ SCOPE — "candidate-
>   generation limited").
> - **The cross-encoder buries gold.** It *does* separate gold from non-gold
>   CE scores when gold is in the candidate set (esp. HousingQA), but it does
>   **not rank gold first**: best-gold median rank 4-5, and gold sits below
>   rank 5 in ~35-40% of gold-hit rows. So gold is present but not surfaced —
>   quantitative support for replacing the CE selector with an LLM judge.
> - Net: **pooling creates the recall headroom; the CE fails to capture it.**
>   Post-hoc downstream-accuracy test of union → {CE-rerank, RRF, LLM-judge} →
>   answer is queued as the next results-lane experiment.

**Open knobs to test:**
- k examples: 1 vs 3 vs 5 (diminishing returns likely; examples eat context).
- N generated passages: 2 vs 3 vs 4.
- Judge input: evidence-only vs evidence+per-passage justification (listwise
  RankGPT-style). Justification may help selection but costs tokens.
- Asymmetric models: weak generator + **strong judge** vs strong generator +
  weak judge vs all-strong. Given the retrieval→oracle gap, selection may be
  where a smart model has the most leverage, and it's only 1 of N calls.

**Cautions:**
- This changes ~3 things at once (examples, diversity, judge). For the paper,
  ablate each against current SCOPE or reviewers will ask which one matters.
- Cost: ~5 LLM calls vs SCOPE's 2. The current defensible headline is "SCOPE is
  the token-efficient method." Frame `parallel3` as a *different point on the
  cost–accuracy frontier* (high-accuracy regime), not a replacement — keep both.

---

## 2. LLM-as-evidence-judge vs cross-encoder

The current cross-encoder (`ms-marco-MiniLM-L-6-v2`) is suspect as the evidence
selector because it is (a) **general-domain** (web relevance, not legal
sufficiency) and (b) **pointwise** — it scores passages independently and can't
reason about whether a *set* of passages collectively resolves the question, or
dedupe redundant passages, or recognize that passage A + passage B together
answer it. An LLM judge does listwise/set-level reasoning.

**Cheaper baseline to beat first:** Reciprocal Rank Fusion (RRF) over the N
ranked lists from the diverse queries. RRF is near-free and often beats
single-query retrieval. If RRF captures most of the gain, the expensive LLM
judge may be unnecessary; if the judge beats RRF, *that delta* is the clean
contribution. Order: RRF first, then LLM judge measured as the delta over RRF.

---

## 3. Cheap dedicated SCOPE generator via retrieval-reward distillation

(Reframe of the "GAN-style generator" idea.) Goal is sound: a small, cheap
model that's elite at producing retrievable SCOPE passages. The GAN *mechanism*
is probably the wrong tool — adversarial training is unstable and "look like the
corpus" (what a discriminator rewards) is misaligned with the real objective,
"retrieve the *specific* gold passage for this question." Risk: a degenerate
generator that's corpus-flavored but non-specific (high similarity, low
retrieval precision).

**Cleaner mechanism — rejection-sampling distillation:**
1. Use the big model to generate N SCOPE passages per training question.
2. Keep the ones that actually retrieve the gold passage @k (retrieval reward).
3. Fine-tune a small model on the winners.

Stable, no discriminator, directly optimizes retrieval success. In the spirit of
Promptagator / InPars / Doc2Query, inverted. The kernel of truth in the GAN
intuition (corpus-realistic *style* helps embeddings land in the right region —
consistent with the exemplar result) comes for free: passages that retrieved
gold are corpus-shaped by construction. Caveat: needs gold-labeled retrieval
data per benchmark and careful train/test separation.

---

## 4. Adaptive planning via snap-disagreement stopping

(Reframe of "external advisor judges when to do another round.") The instinct to
separate the *controller* from the *executor* is right — that's why the earlier
baked-in adaptive controller underperformed. But the bar is high: an adaptive
stopper must beat a fixed-budget baseline at equal-or-lower cost, in crowded
territory (Self-RAG, FLARE, CRAG).

**A cheap, principled signal you already generate and currently discard — the
snap draft answer (the parametric prior):**
- snap draft **==** evidence-grounded answer → confident → stop.
- snap draft **!=** evidence-grounded answer → prior and evidence disagree →
  uncertainty → trigger another round (new doctrinal angle) or escalate to a
  stronger model.

This unifies the adaptive-planning idea with SCOPE: an agreement check grounded
in the architecture, ~free, with a clean ablation (fixed-budget vs
disagreement-triggered) and a tidy narrative — "the discarded draft isn't
wasted, it's a calibration signal."

---

## 5. Diagnostic to collect: cross-encoder score distributions by query type

Motivation: directly test whether SCOPE/HyDE queries retrieve "more confident"
(higher CE) passages than raw, and — more importantly — whether the CE score is
even a good *judge* of gold relevance. If gold passages don't have
systematically higher CE than non-gold, that's quantitative support for
replacing the CE with an LLM judge (Section 2).

**Data already exists** — no new runs. Each retrieval-cache row
(`caches/retrieval/**/*.jsonl`) has: `scores` (per-passage CE scores, top-k),
`query_type` (raw_question / rag_hyde / snap_hyre), `dataset`, `retrieved_ids`,
`gold_ids`, `gold_retrieved`.

Compute, per (dataset × query_type):
- Distribution of CE scores of retrieved passages (mean / median / IQR / tail).
- **CE score of gold passages when retrieved vs non-gold passages** — the key
  separability test. If gold ≈ non-gold in CE, the CE is a weak judge.
- Rank position of the gold passage within the CE-sorted list (does the CE put
  gold at the top, or does gold get buried below distractors?).
- Cross-dataset contrast: weak-query (BarExamQA) vs strong-query (HousingQA).

Hypotheses worth stating: (i) SCOPE/HyDE retrievals have higher absolute CE than
raw on BarExamQA (vocabulary alignment), (ii) but CE poorly separates gold from
non-gold within legal corpora, motivating the LLM judge. This is a results-lane
analysis (read-only over existing caches).

---

## 6. Suggested experiment ordering (cheapest / most diagnostic first)

1. **Confirm the bottleneck split** per benchmark from the gold gaps (Section 0)
   — essentially free, decides everything downstream.
2. **CE score distribution analysis** (Section 5) — read-only, motivates the
   judge.
3. **Multi-query + RRF, raw question included** (Sections 1–2) — cheap, no LLM
   judge; targets the HousingQA gap.
4. **Add the LLM evidence judge** on top of #3 — measure the delta over RRF.
5. **Asymmetric model ablation** on #4 — is the strong-judge call the leverage
   point?
6. Longer horizon: **retrieval-reward distillation** for the cheap generator
   (Section 3) and **snap-disagreement stopping** (Section 4).

---

## Honest caveats to carry into advisor discussions

- HousingQA's gold ceiling is only ~+5pp over raw, so retrieval/selection ideas
  have a bounded payoff there; a chunk of its difficulty is answer conversion.
  Expect `parallel3`-style ideas to shine on BarExamQA-style weak-query tasks
  and help-but-not-dominate on HousingQA.
- Every multi-call idea trades away the token-efficiency headline. Keep SCOPE as
  the efficient operating point and position new methods as a separate
  high-accuracy regime.
- The exemplar / few-shot-in-prompt ideas must keep the example questions out of
  the test set (already noted) and prefer *fixed* examples over per-question
  retrieved ones to avoid leakage and extra cost.
