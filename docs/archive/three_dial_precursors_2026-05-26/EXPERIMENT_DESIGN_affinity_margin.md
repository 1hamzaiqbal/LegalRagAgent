# Pre-Registration — Affinity-Margin Account of Generative Query Expansion (2026-05-26)

Anchor (locked): *"An affinity-margin account of when generative query expansion
helps — and why its retrieval gains don't convert to answers."* Pillar 1 =
retrieval mechanism; Pillar 2 = answer-conversion gap. Snap/HyDE/discard-vs-keep
are mechanism probes, not method claims. Routing = principled negative.

## 1. Model
Query rep `x` scores corpus passages by affinity `aff(x,·)` (cross-encoder CE or
bi-encoder cosine). For gold `g` and the retrieved distractor set `R\{g}`:

  **gold margin**  `M(x) = aff(x, g) − max_{d ∈ R, d≠g} aff(x, d)`

Gold is retrieved (top-k) roughly iff `M > 0`. Raw query margin `M_raw = M(q)`;
expanded (HyDE/Query2Doc/SCOPE) margin `M_exp = M(s)`; `ΔM = M_exp − M_raw`.
NB: this is strictly more than our prior CE-delta `CE(s,g) − CE(q,g)`, which is
only the **gold-affinity** term and ignores the **distractor** term — testing
whether the full margin out-predicts the gold-affinity-only delta is itself novel.

## 2. Hypotheses (pre-stated; each with a kill criterion)
- **P1 (crossover).** Expansion's retrieval benefit is positive when `M_raw` is
  low/negative (headroom) and ≤0 when `M_raw` is already high. *Kill:* benefit is
  monotone-positive across `M_raw`, or no crossover.
- **P2 (monotone gain).** Per-query retrieval gain ∝ `ΔM`, and the **full margin**
  predicts better than gold-affinity-only. *Kill:* |ρ(ΔM, gain)| < 0.2, or full
  margin ≤ gold-affinity-only.
- **P3 (margin dominates confounds).** In a joint model of retrieval gain on
  {`M_raw`, `ΔM`, domain, answer-format, query-length, OOV, perplexity}, margin
  terms dominate; the help→hurt crossover holds **within each dataset**, not only
  across. *Kill:* domain/format/length explain gain as well as margin once both are in.
- **P4 (falsify the hallucination/knowledge-gap cause).** CSQE'24/Weller attribute
  expansion failure to LLM ignorance. We predict failures (`ΔM<0`) cluster on
  **high `M_raw`** (strong query) or low `aff(s,g)` (drift), NOT on high OOV/perplexity.
  *Kill:* OOV/perplexity explain failures as well as `M_raw`.
- **P5 (answer-conversion decoupling).** Even when expansion repairs retrieval,
  answer gain is weakly coupled to retrieval gain. Decompose
  `answer-gain = P(gold enters top-k) × P(model converts | gold present)` and show
  the **conversion** factor is the bottleneck, modulated by distractor mass
  (Power-of-Noise), gold position (Lost-in-the-Middle), and answer-format
  (binary entailment worst). *Kill:* answer-gain tracks retrieval-gain tightly.

## 3. Datasets
- **Gold-labeled (margin computable):** BarExamQA, HousingQA (state-filtered).
  *Add 2–3 standard BEIR sets with qrels* for (a) cross-domain mechanism validation,
  (b) direct comparability to HyDE/Query2Doc/Weller's testbed: candidates SciFact,
  FiQA (finance), NFCorpus (medical), TREC-COVID. Without a non-legal gold-labeled
  set, the mechanism is validated only on legal — BEIR fixes that and connects to
  the literature.
- **Answer-effect only (no gold):** MedQA-USMLE → downstream EM generalization, not
  the margin mechanism (no gold passages). State this limitation explicitly.

## 4. Conditions (query representations)
`raw` · `HyDE` (canonical keep-query, Eq.8) · `HyDE-discard` · `Query2Doc`
(keep+concat) · `SCOPE` (discard). The **keep-vs-discard** contrast directly tests
the margin lever and confronts Query2Doc Table 4 / GAR ("answer-only retrieves
false positives") on their own terms.

## 5. Measurements (per query × condition)
aff-to-gold (CE + cos), max aff-to-distractor, margin `M`, gold rank, Hit@k;
downstream EM. All vs raw → `ΔM`, retrieval-gain, answer-gain. Distractor set =
top-k under the **same retriever** as the condition. Multi-gold (Housing) → max
over the gold set.

## 6. Analysis & reporting instruments
- P2: Spearman/Kendall(`ΔM`, gain), per dataset + pooled; full-margin vs gold-only.
- P1: bin by `M_raw` quantiles → crossover curve.
- P3: within-dataset margin stratification; joint regression w/ standardized coeffs
  / partial-R²; margin terms dominate.
- P4: logistic regression of `1[ΔM<0]` on knowledge signals vs `M_raw`.
- P5: the conversion decomposition above, stratified by distractor mass / position / format.
- **Robustness Index** `RI = (n+ − n−)/N` and **risk-reward curves**
  (Collins-Thompson'09) as the headline per-query reporting; McNemar + bootstrap
  CIs; full N throughout.

## 7. Reuse vs new runs
- **Now, on existing caches (no model calls):** BarExam + Housing raw/HyDE/SCOPE
  retrieval caches + gold ids → P1–P4 confirmatory pass immediately.
- **New (use the Q2 OpenRouter runner):** BEIR sets (generate HyDE/Query2Doc/SCOPE
  + retrieve); `HyDE-discard` variant; `Query2Doc` condition; P5 answer
  decomposition (some answer runs). Model choice = refreshed triad (Q3), once probed.

## 8. Positioning (must out-design / distinguish)
Out-design **Weller'24** (2309.08541): per-query + label-free + geometric margin +
confounds controlled + 2nd domain vs their macro/label-dependent/30-example.
Distinguish **Emami'26** (2604.22661): mechanism + specialist domains +
answer-conversion analysis vs their general-web variant-selection. Falsify
**CSQE'24** hallucination cause. Adopt Collins-Thompson risk language.

## 9. Known limits
Margin needs gold (MedQA = effect only). Cross-domain mechanism rests on BEIR +
legal. Retriever-strength variation (Weller's axis) is an optional extension, not
core.
