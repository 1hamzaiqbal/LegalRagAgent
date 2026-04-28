# Cross-method audit vs golden_passage — Gemma 4 26B-A4B BarExam Tier 3 N=1195

Meeting 2026-04-27 follow-up: is golden_passage uniquely worse than expected, or 
do other methods also flip on the same questions? This pairs each of 7 Tier 3 
methods against `golden_passage` and reports paired transitions plus anchoring rate.

## TL;DR — the paradox is general, and snap-first is the actual mechanism

1. **The "golden_passage paradox" is not a quirk of llm_only.** Three methods beat
   golden_passage on the same paired N=1195: `rag_snap_hyde` (+2.51pp), `snap_only_in_final`
   (+1.92pp), and `llm_only` (+1.09pp). All three are **snap-first architectures** — they let
   the model commit to a prior before (or instead of) consuming retrieved context. The two
   methods that lose to golden_passage materially are `subagent_hybrid` (-4.52pp) and
   `subagent_rag` (-0.50pp), which are cheap-LLM-mediated retrieval pipelines.

2. **Anchoring rate is uniformly high (63-77%) across all 7 methods.** When any method AND
   golden_passage both get a question wrong, they share the *same* wrong answer 63-77% of
   the time — far above the 25% chance level for a 4-option MC. This means BarExam questions
   carry strong distractor lures that mislead the model the same way regardless of whether
   it sees gold context or not. `rag_snap_hyde` anchors most (76.6%), confirming snap+HyDE
   reasons very similarly to gold-injected reasoning.

3. **Symmetric flip is universal.** Every method flips ~similar numbers of answers in both
   directions vs golden_passage; gold-passage injection roughly behaves like adding ~10%
   noisy evidence whose net direction depends on the base method's prior strength.

4. **Mechanism decomposition for the +3.10pp `rag_snap_hyde` lift over `rag_simple`:**
   - `snap_only_in_final` already beats `rag_simple` by ~+2.51pp (snap reasoning alone).
   - `rag_snap_hyde` adds another ~+0.6pp on top via HyDE retrieval.
   - **Snap reasoning does ~80% of the BarExam work**, retrieval adds the marginal lift.
   - Mirrors the MuSiQue mechanism finding: `multi_hyde_diverse`'s lift = ~80% from
     HyDE-style answer passages, ~20% from query diversity — both datasets show that the
     cheap "shape the LLM's reasoning prior" trick dominates the headline.

### Paper-language implications

- **Don't call `golden_passage` an oracle ceiling.** It is a "single gold-passage control"
  that flips 10-20% of answers in both directions vs no-context, with the net direction
  depending on the model's prior strength on the question.
- **Frame the headline as snap-first architecture**, not "RAG helps legal MC". The contribution
  is "letting the model anchor on its prior before reconciling with retrieved evidence".
- `subagent_hybrid` -4.52pp vs golden_passage confirms the gap-routing/cheap-LLM pipeline
  weakness already flagged for `subagent_rag` on MuSiQue (-12.0pp). Consistent failure mode.

Common idx across all 8 modes: **1195**

## Per-method transitions vs golden_passage

Methods sorted by Δ (method - golden_passage). Positive Δ = method beats golden on this paired N.

| Method | N paired | golden EM | method EM | Δ | golden-only right | method-only right |
|---|---:|---:|---:|---:|---:|---:|
| `rag_snap_hyde` | 1195 | 78.66% | 81.17% | +2.51pp | 80 | 110 |
| `snap_only_in_final` | 1195 | 78.66% | 80.59% | +1.92pp | 74 | 97 |
| `llm_only` | 1195 | 78.66% | 79.75% | +1.09pp | 83 | 96 |
| `rag_hyde` | 1195 | 78.66% | 78.91% | +0.25pp | 103 | 106 |
| `subagent_rag` | 1195 | 78.66% | 78.16% | -0.50pp | 125 | 119 |
| `rag_simple` | 1195 | 78.66% | 78.08% | -0.59pp | 99 | 92 |
| `subagent_hybrid` | 1195 | 78.66% | 74.14% | -4.52pp | 175 | 121 |

## Anchoring rate: when method AND golden are both wrong, how often do they share the same wrong pred?

Higher anchor rate = method's failures resemble the gold-passage-induced failures (suggests both 
are misled by the same surface signal in the question, not just random noise).

| Method | both wrong | same wrong pred | different wrong pred | anchor rate |
|---|---:|---:|---:|---:|
| `rag_snap_hyde` | 145 | 111 | 34 | 76.6% |
| `snap_only_in_final` | 158 | 120 | 38 | 75.9% |
| `llm_only` | 159 | 117 | 42 | 73.6% |
| `rag_hyde` | 149 | 109 | 40 | 73.2% |
| `subagent_rag` | 136 | 88 | 48 | 64.7% |
| `rag_simple` | 163 | 121 | 42 | 74.2% |
| `subagent_hybrid` | 134 | 85 | 49 | 63.4% |

## Symmetric-flip check: does every method flip ~equally with golden?

If golden_passage is *uniquely* misleading vs llm_only, it should over-flip llm_only-correct 
answers (high golden-only-right - method-only-right gap). If the flip is symmetric across ALL 
methods, golden is not uniquely bad — it's just one more noisy oracle.

| Method | flips that hurt method (golden right, method wrong) | flips that hurt golden (method right, golden wrong) | net (helps method) |
|---|---:|---:|---:|
| `rag_snap_hyde` | 80 | 110 | +30 |
| `snap_only_in_final` | 74 | 97 | +23 |
| `llm_only` | 83 | 96 | +13 |
| `rag_hyde` | 103 | 106 | +3 |
| `subagent_rag` | 125 | 119 | -6 |
| `rag_simple` | 99 | 92 | -7 |
| `subagent_hybrid` | 175 | 121 | -54 |

## Reading guide

- A **negative** Δ means the method does WORSE than golden_passage on the paired N (these are weak methods).

- A **positive** Δ means the method beats golden_passage. The size of `method-only right` shows how 
often the method recovers from the gold-passage anchor.

- Anchoring rate of ~25% would be chance (4 MC options); rates well above 25% mean the gold passage 
steers BOTH methods toward the same wrong distractor — i.e., the question contains a real lure.
