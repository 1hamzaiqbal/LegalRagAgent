---
title: OPD × SKILL0 Experiment Design — the ladder, the models, the H100 budget
type: design
tags: [opd, skill0, distillation, h100, eit, experiment-design]
created: 2026-07-02
date: 2026-07-02
status: active design — E1 launched, E0 done, infra scaffolding in progress
---

# OPD × SKILL0: experiment design for 1–2 H100s

Companion to [[skill-distillation-bridge]] (the *why*); this page is the
*how*: a five-rung experiment ladder with explicit model choices, VRAM
budgets, decision gates, and what each rung falsifies. Designed so every
rung below E3 runs on the free A100 lane today, and E3/E4 drop onto 1–2
H100s without redesign.

## The question the ladder answers
Can the allocation/effort/arbitration *skills* of agentic retrieval be put
into a small model's weights — and is a big teacher (or a skill file, or
dense teacher logprobs) the ingredient that gets them there? Each rung
isolates one ingredient.

## The ladder

| Rung | Question | Method | Compute | Status |
|---|---|---|---|---|
| **E0** | Is there per-question allocation headroom, and can cheap externals capture it? | Offline bandit replay | Mac, $0 | **DONE — negative/instructive**: headroom 8–24pp, unreachable from features ([[offline-bandit-v0]]) |
| **E1** | Can a 9B *internalize* the allocation predictor from sparse outcome labels? | (question, reader, strategy)→Yes/No LoRA (judge recipe), argmax(score−λ·cost) policy | 1×A100, $0 | **LAUNCHED** — EIT job 93770, 6,136 train pairs, rung-1-identical splits |
| **E2** | Is there a *skill gap* to distill — does a big model with the skill file in context allocate/search better than without? | Inference-only A/B: teacher ± skill markdown on the same test questions | API or 1×A100/H100 inference | designed, not launched |
| **E3** | Does **dense teacher signal (OPD)** beat sparse outcome labels (E1) for internalization? | Student samples on-policy; teacher (with skill context) scores per-token logprobs; reverse-KL update | 1–2×H100 | infra scaffolding now (`scripts/opd/`) |
| **E4** | Does it hold for **multi-turn** search (retrieve→read→re-query→stop) with curriculum withdrawal? | SKILL0-style ICRL ± OPD hybrid on a Search-R1-style env over our corpora | 2×H100 | design only |

**Decision gates.** E2 gates E3: if the teacher shows *no* skill-context
gap, there is nothing to distill beyond E1's labels — stop and publish the
allocation study. E1-vs-E3 is the paper's central contrast: sparse
outcome labels vs dense on-policy teacher tokens for the *same* target
behavior. E4 only proceeds if E3 beats E1 by a real margin (≥3pp policy
accuracy or equal accuracy at ≤half the tokens).

## Model matrix (same-tokenizer constraint is hard)

OPD aligns per-token logprobs → teacher and student MUST share a tokenizer.
Two family tracks:

| Track | Student(s) | Teacher (1×H100, 80GB) | Teacher (2×H100, 160GB) |
|---|---|---|---|
| **Qwen3** (primary — our judge recipe is Qwen) | Qwen3-1.7B (tiny), Qwen3-4B, Qwen3.5-9B (bridge to judge line) | Qwen3-32B bf16 (~65GB, comfortable) | **Qwen3-235B-A22B FP8** (~120GB, TP=2) — the same scale as our prompted-235B baseline, closing a beautiful loop: prompted-235B = 15.3% on the judge task; can its *distilled* student beat it? |
| **Llama** (secondary — matches our reader models) | Llama-3.2-1B/3B, Llama-3.1-8B | Llama-3.3-70B **FP8** (~70GB, tight, low ctx) | Llama-3.3-70B bf16 TP=2 (~140GB, comfortable) — the teacher IS our 70B reader, so "distill the reader's self-knowledge" experiments become possible |
| Smoke (free, today) | Qwen3-1.7B | Qwen3-8B on **A100** | — |

Student training footprint: LoRA r=32 on ≤9B ≈ 20–25GB with grad-ckpt —
coexists on the second H100 next to an FP8 70B only if ctx is short; the
clean layout is **teacher = vLLM server on GPU(s) 0(+1), student trainer on
its own GPU** — hence "1 or even 2 H100s simultaneously" maps to:
1×H100 = 32B-teacher track; 2×H100 = 32B teacher + dedicated student GPU
(fastest iteration), or 235B/70B teacher TP=2 with student squeezed or on
an A100 alongside.

Note on EIT reality: `general-gpu` currently shows one h100 node
(`--gpus h100:1`) and 3 healthy a100-sxm4 nodes × 4 GPUs. If the A100-SXM4s
are 80GB (job 93770 prints `nvidia-smi` to settle this), a **single a100s
node already gives 4×80GB = 2-H100-class capacity for TP teachers** — the
2-GPU designs may be runnable for free today with `--gpus a100-sxm4:2`.

## E3 in detail (the one to get right)

**Task v0**: the E1 allocation task — single-turn, cheap, already has
train/eval infrastructure and five paired cells. Student prompt = question +
reader + strategy menu; student outputs a decision (and optionally a one-line
rationale). **Teacher context = student context + the allocation skill file**
(a markdown distilling the three-dial law: when llm_only wins, when evidence
pays, cost accounting — written from [[thesis-v2]] numbers). The skill file
is the thing being internalized; the student never sees it.

**Objective** (Tinker OPD recipe, implemented in `scripts/opd/opd_loss.py`):
sample completions from the student on-policy; per-token advantage
A_t = logp_teacher(t) − logp_student(t); update = policy-gradient on
−A_t·logp_student(t) (≡ minimizing reverse KL to the teacher on the
student's own distribution), |A_t| clamped, ratio-clipped for stale samples.
Closed-teacher fallback: sequence-level KD (`kd_forward_loss`) on teacher
traces — this arm doubles as the **closed-model comparison** the meeting
asked about.

**Arms for the E3 report** (all evaluated as policies on the rung-1 test
halves, so every number is comparable back to E0/E1):
1. E1 baseline: outcome-label LoRA (sparse signal).
2. OPD from skill-augmented teacher (dense signal + skill).
3. OPD from bare teacher (dense signal, no skill) — isolates the skill file.
4. KD-on-traces from skill-augmented teacher (dense-ish, closed-teacher-compatible).
5. Zero-shot student + skill file in context (no training — SKILL0's
   "follows but doesn't acquire" baseline).

**Throughput budget** (why this fits): student 1.7B–9B sampling at
~10–40 seq/s on one GPU; teacher scoring is prompt-logprobs only (no
generation) — vLLM prefill-bound, ~50k tok/s on H100 for 32B. A 200-step run
at G=4, batch 32, 128 new tokens ≈ 3.3M student tokens to score ≈ minutes of
teacher time per run. The bottleneck is student generate(), not the teacher.

## E4 sketch (design only, 2×H100)
Search-R1-style loop over our corpora via the retrieval caches (BarExam,
Housing state-filtered; actions: retrieve(k∈{1,5,10}) / re-query / answer /
abstain), reward = correct − λ·(tokens + retrieval calls). SKILL0's
curriculum (skill files for effort control + conflict arbitration, budget
decay on measured Δ_k) ± OPD from a skill-augmented teacher. Conflict
arbitration probes: inject contradicting passages (doc-vs-doc) and
prior-contradicting gold (doc-vs-prior) — the meeting's Idea 2 becomes a
measurable sub-skill here. Requires verl-style multi-turn RL or a custom
loop; do NOT build until E3 gates pass. SkillZero repo (archived on EIT)
is the reference implementation to crib the env/loop shape from.

## Risks and open checks
- **SDAR / SKILL1 unread** — novelty gate for the whole E3/E4 story
  ([[skill0]] §why-we-care). Read before E3 launch.
- vLLM `prompt_logprobs` return-size limits on long contexts — smoke test
  covers it.
- Tokenizer drift within families (Qwen3 vs Qwen3.5 vocab — verify identical
  before pairing; smoke test asserts it).
- E1 may already saturate the learnable signal (rung-1's oracle is
  noise-inflated); if E1 ≈ best-fixed-arm and E2 shows no skill gap, the
  honest conclusion is "allocation is not predictable from the question at
  this N" — publishable inside the mechanism paper, and E3 pivots to a task
  with more headroom (multi-turn search directly).

## Links
[[skill-distillation-bridge]] · [[skill0]] · [[offline-bandit-v0]] ·
[[judge-capacity-dial]] · [[thesis-v2]] · [[08-meeting-notes]] ·
[[direction-2026-07]]
