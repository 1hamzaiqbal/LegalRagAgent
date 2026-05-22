# Snap-HyRE Positive Example Handoff - 2026-05-20

Purpose: one concrete example the paper agent can use to explain why
Snap-HyRE can improve legal RAG, plus the current best read on whether the
passage-style exemplar is worth pursuing.

Use this as an explanatory example. For final numeric claims, cite
`docs/signoff_log.md` and the exact detail logs listed below.

## Bottom Line On Exemplar

The exemplar variant looks worth continuing as an analysis/probe variant, not
yet as the canonical method.

Retrieval-side q500/full probe, Gemma 26B:

| Benchmark | Canonical Snap-HyRE Hit@5 | Exemplar Snap-HyRE Hit@5 | Exemplar read |
|---|---:|---:|---|
| BarExamQA q500 | 0.1300 | 0.1360 | Small lift; canonical Snap-HyRE already beats raw strongly. |
| HousingQA q500, unfiltered | 0.0740 | 0.0840 | Small lift; needs state-filtered follow-up before main-matrix use. |
| Legal-Link-EU q500 | 0.6820 | 0.7580 | Meaningful recovery of anchor loss, but still below raw 0.9000. |
| MASLegalBench full proxy | 0.3531 | 0.4257 | Meaningful recovery of same-source proxy, but still below raw 0.7261. |

Answer-side early BarExamQA q55 prefix, Gemma 26B:

| Method | Correct on same 55 labels |
|---|---:|
| `rag_simple` | 46/55 = 83.6% |
| `rag_hyde` | 48/55 = 87.3% |
| canonical `snap_hyre` | 49/55 = 89.1% |
| `rag_rewrite` | 48/55 = 87.3% |
| `snap_hyre_exemplar` | 51/55 = 92.7% |

This is encouraging but not final. The q55 run stopped because the old
no-silent artifact detector falsely treated ordinary prose like "it is a fair
representation" as answer-letter leakage. The detector has been tightened in
`eval/eval_harness.py` and `scripts/analyze_detail_flags.py`, and the tail run
has been relaunched as `exemplar_answer_gemma26_priority_resume_20260520_154427`
from BarExam row 54, then Legal-Link-EU q500, then MASLegalBench full.

Conservative interpretation:

> The exemplar seems to help most where canonical Snap-HyRE loses corpus/style
> anchors, especially Legal-Link-EU and MASLegalBench. On BarExamQA it gives a
> smaller lift because canonical Snap-HyRE already converts fact patterns into
> doctrine-shaped retrieval queries. It is promising enough to continue, but it
> should stay starred/probe-only until the resumed answer-side run completes.

## Clean Canonical Snap-HyRE Win

Recommended example: BarExamQA `qa_TORTS_mbe_1175`, Gemma 26B.

Source logs:

- Raw RAG:
  `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`
- HyDE:
  `logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl`
- Snap-HyRE:
  `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- Rewrite:
  `logs/eval_rag_rewrite_or-gemma4-26b_20260517_124940_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_rewrite-nfull-k5_detail.jsonl`

### What The Question Tests

The question is a torts proximate-cause fact pattern. Driver negligently breaks
Pedestrian's leg. Later, while using crutches because of that injury,
Pedestrian slips on a banana peel in a market and breaks her arm. The correct
answer is that Driver is liable for both injuries.

The key legal issue is not generic "banana peel slip and fall"; it is whether
the original tortfeasor remains liable for subsequent injuries caused by the
plaintiff's impaired condition.

### What Raw RAG Did

Raw RAG answered `D`, which is wrong.

Retrieval:

- `gold_retrieved=false`
- Top retrieved ids:
  `caselaw_12689211_29`, `caselaw_12529134_30`,
  `caselaw_12660293_22`, `caselaw_12521447_33`,
  `caselaw_12574460_62`
- The retrieved material was mostly generic comparative-fault, self-defense,
  slip-and-fall, and intervening-cause case law.

Why it failed: the raw fact pattern embedded like ordinary accident facts, not
like the compact black-letter rule needed to answer the MBE-style question.

### What Snap-HyRE Did

Snap-HyRE answered `B`, which is correct.

Snap-HyRE generated a doctrine-shaped retrieval passage about proximate cause:

> A defendant is liable for damages proximately caused by negligence, including
> foreseeable subsequent injury caused by an intervening negligent act.

Retrieval:

- `gold_retrieved=true`
- Top retrieved ids:
  `mbe_1598`, `mbe_895`, `mbe_2811`, `mbe_1628`,
  `caselaw_12573435_76`
- The retrieved passages include the exact rule family: negligence,
  cause-in-fact, proximate cause, intervening forces, and subsequent injuries
  following an impaired condition.

The final answerer was not shown the snap answer letter. It saw only the
original question and the retrieved passages. This matters because the win is
not answer leakage; the snap stage improved retrieval shape.

### Why This Is A Legitimate Example

This row is especially clean because the controls also show the same failure
mode:

| Method | Prediction | Correct? | Gold retrieved? |
|---|---:|---:|---:|
| `rag_simple` | D | no | no |
| `rag_hyde` | D | no | no |
| `rag_rewrite` | D | no | no |
| `snap_hyre` | B | yes | yes |

The example matches the core paper claim:

> Snap-HyRE helps when the original question is a poor lexical retrieval query.
> The snap stage identifies the doctrinal frame first, and retrieval then
> surfaces rule passages that the raw fact pattern misses.

## Suggested Paper Use

Short version:

> In one BarExamQA proximate-cause row, raw RAG, HyDE, and rewrite all missed
> the answer and failed to retrieve the gold rule passage. Snap-HyRE converted
> the fact pattern into a rule-shaped query about foreseeable subsequent injury
> and proximate cause, retrieved MBE proximate-cause passages including the gold
> evidence, and the final answerer selected the correct option.

Longer analysis version:

> The row illustrates the intended mechanism. The raw question is dominated by
> concrete accident details: driver, pedestrian, crutches, banana peel, and
> supermarket. Dense retrieval follows those surface facts into generic accident
> and intervening-cause case law. Snap-HyRE first reasons to the legal issue:
> whether an initial tortfeasor is liable for later injuries made foreseeable by
> the plaintiff's impaired physical condition. Retrieval from that generated
> passage surfaces compact MBE rule passages, and final accuracy improves.

## Open Follow-Up

- Let `exemplar_answer_gemma26_priority_resume_20260520_154427` finish.
- If BarExam q500 exemplar remains above canonical Snap-HyRE and Legal-Link/MAS
  answer rows recover toward raw parity, promote exemplar to a clearly labeled
  analysis variant.
- Do not fold exemplar into the canonical method unless the resumed answer-side
  run remains clean and improves or preserves downstream accuracy.
