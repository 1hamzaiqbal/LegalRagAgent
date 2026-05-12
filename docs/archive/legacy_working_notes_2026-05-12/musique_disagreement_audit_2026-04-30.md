# MuSiQue Llama 70B Disagreement Audit - 2026-04-30

Join key: `idx`. Common paired rows: 200.

## Source Logs

- `rag`: `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`
- `two_call`: `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl`
- `mhd`: `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`
- `iter_ptable`: `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl`

## Run Summary

| Run | Accuracy | Gold retrieved | Gold-hit count histogram | Avg LLM calls | Avg input toks | Avg output toks |
|---|---:|---:|---|---:|---:|---:|
| `rag` | 55/200 (27.5%) | 168/200 (84.0%) | 0:32, 1:99, 2:58, 3:10, 4:1 | 1.00 | 833 | 84 |
| `two_call` | 74/200 (37.0%) | 173/200 (86.5%) | 0:27, 1:84, 2:78, 3:11 | 2.00 | 1135 | 249 |
| `mhd` | 71/200 (35.5%) | 168/200 (84.0%) | 0:32, 1:82, 2:79, 3:6, 4:1 | 2.00 | 1033 | 417 |
| `iter_ptable` | 72/200 (36.0%) | 184/200 (92.0%) | 0:16, 1:38, 2:103, 3:42, 4:1 | 6.76 | 2507 | 422 |

## Pairwise vs Baseline

| Treatment | Rescued | Harmed | Net | Rescued gold delta | Harmed gold delta | Answer changed | Rescued subjects |
|---|---:|---:|---:|---|---|---:|---|
| `two_call` | 33 | 14 | +19 | less_gold=4, more_gold=21, same_gold=8 | less_gold=9, same_gold=5 | 47/47 | 2-hop=18, 3-hop=11, 4-hop=4 |
| `mhd` | 29 | 13 | +16 | less_gold=2, more_gold=18, same_gold=9 | less_gold=8, same_gold=5 | 42/42 | 2-hop=17, 3-hop=11, 4-hop=1 |
| `iter_ptable` | 43 | 26 | +17 | more_gold=32, same_gold=11 | less_gold=8, more_gold=9, same_gold=9 | 69/69 | 2-hop=27, 3-hop=16 |

## Complementarity

- Static best in this family: 74/200 (37.0%).
- Oracle any-correct across all listed runs: 114/200 (57.0%).
- All listed runs wrong: 86/200 (43.0%).

## Interpretation

- The MuSiQue lift is complementary, not a single stable method win: only 16
  baseline-wrong rows are rescued by all three stronger arms.
- `two_call` and `mhd` mostly rescue rows when they retrieve more gold passage
  ids than `rag` (21/33 and 18/29 rescues), which supports the
  query-formulation/retrieval-bottleneck story.
- `iter_ptable` retrieves more gold ids most aggressively (92.0% gold
  retrieved, 43 rescues), but it also harms 26 baseline-correct rows, including
  9 where it had more gold ids than baseline. That is the clearest warning
  against "more evidence/more reasoning steps" as an unconditional agentic
  claim.
- Paper angle: this supports adaptive evidence allocation and evidence-state
  quality control, not a leaderboard claim that one HyDE/planning variant is
  generally best.

### Rescue Overlap

| Set | Count | Example ids |
|---|---:|---|
| `two_call` rescues | 33 | 2hop__128608_82341, 2hop__24973_5674, 2hop__276637_126089, 2hop__3739_13529 (+29 more) |
| `two_call` & `mhd` | 22 | 2hop__24973_5674, 2hop__3739_13529, 2hop__450600_158262, 2hop__472486_97805 (+18 more) |
| `two_call` & `iter_ptable` | 20 | 2hop__128608_82341, 2hop__3739_13529, 2hop__374495_68633, 2hop__450600_158262 (+16 more) |
| `mhd` rescues | 29 | 2hop__115515_779396, 2hop__13592_49388, 2hop__24973_5674, 2hop__25344_88628 (+25 more) |
| `mhd` & `iter_ptable` | 20 | 2hop__115515_779396, 2hop__3739_13529, 2hop__450600_158262, 2hop__472486_97805 (+16 more) |
| `iter_ptable` rescues | 43 | 2hop__10122_18974, 2hop__115515_779396, 2hop__128608_82341, 2hop__135993_160249 (+39 more) |
| `two_call` & `mhd` & `iter_ptable` | 16 | 2hop__3739_13529, 2hop__450600_158262, 2hop__472486_97805, 2hop__501624_181960 (+12 more) |

## Correctness Patterns

Pattern bit order: `rag`, `two_call`, `mhd`, `iter_ptable`.

| Pattern | Count | Example ids |
|---|---:|---|
| `0000` | 86 | 2hop__10620_79092, 2hop__121145_561444, 2hop__123148_5385, 2hop__129721_40482 (+82 more) |
| `1111` | 25 | 2hop__142699_67465, 2hop__335899_124498, 2hop__35136_35117, 2hop__35173_17335 (+21 more) |
| `0001` | 19 | 2hop__10122_18974, 2hop__135993_160249, 2hop__145282_21711, 2hop__16844_42173 (+15 more) |
| `0111` | 16 | 2hop__3739_13529, 2hop__450600_158262, 2hop__472486_97805, 2hop__501624_181960 (+12 more) |
| `1110` | 14 | 2hop__146919_29244, 2hop__25788_990, 2hop__82858_654855, 3hop1__178435_547811_41132 (+10 more) |
| `1000` | 8 | 2hop__36747_18974, 2hop__512773_346751, 2hop__835710_7298, 3hop1__222497_309482_41132 (+4 more) |
| `0100` | 7 | 2hop__276637_126089, 2hop__403060_92763, 2hop__86689_728109, 3hop2__132957_133292_40768 (+3 more) |
| `0110` | 6 | 2hop__24973_5674, 2hop__581084_828685, 3hop1__302383_503371_21711, 3hop1__491648_339990_15538 (+2 more) |
| `0011` | 4 | 2hop__115515_779396, 2hop__567588_67465, 2hop__91021_19033, 3hop1__226638_538202_84283 |
| `0101` | 4 | 2hop__128608_82341, 2hop__374495_68633, 2hop__511296_2684, 2hop__96194_78303 |
| `0010` | 3 | 2hop__13592_49388, 2hop__25344_88628, 2hop__632870_22402 |
| `1001` | 3 | 2hop__448615_127908, 3hop1__24980_42197_18397, 4hop3__39836_29339_508306_70744 |
| `1010` | 2 | 2hop__10017_18974, 2hop__91248_84207 |
| `1100` | 2 | 2hop__142443_768138, 2hop__703767_37168 |
| `1011` | 1 | 4hop3__373866_5189_38229_86687 |

## Mechanism Read

- Improvements are not pure answer-format noise: every baseline/treatment flip changed the normalized answer string in this audit.
- `more_gold` in rescued rows means the treatment retrieved more gold passage ids than baseline; `same_gold` means the gain came despite equivalent gold-id count.
- Treat this as a mechanism screen, not final causality: exact-match scoring and multi-hop aliasing can still hide semantically acceptable answers.
