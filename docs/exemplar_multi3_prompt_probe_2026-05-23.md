# Multi-Exemplar Passage-Style Probe - 2026-05-23

Status: exploratory probe design. This is not part of the canonical Snap-HyRE
grid unless it passes retrieval and answer-side checks.

## Goal

Test whether Snap-HyRE improves its generated retrieval passage when the first
LLM call sees three corpus-style examples instead of one. The examples are
style signals only: the final answerer still sees only retrieved passages and
the current question.

The leakage control is to exclude any question whose `gold_idx` intersects the
example document ids. Use:

```bash
EVAL_PASSAGE_STYLE_VARIANT=multi3 \
EVAL_EXCLUDE_GOLD_IDS_PATH=configs/exemplar_multi3_exclusions_2026-05-23.json
```

The same options are also accepted by `scripts/build_generation_cache.py`,
`scripts/build_retrieval_cache.py`, and `eval/eval_harness.py` as
`--passage-style-variant multi3` and `--exclude-gold-ids-path ...`.

## BarExamQA Prompt Bank

Source exclusion ids: `mbe_4`, `mbe_20`, `mbe_308`.

Source rows excluded by exact gold passage use: `qa_nan_mbe_7`,
`qa_nan_mbe_23`, `qa_nan_mbe_391`.

System-prompt style signal:

```text
A useful BarExamQA retrieval passage names the doctrine first, then states the
operative element, exception, or admissibility rule in neutral black-letter
form. It does not restate the fact pattern or argue for an answer choice.

Corpus passage excerpt 1: The res ipsa loquitur doctrine enables a jury
presented only with circumstantial evidence to infer negligence from the fact
that an event happened. The criteria include an event that ordinarily does not
occur without negligence, an agency or instrumentality within the defendant's
exclusive control, and no voluntary action or contribution by the plaintiff.

Corpus passage excerpt 2: The Fourth Amendment bars unreasonable searches and
seizures, and a warrantless search is per se unreasonable unless it falls within
a specifically established exception. One exception is valid consent, which must
be knowing and voluntary and given by a person with authority to consent.

Corpus passage excerpt 3: Specific performance is an equitable remedy ordered
when the legal remedy, usually money damages, is inadequate or impracticable.
When land is the subject matter of the agreement, the legal remedy is generally
treated as inadequate because each parcel of land is unique.
```

Feasibility read: this is low-risk for BarExamQA because the three gold passage
ids appear in only three rows total. The examples are deliberately spread across
torts, criminal procedure, and contracts/real-property doctrine so they teach
the passage shape rather than one doctrine.

## HousingQA Prompt Bank

Source exclusion ids: `1508532`, `1038490`, `1727814`.

Representative source rows: `hqa_Idaho_297`, `hqa_Montana_6968`,
`hqa_California_3428`.

Exact gold-id exclusion removes 21 HousingQA rows:
`hqa_California_3428`, `hqa_California_3429`, `hqa_California_3430`,
`hqa_California_3431`, `hqa_California_3432`, `hqa_Idaho_297`,
`hqa_Montana_6963`, `hqa_Montana_6964`, `hqa_Montana_6965`,
`hqa_Montana_6966`, `hqa_Montana_6967`, `hqa_Montana_6968`,
`hqa_Montana_6969`, `hqa_Montana_6970`, `hqa_Montana_6971`,
`hqa_Montana_6972`, `hqa_Montana_6973`, `hqa_Montana_6974`,
`hqa_Montana_6975`, `hqa_Montana_6976`, `hqa_Montana_6977`.

System-prompt style signal:

```text
A useful HousingQA retrieval passage sounds like a state statutory definition
or landlord-tenant procedure section. It should preserve the state or territory
named in the question, preserve legal terms from the question, name the actor
and authority when relevant, and avoid guessing a yes/no answer.

Corpus passage excerpt 1: In Idaho eviction proceedings, an appeal taken by the
defendant does not stay proceedings upon the judgment unless the court so
directs.

Corpus passage excerpt 2: In Montana, if the landlord purposefully or
negligently fails to supply heat, running water, hot water, electricity, gas, or
other essential services, the tenant may give written notice specifying the
breach and may procure reasonable services and deduct their actual and
reasonable cost from rent, recover damages based on diminished rental value, or
procure reasonable substitute housing during the noncompliance period.

Corpus passage excerpt 3: In California, a tenant or subtenant in possession of
a rental housing unit under a month-to-month lease or periodic tenancy when the
property is sold in foreclosure must receive 90 days' written notice to quit
before removal. A tenant holding under a fixed-term residential lease entered
before the foreclosure sale may remain until the end of the lease term unless a
statutory exception applies.
```

Feasibility read: this is acceptable for HousingQA because it removes 21/6853
rows and covers three distinct statutory shapes: eviction procedure, tenant
remedies for essential services, and foreclosure tenancy protection. It avoids
the very broad Alabama nonpayment notice statute, which would force a larger
exclusion set.

## Probe Commands

Two probe shapes are now supported:

- `multi3`: put three exemplar snippets into one Snap-HyRE generation prompt.
- `snap_hyre_exemplar_parallel3`: run three independent exemplar-conditioned
  Snap-HyRE generation calls, one orthogonal exemplar per call, then retrieve
  with the three generated passages as a multi-query bundle.

Generation cache:

```bash
NO_SILENT_FALLBACK=1 EVAL_PASSAGE_STYLE_VARIANT=multi3 \
EVAL_EXCLUDE_GOLD_IDS_PATH=configs/exemplar_multi3_exclusions_2026-05-23.json \
~/.local/bin/uv run python scripts/build_generation_cache.py \
  --mode snap_hyre_exemplar \
  --provider or-gemma4-26b \
  --dataset barexam \
  --questions 50 \
  --passage-style-variant multi3 \
  --exclude-gold-ids-path configs/exemplar_multi3_exclusions_2026-05-23.json \
  --out caches/generation/probes/barexam_q50_seed42_or-gemma4-26b_snap_hyre_exemplar_multi3.jsonl
```

Retrieval cache:

```bash
EVAL_PASSAGE_STYLE_VARIANT=multi3 \
~/.local/bin/uv run python scripts/build_retrieval_cache.py \
  --dataset barexam \
  --questions 50 \
  --query-type hyre_cache \
  --hyre-cache-path caches/generation/probes/barexam_q50_seed42_or-gemma4-26b_snap_hyre_exemplar_multi3.jsonl \
  --expected-provider or-gemma4-26b \
  --passage-style-variant multi3 \
  --exclude-gold-ids-path configs/exemplar_multi3_exclusions_2026-05-23.json \
  --max-k 10 \
  --out caches/retrieval/probes/barexam_q50_seed42_or-gemma4-26b_snap_hyre_exemplar_multi3_k10.jsonl
```

For HousingQA, use the same generation command with `--dataset housing`, and
use `--housing-state-filter` on the retrieval-cache build.

Parallel3 retrieval-only probe:

```bash
NO_SILENT_FALLBACK=1 EVAL_GENERATION_FORMAT_RETRY=1 \
OPENROUTER_PROVIDER_ONLY=Cloudflare LLM_MAX_COMPLETION_TOKENS=2048 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
~/.local/bin/uv run python scripts/run_choice_aware_retrieval_probe.py \
  --provider or-gemma4-26b \
  --dataset barexam \
  --questions 20 \
  --modes rag_simple snap_hyre snap_hyre_exemplar_parallel3 \
  --exclude-gold-ids-path configs/exemplar_multi3_exclusions_2026-05-23.json \
  --max-k 10 \
  --ks 1 5 10 \
  --out caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_parallel3_probe.jsonl \
  --summary-out docs/generated/barexam_q20_snap_hyre_exemplar_parallel3_probe.md
```

For HousingQA, add `--housing-state-filter`. For Legal-Link-EU, omit the
exclusion file for now because the current EU parallel3 bank uses relation
style templates rather than direct gold document ids. MASLegalBench has a
parallel3 bank, but its useful retrieval score is the same-source proxy, not
gold-id Hit@k, so the summary code needs a source-proxy extension before MAS
results should be interpreted.

## Promotion Bar

Keep this as an analysis variant unless it clears all of the following:

- generation cache has zero errors, parse failures, answer artifacts, fallbacks,
  or missing snap final lines;
- retrieval cache has no duplicate labels, short rows, empty retrieval rows, or
  missing ids;
- q50 or q500 retrieval improves over canonical `snap_hyre` after the exemplar
  gold-id exclusions;
- answer-side q50 does not show a conversion drop large enough to swamp the
  retrieval gain.

The prior single-exemplar probe is the baseline to beat. On q500, it gave a
modest BarExamQA retrieval lift over canonical Snap-HyRE (+0.6pp Hit@5, lower
MRR@5) and a modest unfiltered HousingQA lift; later state-filtered Housing
q500 evidence was stronger. Multi-exemplar prompting should therefore be judged
on whether it improves robustness, not merely whether it can match the older
single-exemplar result.

## q20 Smoke Results

Run date: 2026-05-23. Provider: `or-gemma4-26b` with OpenRouter Cloudflare
route. Generation used `NO_SILENT_FALLBACK=1` and
`EVAL_GENERATION_FORMAT_RETRY=1`.

New generation caches:

- `caches/generation/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_multi3.jsonl`
- `caches/generation/probes/housing_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_multi3.jsonl`
- `caches/generation/probes/housing_q20_seed42_or-gemma4-26b_snap_hyre.jsonl`

Generation health: all three caches wrote 20/20 rows with zero errors, parse
failures, answer-artifact flags, or missing snap final lines. The multi3 caches
record `passage_style_signal_variant=multi3` and the expected exemplar ids.

BarExamQA q20 retrieval, same labels:

| Method | Cache | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| Raw question | `caches/retrieval/full/barexam_q20_seed42_raw_question_k10.jsonl` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Canonical Snap-HyRE | `caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` | 0.0000 | 0.0500 | 0.0250 | 0.1000 |
| Single exemplar | `caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl` | 0.0500 | 0.1500 | 0.0850 | 0.1500 |
| Multi3 exemplar | `caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_multi3_k10.jsonl` | 0.0500 | 0.1000 | 0.0750 | 0.1000 |

BarExamQA read: multi3 improves over canonical Snap-HyRE, but the old single
exemplar is still stronger on this q20 slice.

HousingQA q20 retrieval, same labels, state-filtered:

| Method | Cache | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| Raw question | `caches/retrieval/probes/housing_q20_seed42_statefilter_raw_question_k10.jsonl` | 0.2500 | 0.4500 | 0.3292 | 0.5500 |
| Canonical Snap-HyRE | `caches/retrieval/probes/housing_q20_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl` | 0.2000 | 0.3500 | 0.2667 | 0.3500 |
| Single exemplar | `caches/retrieval/probes/housing_q20_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl` | 0.1500 | 0.3000 | 0.2042 | 0.4500 |
| Multi3 exemplar | `caches/retrieval/probes/housing_q20_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_multi3_k10.jsonl` | 0.1500 | 0.2500 | 0.1917 | 0.3000 |

HousingQA read: multi3 is not promising under state-filtered retrieval on this
slice. The state filter already fixes much of the cross-jurisdiction failure
mode, and the exemplar prompt appears to pull generated passages away from the
exact statutory anchors more often than it helps.

Current recommendation: do not promote multi3 as written. Keep the plumbing and
exclusion controls, but if we continue this line, test a more conservative
variant: one exemplar per dataset, or a shorter three-bullet style rubric with
no extra substantive statutory/doctrinal text.

## Parallel3 Smoke Results

Run date: 2026-05-23. Provider: `or-gemma4-26b` with OpenRouter Cloudflare
route. Each `snap_hyre_exemplar_parallel3` row used three independent
Snap-HyRE generation calls and pooled the three generated passages for
retrieval. The calls are independent but executed serially in the probe script
so the existing call counters and failure guards remain reliable.

New outputs:

- `caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_parallel3_probe.jsonl`
- `docs/generated/barexam_q20_snap_hyre_exemplar_parallel3_probe.md`
- `caches/retrieval/probes/housing_q20_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_parallel3_probe.jsonl`
- `docs/generated/housing_q20_statefilter_snap_hyre_exemplar_parallel3_probe.md`
- `caches/retrieval/probes/legal_link_eu_q10_seed42_or-gemma4-26b_snap_hyre_exemplar_parallel3_probe.jsonl`
- `docs/generated/legal_link_eu_q10_snap_hyre_exemplar_parallel3_probe.md`

BarExamQA q20:

| Method | Calls/row | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| Raw question | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Canonical Snap-HyRE | 1 | 0.0000 | 0.0500 | 0.0250 | 0.0500 |
| Parallel3 exemplar | 3 | 0.0000 | 0.0500 | 0.0250 | 0.1500 |

BarExamQA read: parallel3 increased deeper exposure at Hit@10, but did not
improve Hit@5 over canonical Snap-HyRE on this q20 run. The extra two calls
are not justified unless q50/q500 shows a larger top-k gain.

HousingQA q20, state-filtered:

| Method | Calls/row | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| Raw question | 0 | 0.2500 | 0.4500 | 0.3292 | 0.5500 |
| Canonical Snap-HyRE | 1 | 0.2500 | 0.4000 | 0.3017 | 0.4000 |
| Parallel3 exemplar | 3 | 0.3000 | 0.3500 | 0.3167 | 0.3500 |

HousingQA read: parallel3 improved Hit@1/MRR slightly over canonical but
reduced Hit@5 and Hit@10. With state filtering enabled, raw question retrieval
is still strongest on this slice.

Legal-Link-EU q10:

| Method | Calls/row | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| Raw question | 0 | 0.7000 | 0.9000 | 0.7833 | 1.0000 |
| Canonical Snap-HyRE | 1 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| Parallel3 exemplar | 3 | 0.5000 | 0.7000 | 0.5700 | 0.7000 |

Legal-Link-EU read: this is the one promising signal. Parallel3 recovered a
substantial part of the canonical Snap-HyRE retrieval loss, though raw question
retrieval remains far stronger because it preserves source/target act anchors.
If scaling anything from this family, Legal-Link-EU is the best next q50 target.

Current recommendation after parallel3: do not scale HousingQA parallel3. For
BarExamQA, scale only if we specifically care about Hit@10 or want a q50 check
before dropping it. For Legal-Link-EU, run q50 retrieval-only next and inspect
whether the exemplar calls preserve CELEX/source-target anchors consistently.

## Legal-Link-EU q50/q100 Results

Run date: 2026-05-25. Provider: `or-gemma4-26b`, OpenRouter Cloudflare route.
Seed: `42`. Retrieval depth: `k=10`. Guards: `NO_SILENT_FALLBACK=1`,
`EVAL_GENERATION_FORMAT_RETRY=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`.

Outputs:

- `caches/retrieval/probes/legal_link_eu_q50_seed42_or-gemma4-26b_snap_hyre_exemplar_parallel3_ce22000_probe.jsonl`
- `docs/generated/legal_link_eu_q50_snap_hyre_exemplar_parallel3_ce22000_probe.md`
- `caches/retrieval/probes/legal_link_eu_q100_seed42_or-gemma4-26b_snap_hyre_exemplar_parallel3_ce22000_probe.jsonl`
- `docs/generated/legal_link_eu_q100_snap_hyre_exemplar_parallel3_ce22000_probe.md`

The sampling is nested for this dataset and seed: q50 is the first 50 labels of
q100, and q100 is the first 100 labels of q500. The current probe script writes
fresh q50/q100 files rather than resuming from smaller runs, so q100 reran the
first 50 rows for clean gate accounting.

Health was clean in both q50 and q100: zero row errors, zero parse failures,
zero answer-artifact flags, zero empty retrieval rows, and all rows scored.

q50 retrieval:

| Method | Calls/row | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw question | 0 | 0.7400 | 0.1490 | 0.7400 | 0.9200 | 0.4740 | 0.8233 | 0.9600 | 0.6420 | 0.8287 |
| Canonical Snap-HyRE | 1 | 0.3600 | 0.0720 | 0.3600 | 0.5800 | 0.2940 | 0.4483 | 0.7400 | 0.4310 | 0.4679 |
| Parallel3 exemplar | 3 | 0.4600 | 0.0930 | 0.4600 | 0.6800 | 0.3300 | 0.5497 | 0.8000 | 0.4470 | 0.5642 |

q50 decision: parallel3 beat canonical Snap-HyRE by +10pp Hit@5, exactly
meeting the q100 promotion bar. Row-level Hit@5 comparison: 8 wins, 3 losses,
39 ties.

q50 rows where parallel3 beat canonical Snap-HyRE at Hit@5:

- `complex_legallink_31985R1672_52014XC0312(01)_rendered_obsolete_by__row0044`
- `complex_legallink_31987R0569_31985R3143_article_5.4_completes__row0056`
- `complex_legallink_31992R3821_31989R3390_implicitly_repeals__row0113`
- `complex_legallink_32005D0681_32004D0566_implicitly_repeals__row0481`
- `complex_legallink_32006R2008_52014XC0312(01)_rendered_obsolete_by__row0590`
- `complex_legallink_32009R1135_32008D0798_repeals__row0768`
- `complex_legallink_32011D0171_32010D0573_extends_validity__row0812`
- `complex_legallink_32023R2835_32002R1007_implicitly_repeals__row1119`

q50 rows where parallel3 lost to canonical Snap-HyRE at Hit@5:

- `complex_legallink_32003R0490_32003R0301_repeals__row0319`
- `complex_legallink_32004D0468R(01)_32004D0468_corrects__row0381`
- `complex_legallink_32008R0950R(01)_32008R0950_corrects__row0714`

q100 retrieval:

| Method | Calls/row | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw question | 0 | 0.7100 | 0.1425 | 0.7100 | 0.9100 | 0.4410 | 0.7995 | 0.9500 | 0.5750 | 0.8048 |
| Canonical Snap-HyRE | 1 | 0.4300 | 0.0865 | 0.4300 | 0.6200 | 0.2590 | 0.5003 | 0.7200 | 0.3515 | 0.5143 |
| Parallel3 exemplar | 3 | 0.4100 | 0.0825 | 0.4100 | 0.6600 | 0.3030 | 0.5057 | 0.7800 | 0.3955 | 0.5213 |

q100 decision: stop before q500. Parallel3 remained above canonical Snap-HyRE
on Hit@5 (+4pp) and Hit@10 (+6pp), but it missed the predefined q500 promotion
bar of +5pp Hit@5. Row-level Hit@5 comparison: 16 wins, 12 losses, 72 ties.

q100 rows where parallel3 beat canonical Snap-HyRE at Hit@5:

- `complex_legallink_31972R2846_31969R2638_annex_1_completes__row0010`
- `complex_legallink_31984L0643R(01)_31984L0643_corrects__row0031`
- `complex_legallink_31985R1672_52014XC0312(01)_rendered_obsolete_by__row0044`
- `complex_legallink_31986L0155_31969L0208_annex_1_completes__row0051`
- `complex_legallink_31987R0569_31985R3143_article_5.4_completes__row0056`
- `complex_legallink_31992R3821_31989R3390_implicitly_repeals__row0113`
- `complex_legallink_31997R2616R(01)_31997R2616_corrects__row0156`
- `complex_legallink_32004D0255_32002D0611_repeals__row0361`
- `complex_legallink_32004R0922_32000R2348_implicitly_repeals__row0436`
- `complex_legallink_32005D0681_32004D0566_implicitly_repeals__row0481`
- `complex_legallink_32006R2008_52014XC0312(01)_rendered_obsolete_by__row0590`
- `complex_legallink_32009R1135_32008D0798_repeals__row0768`
- `complex_legallink_32011D0171_32010D0573_extends_validity__row0812`
- `complex_legallink_32017D0381_32014D0119_extends_application__row0988`
- `complex_legallink_32022R0994_article_2f_32014R0833_extends_application__row1098`
- `complex_legallink_32023R2835_32002R1007_implicitly_repeals__row1119`

q100 rows where parallel3 lost to canonical Snap-HyRE at Hit@5:

- `complex_legallink_31990R1844_52004XC1211(01)_rendered_obsolete_by__row0088`
- `complex_legallink_32003R0490_32003R0301_repeals__row0319`
- `complex_legallink_32004D0039_32000D0759_repeals__row0352`
- `complex_legallink_32004D0468R(01)_32004D0468_corrects__row0381`
- `complex_legallink_32004R1886_32004R0275_extends_application__row0458`
- `complex_legallink_32006E0913_32004E0847_extends_validity__row0542`
- `complex_legallink_32006L0130_52025XC00663_rendered_obsolete_by__row0551`
- `complex_legallink_32007L0055_31986L0362_annex_2_completes__row0613`
- `complex_legallink_32011R0401_32011R0051_repeals__row0837`
- `complex_legallink_32013R1420_32000R2814_repeals__row0921`
- `complex_legallink_32015R0937_32004R0487_implicitly_repeals__row0966`
- `complex_legallink_32017D2410_32007D0025_extends_application__row1018`

Behavioral read: parallel3 is directionally useful for Legal-Link-EU, but not
yet strong enough to justify the extra two model calls or a q500 run. Exact
CELEX anchor preservation remains poor: q100 canonical Snap-HyRE preserved an
exact source CELEX in 0/100 generated passages and target CELEX in 0/100;
parallel3 preserved source CELEX in 1/100 and target CELEX in 0/100. The lift
appears to come from better relation/topic wording and broader candidate
coverage, while raw question retrieval still dominates because it carries the
source/target identifiers directly.

Recommendation: do not promote parallel3 as-is. If this branch continues, test
a selector variant: generate three exemplar-conditioned passages, score them
against the original question or CELEX-bearing query, keep the best one or two,
and then retrieve. That tests whether exemplars improve generation without
requiring the cross-encoder to clean up three equally pooled, partly
off-target query bundles.
