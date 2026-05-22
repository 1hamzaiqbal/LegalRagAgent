# Snap-HyRE Paper Agent Handoff - 2026-05-20

Audience: paper-writing agent. Purpose: give the strongest current
Snap-HyRE story without forcing the writer to reconstruct it from live logs.

Use this as a narrative guide, not as the final cite gate. For exact numeric
claims, verify against `docs/signoff_log.md` first, then `current_status.md`
for active rows. A `*` below means the row is active/provisional or the method
is probe-only.

## Source Files To Open First

- `current_status.md`: live completion and metric dashboard.
- `docs/signoff_log.md`: cite-or-not gate for full signed rows.
- `docs/paper_meeting_handoff_2026-05-20.md`: broader meeting synthesis.
- `docs/snap_hyre_good_example_handoff_2026-05-20.md`: concrete BarExamQA
  Snap-HyRE win and current exemplar-worth-it read.
- `docs/retrieval_passage_exemplar_probe_2026-05-20.md`: exemplar probe details.
- `docs/benchmark_paradigm_audit_2026-05-20.md`: why CaseHOLD/SCALR are excluded.
- `docs/housingqa_state_filtered_process_2026-05-20.md`: Housing state-filter contract.
- `paper/snap_hyre_2025_05_18/current_audited_rows.csv`: paper asset input.
- `paper/snap_hyre_2025_05_18/tables/current_snap_deltas.tex`: current paper table.
- `paper/snap_hyre_2025_05_18/tables/current_snap_vs_controls.tex`: method-control table.

## Current State At A Glance

Snapshot source: `current_status.md`, 2026-05-20 14:54 CDT.

- Active matrix: BarExamQA, HousingQA, Legal-Link-EU, MASLegalBench.
- Active models: `groq-llama8b`, `or-gemma4-26b`, `groq-llama70b`.
- Canonical modes: `llm_only`, `rag_simple`, `golden_passage`,
  `golden_plus_neighbors`, `rag_hyde`, `snap_hyre`, `rag_rewrite`.
- Signed/full answer cells: 61/78.
- Active cells: 3/78, all HousingQA.
- Excluded from main matrix: CaseHOLD and LegalBench-SCALR, because the local
  setup used answer-option/holding pools as corpora rather than reference
  corpora.

Housing state as of the 15:12 CDT dashboard refresh:

| Benchmark | Model | Mode | Progress | Current partial f_acc | Health read |
|---|---|---|---:|---:|---|
| HousingQA | `groq-llama70b` | `snap_hyre` | 1711/6853 active | 60.7%* | strict retrieval/doc/HyRE cache replay; 0 fallback keys, 0 cache misses, 0 state-filter misses |
| HousingQA | `groq-llama8b` | `rag_simple` | 5740/6853 active | 62.6%* | strict state-filter raw retrieval/doc cache replay; 0 fallback keys, 0 cache misses |
| HousingQA | `or-gemma4-26b` | `llm_only` | 3680/6853 paused | 55.0%* | paused to prioritize exemplar; 0 fallback keys, 0 missing predictions |

Do not treat the starred partial Housing f_acc values as final.

Priority follow-up:

- `housing_gemma_llm_only_tail_parasail_20260520_0614` was paused at
  3,678 total detail rows so the Gemma/OpenRouter lane could be used for the
  exemplar probe. It can be resumed later from the next Housing sample index.
- `exemplar_answer_gemma26_priority_20260520_151049` stopped at BarExam 55/500
  due to a false-positive no-silent artifact guard on ordinary prose ("it is a
  fair representation"). The guard was tightened in `eval/eval_harness.py` and
  `scripts/analyze_detail_flags.py`.
- `exemplar_answer_gemma26_priority_resume_20260520_154427` is now running in
  tmux from BarExam row 54, then Legal-Link-EU q500, then MASLegalBench full.
- It uses `scripts/local/run_snap_hyre_exemplar_answer_probe.sh` with strict
  generation/retrieval cache replay from `caches/generation/probes` and
  `caches/retrieval/probes`; the active tail detail log prefix is
  `logs/eval_snap_hyre_exemplar_or-gemma4-26b_20260520_154432_barexam_...`.
- HousingQA is intentionally excluded from this queued exemplar answer probe
  because the available q500 Housing exemplar cache is the old unfiltered
  probe, while the current main Housing setup requires state-filtered
  retrieval.
- Before queueing, 20-row hydration smokes passed for all three retrieval
  caches with 20/20 cache hits.

## Best Current Snap-HyRE Claim

The cleanest and strongest claim is BarExamQA:

> Snap-HyRE improves both retrieval exposure and downstream answer accuracy
> over raw-question RAG on BarExamQA for all three active models.

BarExamQA signed/full rows:

| Model | Raw RAG f_acc | Snap-HyRE f_acc | Delta | Raw Hit@5 | Snap-HyRE Hit@5 | Retrieval delta |
|---|---:|---:|---:|---:|---:|---:|
| `groq-llama8b` | 54.5% | 56.9% | +2.4pp | 0.0142 | 0.0954 | +0.0812 |
| `or-gemma4-26b` | 78.0% | 82.0% | +4.0pp | 0.0142 | 0.1205 | +0.1063 |
| `groq-llama70b` | 74.6% | 79.7% | +5.1pp | 0.0142 | 0.1105 | +0.0963 |

Snap-HyRE also beats HyDE retrieval on BarExamQA:

| Model | HyDE Hit@5 | Snap-HyRE Hit@5 | Delta |
|---|---:|---:|---:|
| `groq-llama8b` | 0.0828 | 0.0954 | +0.0126 |
| `or-gemma4-26b` | 0.1138 | 0.1205 | +0.0067 |
| `groq-llama70b` | 0.1046 | 0.1105 | +0.0059 |

Paper wording:

> On BarExamQA, raw question retrieval almost never surfaces the gold rule
> passage, while Snap-HyRE converts fact-pattern questions into doctrine-shaped
> retrieval text. This yields a consistent retrieval lift and a downstream
> accuracy lift across 8B, 26B, and 70B models.

## Why Emphasize Gemma 26B

Gemma 26B is the best interpretive center of gravity for the paper.

- The 8B row is useful but often reasoning-limited.
- The 70B row is useful but may answer from legal priors even when retrieval is
  imperfect.
- Gemma 26B is the most method-reactive middle model: strong enough to use
  evidence, weak enough that retrieval/query changes visibly move outcomes.

Gemma 26B high-signal rows:

| Benchmark | Raw RAG f_acc | HyDE f_acc | Snap-HyRE f_acc | Snap-HyRE read |
|---|---:|---:|---:|---|
| BarExamQA | 78.0% | 80.3% | 82.0% | Best non-oracle canonical row; +4.0pp over raw and +1.7pp over HyDE. |
| Legal-Link-EU | 78.5% | 74.2% | 78.3% | Downstream parity with raw (-0.2pp) and +4.1pp over HyDE; Snap retrieval beats HyDE/rewrite but trails raw anchors. |
| MASLegalBench | 94.7% | 95.4% | 93.4% | Miss under same-source proxy; useful boundary condition, not a headline win. |
| HousingQA | -- | -- | -- | State-filtered Gemma retrieval rows still not complete; only `llm_only` is active. |

Suggested paper framing:

> We focus interpretation on Gemma 26B because it is neither too weak to use
> retrieved evidence nor so strong that retrieval changes are masked by prior
> legal knowledge.

## Additional Positive Or Parity Signals

Legal-Link-EU:

- Raw RAG is very strong because questions contain source/target legal anchors.
- Gemma 26B Snap-HyRE reaches answer parity with raw RAG: 78.3% vs 78.5%.
- Gemma 26B Snap-HyRE is +4.1pp over HyDE final accuracy: 78.3% vs 74.2%.
- Gemma 26B Snap-HyRE retrieval beats HyDE and rewrite: Hit@5 0.6788 vs 0.4898
  HyDE and 0.5439 rewrite.
- Llama 70B Snap-HyRE retrieval also beats HyDE/rewrite: Hit@5 0.7249 vs
  0.5466 HyDE and 0.6584 rewrite, but final accuracy trails raw.

MASLegalBench:

- MAS retrieval is a same-source proxy, not official gold-evidence retrieval.
- `groq-llama8b` Snap-HyRE improves final accuracy over raw RAG:
  88.1% vs 86.1% (+2.0pp).
- `groq-llama70b` Snap-HyRE is near downstream parity with raw:
  92.1% vs 92.7% (-0.6pp).
- Gemma 26B misses under canonical Snap-HyRE: 93.4% vs 94.7% raw and
  95.4% HyDE.

HousingQA:

- Housing requires a jurisdiction state filter for fair retrieval. This is a
  dataset-interface correction applied uniformly, not a Snap-HyRE-specific
  tweak.
- State-filtered raw retrieval cache is clean and much stronger than old
  unfiltered raw retrieval: Hit@5/MRR@5 0.3695/0.2330 vs old unfiltered
  0.0282/0.0148.
- State-filtered Llama 70B Snap-HyRE retrieval cache is clean but trails raw:
  Hit@5/MRR@5 0.2311/0.1427.
- The Llama 70B Snap-HyRE answer row is active and clean so far; use only as
  provisional until full/signoff.

## Exemplar Probe: Promising But Starred

`snap_hyre_exemplar` is not canonical yet. It gives the generator one fixed,
sanitized real passage-style exemplar for the target dataset. It does not
provide row-specific evidence, document ids, answer labels, or the current
row's gold passage.

The q20 Gemma probe was directionally good:

| Benchmark | Raw Hit@5/MRR@5 | Canonical Snap-HyRE | Exemplar Snap-HyRE* |
|---|---:|---:|---:|
| BarExamQA | 0.0000 / 0.0000 | 0.0500 / 0.0250 | 0.1500 / 0.0850 |
| HousingQA | 0.0000 / 0.0000 | 0.0500 / 0.0500 | 0.1000 / 0.1000 |
| Legal-Link-EU | 0.8500 / 0.7167 | 0.7500 / 0.5808 | 0.7500 / 0.7000 |
| MASLegalBench | 0.7000 / 0.5892 | 0.4000 / 0.2475 | 0.5000 / 0.2908 |

The larger q500/full retrieval probe also supports the exemplar as a useful
analysis variant:

| Benchmark | Raw Hit@5 | Canonical Snap-HyRE Hit@5 | Exemplar Snap-HyRE Hit@5* | Read |
|---|---:|---:|---:|---|
| BarExamQA q500 | 0.0160 | 0.1300 | 0.1360 | Snap-HyRE already wins; exemplar adds a small lift. |
| HousingQA q500, unfiltered | 0.0260 | 0.0740 | 0.0840 | Snap-HyRE wins unfiltered; state-filtered answer/retrieval follow-up needed. |
| Legal-Link-EU q500 | 0.9000 | 0.6820 | 0.7580 | Exemplar recovers part of anchor loss but raw remains strongest. |
| MASLegalBench full proxy | 0.7261 | 0.3531 | 0.4257 | Exemplar improves canonical Snap-HyRE but remains below raw. |

Answer-side exemplar smoke:

- BarExamQA q20, Gemma 26B, `snap_hyre_exemplar`: 19/20 = 95.0%.
- The one miss retrieved gold and had a correct snap prior, but the final
  answerer selected the wrong option. That is an evidence-use/option-mapping
  miss, not a retrieval miss.

Safe paper wording:

> A fixed passage-style exemplar partially repairs generated-query anchor loss
> and modestly improves Snap-HyRE retrieval in the Gemma 26B probes. We treat
> this as an analysis variant, not part of the main canonical grid, until
> larger answer-side runs finish.

Best-case starred wording if the paper wants to preview it:

> In provisional Gemma 26B probes, a single sanitized corpus-style exemplar
> moved Snap-HyRE toward parity on anchor-rich datasets while preserving the
> BarExamQA/Housing retrieval lift. Full answer-side scaling is in progress.

## Do Not Overclaim

Avoid saying Snap-HyRE universally beats raw RAG.

- It clearly beats raw RAG on BarExamQA across all three models.
- It reaches useful parity/help on selected Legal-Link/MAS cells.
- It can trail raw retrieval on anchor-rich tasks where the raw question
  already contains source, target, jurisdiction, title, or notice anchors.
- Housing is not interpretable without the state filter. Old unfiltered
  Housing rows should be labeled provenance/ablation, not main results.
- MAS retrieval metrics are same-source proxies, not passage-level gold qrels.

Best conservative claim:

> Snap-HyRE helps when the original legal question is a weak lexical retrieval
> query. Its strongest current evidence is BarExamQA, where retrieval exposure
> and final accuracy improve across all three models. On anchor-rich datasets,
> Snap-HyRE can match or beat generated-query controls and sometimes reaches
> downstream parity with raw RAG, but raw anchors can remain the best retrieval
> signal.

## Useful Result Snippets For Manuscript Drafting

BarExamQA sentence:

> BarExamQA is the clearest demonstration: Snap-HyRE raises Hit@5 from 0.0142
> to 0.0954/0.1205/0.1105 across 8B/26B/70B, and raises final accuracy by
> +2.4/+4.0/+5.1pp over raw RAG.

Gemma 26B sentence:

> The Gemma 26B row is the most informative method-sensitivity slice: on
> BarExamQA, Snap-HyRE reaches 82.0%, outperforming raw RAG (78.0%), HyDE
> (80.3%), rewrite (80.7%), and all non-oracle controls.

Legal-Link sentence:

> Legal-Link-EU shows the boundary condition: raw retrieval is strongest when
> source/target anchors are already present, but Gemma 26B Snap-HyRE still
> reaches raw-RAG answer parity and substantially beats HyDE retrieval and
> final accuracy.

Exemplar sentence:

> Provisional exemplar probes suggest that a fixed corpus-style passage example
> can recover part of Snap-HyRE's anchor loss on Legal-Link-EU and MASLegalBench,
> while preserving the retrieval-lift story on BarExamQA and HousingQA.

## Open Work Before Final Claims

- Finish and sign off active Housing rows, especially Llama 70B state-filtered
  `snap_hyre`.
- Decide whether to scale `snap_hyre_exemplar` answer-side beyond q20/q500
  retrieval probes.
- If exemplar is promoted, add a method subsection that clearly separates
  canonical `snap_hyre` from `snap_hyre_exemplar`.
- Refresh paper tables from signed rows only after active Housing jobs finish.
- Keep current claims footnoted where they rely on active/probe rows.
