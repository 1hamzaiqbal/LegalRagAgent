# Paper Meeting Handoff - 2026-05-20

Audience: writing agent and meeting prep. This is a paper-facing synthesis of
the strongest currently available Snap-HyRE signals. Use `docs/signoff_log.md`
as the citation gate for final numeric claims, and `current_status.md` for the
latest operational snapshot.

## Current Matrix State

As of the 2026-05-20 midday status snapshots:

- Active exact-scored benchmarks: BarExamQA, HousingQA, Legal-Link-EU,
  MASLegalBench.
- Excluded from the main exact-scored matrix: CaseHOLD and LegalBench-SCALR.
  Their local corpora are answer-option pools, not reference corpora.
- Active models: `groq-llama8b`, `or-gemma4-26b`, `groq-llama70b`.
- Canonical modes: `llm_only`, `rag_simple`, `golden_passage`,
  `golden_plus_neighbors`, `rag_hyde`, `snap_hyre`, `rag_rewrite`.
- MASLegalBench has no official passage-level gold labels, so
  `golden_passage` and `golden_plus_neighbors` are not applicable. MAS
  retrieval exposure should be called a same-source proxy.
- HousingQA retrieval rows now require jurisdiction state filtering for the
  main matrix. Existing unfiltered Housing retrieval rows are provenance /
  ablation only.

## Recommended Conservative Claim

The safest paper-facing claim is:

> Snap-HyRE can improve retrieval exposure, downstream answer accuracy, or both
> when the original legal question is a weak lexical retrieval query. Across
> the current matrix, its clearest result is BarExamQA, where it improves both
> retrieval Hit@5 and final accuracy for all three models. On anchor-rich
> datasets, Snap-HyRE often reaches downstream parity or improves over other
> generated-query baselines, but it can trail raw question retrieval when the
> raw question already contains high-value document identifiers.

This is stronger and safer than claiming universal dominance over `rag_simple`.
It also gives the analysis section a real explanation: Snap-HyRE helps by
turning fact patterns into doctrine-shaped retrieval queries, but generated
abstraction can remove source/title/jurisdiction anchors that raw retrieval
uses well.

## Immediate Paper Folder Gaps

The active manuscript is under `paper/snap_hyre_2025_05_18/`.

- The current experiment section describes HousingQA as a state-statute corpus
  but does not yet state the jurisdiction state-filter requirement.
- Current paper-generated tables still include old Housing retrieval rows as
  main results. Those rows should be relabeled as unfiltered national-corpus
  provenance or removed from the main matrix until state-filtered rows land.
- Paper coverage tables may still say 71/78 cells; current dashboard coverage
  after the Housing reset is 61/78 signed/full plus one active Gemma Housing
  `llm_only` row.
- The paper already treats `snap_hyre_exemplar` as probe-only, which is good.
  It includes q20 evidence but not the q500 retrieval-scale probe summarized
  below.
- If the paper explains benchmark exclusions, add the concrete reason:
  CaseHOLD/SCALR local corpora embedded answer-option holdings, which is not
  the intended reference-corpus retrieval paradigm.

## Strongest Snap-HyRE Supremacy Signal

BarExamQA is the clean positive headline.

- Downstream final accuracy, `snap_hyre - rag_simple`:
  - `groq-llama8b`: +2.4pp.
  - `or-gemma4-26b`: +4.0pp.
  - `groq-llama70b`: +5.1pp.
- Retrieval Hit@5, raw question vs `snap_hyre`:
  - `groq-llama8b`: 0.0142 to 0.0954.
  - `or-gemma4-26b`: 0.0142 to 0.1205.
  - `groq-llama70b`: 0.0142 to 0.1105.
- Snap-HyRE also beats HyDE retrieval on BarExamQA for all three active
  models, although by smaller margins than the raw-RAG comparison.

Narrative: BarExamQA questions are fact-pattern-heavy and raw embedding search
mostly retrieves case-law fact paragraphs. Snap-HyRE converts the problem into a
more doctrine-shaped query, which shifts retrieval toward MBE/rule passages and
improves final answers.

Conservative parity/help ledger against raw RAG:

| Dataset | Model | Final accuracy delta | Retrieval delta | Conservative read |
|---|---|---:|---:|---|
| BarExamQA | `groq-llama8b` | +2.4pp | +0.081 Hit@5 | Helps both. |
| BarExamQA | `or-gemma4-26b` | +4.0pp | +0.106 Hit@5 | Helps both; best non-oracle row. |
| BarExamQA | `groq-llama70b` | +5.1pp | +0.096 Hit@5 | Helps both. |
| HousingQA, unfiltered provenance | `groq-llama8b` | +0.1pp | +0.047 Hit@5 | Retrieval helps; answer parity. |
| HousingQA, unfiltered provenance | `groq-llama70b` | +5.1pp | +0.061 Hit@5 | Helps both. |
| Legal-Link-EU | `groq-llama8b` | -5.4pp | -0.531 Hit@5 | Fails; raw anchor task. |
| Legal-Link-EU | `or-gemma4-26b` | -0.2pp | -0.227 Hit@5 | Answer parity, retrieval anchor loss. |
| Legal-Link-EU | `groq-llama70b` | -3.2pp | -0.181 Hit@5 | Fails vs raw, but beats HyDE retrieval. |
| MASLegalBench | `groq-llama8b` | +2.0pp | -0.327 same-source@5 | Answer helps despite proxy retrieval loss. |
| MASLegalBench | `or-gemma4-26b` | -1.3pp | -0.373 same-source@5 | Slight answer loss; high-ceiling source-proxy caveat. |
| MASLegalBench | `groq-llama70b` | -0.6pp | -0.294 same-source@5 | Answer parity despite proxy retrieval loss. |

With a +/-1pp downstream parity band, the current signed/provenance ledger has
8/11 Snap-HyRE cells that either improve final accuracy, improve retrieval, or
remain downstream-parity with raw RAG. The three clear misses are Legal-Link-EU
8B, Legal-Link-EU 70B against raw anchors, and MAS Gemma under the same-source
proxy. Housing rows in this table must stay labeled unfiltered provenance until
state-filtered reruns land.

Secondary generated-retrieval comparison: on Legal-Link-EU, Snap-HyRE beats
HyDE retrieval for Gemma 26B and Llama 70B, and Gemma Snap-HyRE is +4.1pp final
accuracy over HyDE. That supports a narrower claim that snap-conditioned
generation can be better than question-only HyDE even when both trail raw anchor
retrieval.

## Important Secondary Positive Signals

HousingQA has a strong directional story, but the fair state-filtered rows are
still in progress.

- In the old unfiltered/provenance full rows, Snap-HyRE improved downstream
  accuracy over `rag_simple` for available signed Groq rows:
  - `groq-llama8b`: +0.2pp.
  - `groq-llama70b`: +5.1pp.
- Retrieval failure diagnosis is clean: unfiltered raw Housing retrieval mostly
  matches wrong-jurisdiction statutes. State filtering is now required for all
  Housing retrieval methods, not only Snap-HyRE.
- State-filtered raw retrieval cache is now complete and clean:
  6,853/6,853 rows, duplicate keys 0, empty retrieval 0, rows without gold 0.
  Raw state-filtered Hit@5/MRR@5 is 0.3695/0.2330, versus old unfiltered raw
  Hit@5/MRR@5 0.0282/0.0148. This is a major fair-interface correction, not a
  Snap-HyRE result by itself.
- The attempted full Housing state-filtered `groq-llama70b` `snap_hyre`
  retrieval cache exited early at 1,344/6,853 rows, with no final write line.
  It was resumed at 13:33 CDT, completed, and audited cleanly. The result is a
  boundary condition rather than a win: state-filtered Snap-HyRE Hit@5/MRR@5 is
  0.2311/0.1427, below state-filtered raw retrieval at 0.3695/0.2330. Do not
  frame Housing as a Snap-HyRE retrieval win unless the downstream answer row
  reaches parity or improves.
- The highest-value active Housing row is now `groq-llama8b` `rag_simple` with
  the state-filtered raw retrieval cache and strict document-cache replay. The
  run preflight validated 20/20 retrieval-cache hits and 20/20 document-cache
  hits before launching the answer row.
- The `groq-llama70b` state-filtered `snap_hyre` answer row is now running
  through strict retrieval/doc/HyRE cache replay. Its preflight validated 20/20
  retrieval-cache hits and 20/20 document-cache hits.

Provisional wording: "HousingQA requires a jurisdiction-aware retrieval
interface; the state-filtered raw cache increases Hit@5 from 0.028 to 0.370,
and generated-query state-filtered caches are running to test whether Snap-HyRE
adds value after jurisdiction scope is fixed."

## Exemplar Probe Signal

`snap_hyre_exemplar` is a probe variant, not a canonical method. It gives the
generation call one fixed, sanitized real passage-style exemplar for the target
dataset, without row-specific evidence, answer labels, or document ids.

Best current result: q500 retrieval probe with `or-gemma4-26b`.

| Benchmark | Raw Hit@5 | Canonical Snap-HyRE Hit@5 | Exemplar Snap-HyRE Hit@5 | Takeaway |
|---|---:|---:|---:|---|
| BarExamQA | 0.0160 | 0.1300 | 0.1360 | Confirms Snap-HyRE retrieval lift over raw; exemplar adds modest Hit@10/Hit@5 help. |
| HousingQA | 0.0260 | 0.0740 | 0.0840 | Confirms generated retrieval improves over raw in unfiltered setting; state-filtered follow-up is needed. |
| Legal-Link-EU | 0.9000 | 0.6820 | 0.7580 | Exemplar recovers some anchor loss but raw remains strongest. |
| MASLegalBench | 0.7261 | 0.3531 | 0.4257 | Same-source proxy improves over canonical Snap-HyRE but remains below raw. |

Use as analysis/probe language only: "A passage-style exemplar partially
repairs Snap-HyRE's anchor loss and strengthens the query-shape interpretation,
but it is not yet promoted to the canonical grid."

If follow-up answer-side or state-filtered runs lift exemplar Snap-HyRE to
parity with raw RAG on the anchor-rich datasets, that is still useful. The
paper-facing story would be that a single fixed Snap-HyRE method is the main
result, while lightweight passage-style calibration is an analysis variant that
can recover raw-query anchors without changing the final-answer interface. The
current q500 retrieval evidence supports "moves toward parity," not "reaches
parity" on Legal-Link-EU or MASLegalBench.

Gap-to-raw framing:

- Legal-Link-EU q500: exemplar closes 0.076/0.218 = 34.9% of the canonical
  Snap-HyRE Hit@5 gap to raw question retrieval.
- MASLegalBench q303/full proxy: exemplar closes 0.073/0.373 = 19.5% of the
  canonical Snap-HyRE same-source@5 gap to raw question retrieval.
- BarExamQA and HousingQA do not need gap-closing framing because canonical
  Snap-HyRE already beats raw; exemplar adds a modest further lift.

## Honest Negative / Boundary Conditions

Do not claim universal Snap-HyRE dominance over raw RAG.

- Legal-Link-EU raw retrieval is very strong because questions preserve
  source/target act anchors. Snap-HyRE can abstract away those anchors.
- MASLegalBench raw retrieval is strong under the same-source proxy because the
  question often preserves source-notice cues. Snap-HyRE can drift into generic
  GDPR/security language.
- These are useful boundary conditions: Snap-HyRE helps when raw questions are
  weak lexical retrieval queries; it can hurt when raw questions already contain
  high-value identifiers.

## Fairness / Benchmark Framing

Use `docs/benchmark_paradigm_audit_2026-05-20.md` as the benchmark-method
fairness source.

Key framing:

- Active corpora are reference/support corpora, not answer-option pools.
- CaseHOLD and LegalBench-SCALR are excluded because every displayed option was
  embedded as a corpus document in the local setup.
- Housing state filtering is a dataset-interface correction, not a new
  Snap-HyRE method. It is applied uniformly to all Housing retrieval methods.
- Cross-dataset averages should be described as benchmark-normalized method
  cells, not as a single identical-corpus experiment.

## What To Say In The Meeting

Safe headline:

> Snap-HyRE is strongest when the raw question is a bad retrieval query. On
> BarExamQA it consistently improves both retrieval exposure and final accuracy
> across all three models. The failure cases on Legal-Link-EU and MAS are not
> harness failures; they show that when the raw question already contains
> document/source anchors, generated legal abstraction can remove useful
> retrieval keys.

Safe caveat:

> HousingQA is being rerun with a required state filter because the national
> corpus otherwise turns the task into cross-jurisdiction matching. We should
> treat previous unfiltered Housing retrieval rows as provenance only.

Provisional add-on:

> A q500 real-passage exemplar probe suggests passage-style guidance can improve
> canonical Snap-HyRE retrieval on all four datasets, especially Legal-Link-EU
> and MAS where canonical Snap-HyRE loses anchors, but this remains an analysis
> variant pending answer-side checks.

## Do Not Overclaim

- Do not cite incomplete Housing state-filter rows as final.
- Do not describe MAS retrieval metrics as official gold-passage Hit@k/MRR.
- Do not claim `snap_hyre_exemplar` is canonical.
- Do not resurrect CaseHOLD/SCALR in the main matrix.
- Do not use raw `logs/experiments.jsonl` rows without checking
  `docs/signoff_log.md`.
