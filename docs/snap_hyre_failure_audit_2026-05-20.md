# Snap-HyRE Failure Audit - 2026-05-20

Purpose: identify places where `snap_hyre` is weaker than simple RAG or other
query-generation variants, and separate real method weaknesses from harness
health issues.

Sources checked:

- `current_status.md` snapshots from 2026-05-20 06:32, 06:44, 06:56, 07:07,
  07:13, 07:15, and 07:21 CDT.
- `docs/signoff_log.md` for signed result gates.
- Full retrieval caches under `caches/retrieval/full/*_qfull_seed42_*_k10.jsonl`.
- Full signed Snap-HyRE detail logs for BarExamQA, Legal-Link-EU, and
  MASLegalBench.
- HousingQA Llama 70B full detail log and the merged/signed HousingQA
  Llama 8B Snap-HyRE detail log.
- Active HousingQA Llama 8B `rag_rewrite` partial logs through the failed-closed
  prefix, one-row repair, and resumed tail.
- Gemma 26B q500/qfull real-passage exemplar generation cache health.
- Fresh paired raw-vs-Snap retrieval/answer disagreement scan over the signed
  full rows and active HousingQA partials.

## Bottom Line

We are not seeing silent fallback or truncation-cooking as the main problem. The
main problem is substantive and dataset-shaped:

- BarExamQA: Snap-HyRE helps retrieval over raw question retrieval across all
  three active models.
- HousingQA: Snap-HyRE helps over raw question retrieval, but loses to HyDE,
  especially in several states where jurisdiction matching is hard.
- Legal-Link-EU: raw question retrieval is very strong because the question
  preserves source/target act anchors. Snap-HyRE often abstracts away those
  anchors. Gemma 26B and Llama 70B Snap-HyRE beat HyDE here, but all Snap-HyRE
  variants are below raw `rag_simple` retrieval.
- MASLegalBench: raw question retrieval has the strongest same-source proxy.
  Snap-HyRE tends to generate generic GDPR/security passages and retrieve
  similar provisions from the wrong penalty notice.

This means the current evidence does not support a universal claim that
Snap-HyRE improves retrieval over `rag_simple` on every current benchmark. The
better claim is narrower: Snap-HyRE improves retrieval when the raw question is
a fact pattern or weak lexical query; it can hurt when raw questions contain
high-value document/source anchors.

## Retrieval Delta Summary

Hit@5 deltas are `snap_hyre - raw_question`; MAS uses the same-source-document
proxy, not official qrels.

| Dataset | Model | Raw Hit@5 | Snap Hit@5 | Delta | HyDE Hit@5 | Snap - HyDE | Snap wins vs raw | Snap losses vs raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | `groq-llama8b` | 0.0142 | 0.0954 | +0.0812 | 0.0828 | +0.0126 | 110 | 13 |
| BarExamQA | `or-gemma4-26b` | 0.0142 | 0.1205 | +0.1063 | 0.1138 | +0.0067 | 138 | 11 |
| BarExamQA | `groq-llama70b` | 0.0142 | 0.1105 | +0.0962 | 0.1046 | +0.0059 | 128 | 13 |
| HousingQA | `groq-llama8b` | 0.0282 | 0.0747 | +0.0465 | 0.1246 | -0.0499 | 459 | 140 |
| HousingQA | `groq-llama70b` | 0.0282 | 0.0892 | +0.0610 | 0.1665 | -0.0773 | 538 | 120 |
| Legal-Link-EU | `groq-llama8b` | 0.9059 | 0.3753 | -0.5306 | 0.4756 | -0.1003 | 32 | 630 |
| Legal-Link-EU | `or-gemma4-26b` | 0.9059 | 0.6788 | -0.2272 | 0.4898 | +0.1890 | 51 | 307 |
| Legal-Link-EU | `groq-llama70b` | 0.9059 | 0.7249 | -0.1810 | 0.5466 | +0.1783 | 52 | 256 |
| MASLegalBench | `groq-llama8b` | 0.7261 | 0.3993 | -0.3267 | 0.4818 | -0.0825 | 12 | 111 |
| MASLegalBench | `or-gemma4-26b` | 0.7261 | 0.3531 | -0.3729 | 0.3894 | -0.0363 | 16 | 129 |
| MASLegalBench | `groq-llama70b` | 0.7261 | 0.4323 | -0.2937 | 0.4587 | -0.0264 | 20 | 109 |

## Downstream Accuracy Check

Accuracy deltas are `snap_hyre - rag_simple` from the current operational
matrix.

| Dataset | `groq-llama8b` | `or-gemma4-26b` | `groq-llama70b` | Interpretation |
|---|---:|---:|---:|---|
| BarExamQA | +2.4pp | +4.0pp | +5.1pp | Retrieval gains mostly translate. |
| HousingQA | +0.2pp | n/a | +5.1pp | Retrieval is still weak, but evidence sometimes improves final accuracy. |
| Legal-Link-EU | -5.4pp | -0.2pp | -3.2pp | Anchor loss hurts or neutralizes final answers. |
| MASLegalBench | +2.0pp | -1.3pp | -0.6pp | Final accuracy is high overall; same-source retrieval proxy drops do not always translate into answer drops. |

## Paired Retrieval/Answer Disagreement Check

The following scan compares `rag_simple` raw-question retrieval against
`snap_hyre` on the same question ids. Retrieval is Hit@5 except for
MASLegalBench, where it is the same-source-document proxy. `good` counts rows
where Snap both newly retrieved the target/proxy and fixed a raw wrong answer.
`bad` counts rows where raw retrieved the target/proxy and answered correctly
but Snap missed the target/proxy and answered wrong. `ret-win/ans-loss` counts
rows where Snap improved retrieval exposure but the final answer regressed.

| Dataset | Model | Raw hit | Snap hit | Raw acc | Snap acc | Good | Bad | Ret-win/ans-loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | `groq-llama8b` | 0.0142 | 0.0954 | 0.5448 | 0.5690 | 31 | 5 | 11 |
| BarExamQA | `or-gemma4-26b` | 0.0142 | 0.1205 | 0.7799 | 0.8201 | 14 | 1 | 7 |
| BarExamQA | `groq-llama70b` | 0.0142 | 0.1105 | 0.7456 | 0.7975 | 10 | 1 | 3 |
| HousingQA | `groq-llama8b` | 0.0282 | 0.0747 | 0.5727 | 0.5745 | 101 | 42 | 71 |
| HousingQA | `groq-llama70b` | 0.0282 | 0.0892 | 0.4729 | 0.5242 | 132 | 29 | 25 |
| Legal-Link-EU | `groq-llama8b` | 0.9059 | 0.3753 | 0.5741 | 0.5200 | 6 | 167 | 4 |
| Legal-Link-EU | `or-gemma4-26b` | 0.9059 | 0.6788 | 0.7853 | 0.7835 | 10 | 57 | 3 |
| Legal-Link-EU | `groq-llama70b` | 0.9059 | 0.7249 | 0.7533 | 0.7214 | 4 | 37 | 5 |
| MASLegalBench | `groq-llama8b` | 0.7261 | 0.3993 | 0.8614 | 0.8812 | 0 | 8 | 0 |
| MASLegalBench | `or-gemma4-26b` | 0.7261 | 0.3531 | 0.9472 | 0.9340 | 0 | 2 | 1 |
| MASLegalBench | `groq-llama70b` | 0.7261 | 0.4323 | 0.9274 | 0.9208 | 2 | 3 | 0 |

This check sharpens the narrative. BarExamQA is the clean positive case: Snap
creates many more retrieval wins than losses and the answer wins dominate. On
Legal-Link-EU, raw retrieval is already near-oracle because the question carries
act identifiers, so Snap has many more retrieval-loss/answer-loss rows. On
MASLegalBench, source grounding is the weak point, but answer accuracy is near
ceiling enough that retrieval proxy drops do not always move final accuracy.

Worst grouped raw-vs-Snap retrieval deltas in the fresh scan:

- HousingQA, Llama 70B: Arkansas (-0.044), South Carolina (-0.044),
  New Hampshire (-0.040), South Dakota (-0.040), Rhode Island (-0.029).
- HousingQA, Llama 8B: Arkansas (-0.089), South Carolina (-0.051),
  Rhode Island (-0.041), Alaska (-0.022), Maine (-0.019).
- Legal-Link-EU, Gemma 26B: `rendered_obsolete_by` (-0.453), `corrects`
  (-0.304), `extends_validity` (-0.261), `repeals` (-0.236).
- Legal-Link-EU, Llama 70B: `rendered_obsolete_by` (-0.447),
  `extends_validity` (-0.248), `corrects` (-0.180), `repeals` (-0.149).
- MASLegalBench, Gemma 26B: `psni-penalty-notice` (-0.706),
  `birthlink-mpn` (-0.583), `advanced-penalty-notice-20250327` (-0.529),
  `clearview-ai-inc-mpn-20220518` (-0.462).

## Concrete Failure Patterns

Raw retrieval failure on BarExamQA:

- Full-cache diagnostic: `scripts/audit_retrieval_id_alignment.py` found
  1149/1149 unique full-row gold ids in Chroma. Every BarExamQA gold id is
  `mbe_*`, but raw `rag_simple` top-1 is `caselaw_*` on 1063/1195 rows and the
  raw top five contain at least one `caselaw_*` on 1182/1195 rows. Raw Hit@1 is
  0.0033, Hit@5 is 0.0142, and Hit@10 is 0.0218 overall. The top five are all
  case-law passages on 881/1195 rows. Raw Hit@5 is 0.0160 on the q500 exemplar
  slice. Canonical Gemma Snap-HyRE reaches Hit@5 0.1205 overall and 0.1300 on
  that same q500 slice.
- Example: `qa_nan_mbe_73`
- Raw top-5 retrieved case-law passages keyed to surface fact-pattern terms and
  missed gold `mbe_58`; the real-passage exemplar Snap-HyRE probe retrieved
  `mbe_58` at rank 5.
- Example: `qa_nan_mbe_58`
- Raw top-5 retrieved recording/deed-ish distractors and missed gold `mbe_49`;
  the real-passage exemplar probe retrieved `mbe_49` at rank 2.
- Interpretation: raw BarExamQA questions often look like fact narratives, not
  compact doctrine queries. Snap-style prior reasoning can convert those
  narratives into the legal concept the corpus indexes better.
- Additional q500/full diagnosis: on the full raw cache, Hit@1/5/10 is
  0.0033/0.0142/0.0218 and MRR@5 is 0.0068. The raw top-1 prefix distribution
  is 1063 `caselaw`, 118 `mbe`, 12 `wex`, and 2 `ucc`; at least one `caselaw`
  passage appears in top-5 on 1182/1195 rows and all five top passages are
  `caselaw` on 881/1195 rows. In the q500 probe, raw top-1 is still mostly
  `caselaw` (447/500), while canonical Gemma Snap-HyRE shifts top-1 to 264
  `mbe` / 183 `caselaw` and the real-passage exemplar shifts it to 265 `mbe` /
  175 `caselaw`. That supports a query-shape failure diagnosis rather than an
  absent-gold or hydration failure.
- Example q500 raw miss fixed by exemplar: `mbe_339` asks about a requirements
  contract/UCC buyer variation issue. Raw retrieved payment/default fact
  paragraphs from unrelated case law; the exemplar query retrieved `mbe_2640`
  at rank 1, plus requirements-contract and UCC good-faith passages. Example
  `mbe_73` similarly raw-retrieved traffic-accident case facts while the
  exemplar retrieved MBE proximate-cause/legal-cause rule passages including
  gold `mbe_58`.

Raw retrieval failure on HousingQA:

- Full-cache diagnostic: `scripts/audit_retrieval_id_alignment.py` found
  990/990 unique full-row gold ids in Chroma. Raw `rag_simple` retrieves a
  same-state statute at rank 1 on only 2.3% of rows and includes any same-state
  statute in the top five on only 9.4% of rows. 90.6% of rows both miss the
  gold id and have no same-state statute in the top five. Raw Hit@1 is 0.0079,
  Hit@5 is 0.0282, and Hit@10 is 0.0511 overall. The weak raw Hit@5 therefore
  mostly comes from cross-jurisdiction lexical matches, not absent retrieved
  text or scoring bugs.
- Example: `hqa_Michigan_7987`
- Raw top-5 retrieved statutes from other jurisdictions and missed gold
  `923935`; the real-passage exemplar probe retrieved `923935` at rank 1.
- Example: `hqa_Michigan_4232`
- Raw top-5 again retrieved wrong-jurisdiction statutes and missed the Michigan
  gold set; the exemplar probe retrieved `923919` at rank 1.
- Interpretation: raw HousingQA retrieval struggles because many state statutes
  have near-identical eviction/landlord-tenant wording. Adding state-specific
  legal framing can help, but the full Snap-HyRE runs still trail HyDE on
  several states.
- Additional q500/full diagnosis: on the full raw cache, Hit@1/5/10 is
  0.0079/0.0282/0.0511 and MRR@5 is 0.0148. Same-state retrieval is the
  bottleneck: raw top-1 is same-state on only 2.3% of rows, any top-5 passage is
  same-state on 9.4%, and 90.6% of rows have no same-state statute in top-5.
  Llama 70B HyDE raises any same-state@5 to 63.9% and Hit@5 to 0.1665; Llama
  70B Snap-HyRE raises any same-state@5 to 53.6% and Hit@5 to 0.0892. In the
  q500 Gemma probe, raw any same-state@5 is 9.6%, canonical Snap-HyRE is 32.8%,
  and real-passage exemplar Snap-HyRE is 32.4%. So the generated methods repair
  jurisdiction grounding substantially, but exact-statute matching remains
  harder than with HyDE.
- Worst raw states with zero Hit@5 and at least 20 questions include Wyoming,
  Puerto Rico, Michigan, Georgia, Louisiana, Maryland, Utah, West Virginia,
  Washington, Oklahoma, Delaware, Massachusetts, Vermont, and New York. Common
  top-1 wrong-state attractors include many states retrieving Arkansas, Texas,
  or South Carolina statutes for generic eviction wording.
- Example q500 raw miss fixed by generated retrieval: `hqa_Michigan_7987`
  asks whether "writ of restitution" refers to a court order for removal. Raw
  retrieved North Carolina, Colorado, Nevada, Hawaii, and Washington statutes;
  Snap-HyRE and exemplar both retrieved Michigan `MI Comp L § 600.5744` at rank
  1. Example `hqa_Michigan_4232` asks whether eviction cases are first heard in
  court of common pleas. Raw retrieved Arkansas/Ohio material; Snap-HyRE
  retrieved Michigan `MI Comp L § 600.5704` at rank 1 and the exemplar retrieved
  it at rank 2.

Legal-Link-EU anchor loss:

- Example: `complex_legallink_31983R1303_31982R2834_implicitly_repeals__row0025`
- `rag_simple`: raw retrieved both relevant act anchors, including
  `31982R2834` and `31983R1303`; answer correct.
- `snap_hyre`: generated a generic import-licence/security query and retrieved
  related but wrong regulations such as `31975R2042_annex_2`, `32006R1885`, and
  `32003R0701`; answer wrong.
- Interpretation: Snap-HyRE's legal abstraction can erase source/target
  identifiers that are already highly informative in this dataset.

MASLegalBench same-source loss:

- Example: `maslb_135bfe175427134f`
- `rag_simple`: retrieved same-source Clearview passages and answered correctly.
- `snap_hyre`: generated a broad GDPR Article 32/security query and retrieved
  passages from Central YMCA, Advanced, and Cabinet Office notices; answer
  wrong.
- Additional example: `maslb_a66d2c47e3ac628a`
- `rag_simple`: retrieved same-source Birthlink passages and answered correctly.
- `snap_hyre`: reasoned toward a broad accountability/principles framing and
  retrieved Chapter II/principle passages from PSNI and 23andMe notices instead
  of Birthlink. The evidence is legally adjacent but source-misaligned.
- Interpretation: the generated passage finds semantically related law, but the
  benchmark needs source-notice grounding more than generic doctrinal grounding.

HousingQA jurisdiction loss:

- Example: `hqa_Alabama_163`
- HyDE retrieved the specific Alabama nonpayment statute `AL Code 35-9A-421`;
  Snap-HyRE retrieved nearby Alabama eviction/jurisdiction sections but missed
  the gold statute and answered wrong.
- Worst Llama 70B Snap-vs-HyDE retrieval states by Hit@5 delta included
  Wisconsin (-0.273), Washington (-0.234), Michigan (-0.222), District of
  Columbia (-0.202), and Idaho (-0.193).
- Interpretation: Snap-HyRE is often state-aware, but it may reason to a nearby
  legal concept instead of the exact statutory hook.

Fresh state-level check for HousingQA Llama 70B still points to exact-statute
matching rather than fallback or truncation as the weakness. Worst Snap-HyRE
minus HyDE Hit@5 states, among states with at least 20 rows, are Wisconsin
(-0.273), Washington (-0.234), Michigan (-0.222), District of Columbia
(-0.202), Idaho (-0.193), Kansas (-0.192), Rhode Island (-0.176), Alabama
(-0.170), Nebraska (-0.139), and California (-0.138).

Fresh subject-level check for Legal-Link-EU Gemma 26B shows Snap-HyRE is closest
to raw retrieval on `completes` (-0.025 Hit@5) and worst on
`rendered_obsolete_by` (-0.453), followed by `corrects` (-0.304),
`extends_validity` (-0.261), `repeals` (-0.236), `extends_application`
(-0.186), and `implicitly_repeals` (-0.124). This again looks like source /
target act anchor loss, not a bad API or scoring setup.

BarExamQA retrieval-positive but answer-regression example:

- Example: `qa_CRIM. LAW_mbe_565`
- Raw question retrieval missed gold while Snap-HyRE retrieval hit a relevant
  MBE/accomplice-liability passage.
- The Snap-HyRE answer still regressed from correct to wrong.
- Additional example: `qa_nan_mbe_270`
- Raw question retrieval missed the gold MBE but answered correctly; Snap-HyRE
  retrieved a possibility-of-reverter passage and still selected the wrong
  option. This is answer-stage option mapping failure after improved retrieval,
  not a retrieval miss.
- Interpretation: retrieval exposure can improve while final option mapping
  still fails. This is important for the paper narrative: retrieval improvement
  and downstream accuracy are related but not identical outcomes.

## Harness Health

Checked Snap-HyRE full/active detail logs for errors, missing predictions,
empty retrieval, exact final answer lines, fallback markers, think tags,
near-cap outputs, cache misses, and HyRE cache misses.

## Housing Metadata Filter Follow-Up

The HousingQA failure mode is strong enough that a Housing-specific metadata
process is justified, as long as it is applied uniformly across retrieval
methods rather than only to Snap-HyRE. The corpus mixes statutes from all
jurisdictions, and raw dense retrieval often returns topically similar statutes
from Arkansas, Texas, South Carolina, Hawaii, etc. for questions about another
state. That is a corpus-scope problem, not an answer-label leak.

Implementation change landed 2026-05-20:

- `EvalConfig` now has `housing_state_filter`, and the harness also respects
  `EVAL_HOUSING_STATE_FILTER=1`.
- When enabled, `rag_simple`, `rag_hyde`, `snap_hyre`, `rag_rewrite`,
  `golden_plus_neighbors`, and the exemplar probe variants all use the same
  Chroma `where={"state": <question_state_lowercase>}` constraint for
  HousingQA retrieval.
- `scripts/build_retrieval_cache.py` has the matching
  `--housing-state-filter` flag and writes row-specific `where` values into
  retrieval caches, so strict replay keys cannot collide with unfiltered
  caches.
- `scripts/local/run_answer_cell.sh` appends `_statefilter` to the default
  Housing cache scope when `EVAL_HOUSING_STATE_FILTER=1`, preventing accidental
  reuse of unfiltered retrieval caches.
- Housing prompts now explicitly preserve the question jurisdiction and tell
  final answerers not to treat another jurisdiction's passage as controlling.

Older source-gated diagnostics already support this direction. After the
metadata casing bug was fixed, Housing `rag_state_filter` on the deterministic
N=200 slice reached 61.5% at k=5 and 62.5% at k=10 with zero empty retrieval,
versus unfiltered `rag_simple` at 53.5% k=5 and 58.0% k=10. Later archived
N=200 diagnostic runs reported `rag_state_filter` 60.5% and
`snap_hyre_state` 63.0%. Those are not part of the current comprehensive grid,
but they are strong evidence that the filter is a legitimate Housing
ablation/proposed replacement path.

Operational caution: do not build the full state-filter retrieval caches while
the active Housing Llama 70B dynamic Chroma run is resident and swap is full.
The first safe next step is a q100/q500 retrieval-only state-filter cache pass
for `rag_simple`, `rag_hyde`, `snap_hyre`, and `golden_plus_neighbors`, then a
Gemma 26B q100/q500 answer probe if the retrieval caches are clean.

- Signed full Snap-HyRE rows for BarExamQA, Legal-Link-EU, MASLegalBench, and
  HousingQA Llama 70B have zero errors, zero missing predictions, zero exact
  final-answer violations, zero truthy fallback markers, and zero think-tag
  artifacts in the checked logs.
- HousingQA Llama 8B original partial log had one no-silent-fallback violation
  on `hqa_South Carolina_4486`; this was the known row that triggered the retry
  repair patch. The repair tail completed, the merged full row is signed in
  `docs/signoff_log.md`, and the merged audit has zero exact-final violations,
  zero fallback markers, and zero cache misses.
- The HousingQA Llama 8B `rag_rewrite` row failed closed, not silently, on
  `hqa_California_5349`: the first answer said `Answer: Unknown`, the existing
  format retry preserved that invalid non-label, and `NO_SILENT_FALLBACK`
  blocked the cell with `missing_required_final_answer_line` at 555/6853 rows.
  This exposed a retry-prompt weakness for missing-prediction rows whose prior
  answer refuses to choose. The missing-prediction retry path was tightened to
  perform a logged same-model/same-evidence forced discrete repair. A one-row
  repair rerun for `hqa_California_5349` then passed cleanly with `Answer: No`.
  The completed merged row is signed in `docs/signoff_log.md`:
  4342/6853 = 63.4%, 702/6853 gold retrieved, Hit@5 0.1024 / MRR@5 0.0601,
  and 0 errors, 0 missing predictions, 0 empty retrieval rows, 0 fallback keys,
  0 think tags, and 0 near-cap rows.
- Active follow-up rows checked through the 08:33 status refresh were still
  structurally clean: HousingQA `groq-llama70b` `rag_rewrite` had 913 detail
  rows at 06:44, 1225 rows at 06:56, and 1507 combined rows at 07:07 with
  1666 combined rows at 07:13 and 1708 combined rows at 07:15; all checked
  points had 0 errors, 0 missing
  predictions, 0 fallback keys, 0 think tags, and 0 empty retrieval rows.
  At 07:21 it reached 1872 combined rows with the same clean health fields.
  At 08:36 the row failed closed with `exit=137` at 3732/6853 rows. The detail
  log through that point remained structurally clean: 0 errors, 0 missing
  predictions, 0 parse failures, 0 empty retrieval rows, 0 fallback markers,
  0 think tags, maximum output tokens 777, and 3732/3732 call-trace rows. This
  is an OOM/process-kill operational failure, not a silent fallback or answer
  corruption; the row needs a tail resume from `SAMPLE_START=3732`.
  HousingQA
  `or-gemma4-26b` `llm_only` had 287 combined prefix plus tail detail rows at
  06:44, 470 combined rows at 06:56, and 611 combined rows at 07:07 with
  706 combined rows at 07:13 and 737 combined rows at 07:15; all checked
  points had 0 errors, 0 missing
  predictions, 0 fallback keys, and expected empty retrieval rows only because
  the mode is `llm_only`. At 07:21 it reached 827 combined rows with the same
  clean health fields. HousingQA
  `or-gemma4-26b` `rag_simple` had 99 combined rows at
  06:44, 166 combined rows at 06:56, and 233 combined rows at 07:07 after a
  9-row prefix, with 272 combined rows at 07:13 and 281 combined rows at
  07:15. At 07:21 it reached 314 combined rows with the same clean health
  fields. The 9-row
  explicit `OPENROUTER_PROVIDER_ONLY=Novita` prefix was stopped for throughput
  and the tail resumed with explicit `OPENROUTER_PROVIDER_ONLY=Cloudflare`.
  The combined active prefix/tail had 0 errors, 0 missing predictions, 0 empty
  retrieval rows, 0 fallback keys, 0 think tags, and 5 retrieved passages per
  row. A first Cloudflare tail attempt failed before answer calls because
  `SAMPLE_START=9` altered `CACHE_SCOPE`; the corrected tail pins
  `CACHE_SCOPE=qfull_seed42` and reuses the full raw retrieval/doc cache.
- A fresh full-log structural scan over signed Snap-HyRE rows found 0 errors,
  0 missing predictions, 0 empty retrieval rows, 0 truthy fallback markers,
  0 think-tag artifacts, 0 HyRE cache misses, and 0 retrieval-cache misses in
  the cached Snap rows. The only recurring caveat is small-model near-cap output
  rows: Llama 8B has near-1900-token rows on BarExamQA, HousingQA,
  Legal-Link-EU, and MASLegalBench, but those rows still have exact final
  answer lines and logged same-model repair/retry metadata where needed.
- The q500/qfull real-passage exemplar generation probe remains provisional but
  is structurally clean. BarExamQA q500, HousingQA q500 canonical, HousingQA
  q500 exemplar, Legal-Link-EU q500 exemplar, and MASLegalBench qfull exemplar
  generation caches are complete. The generation caches have 0 errors,
  0 parse failures, 0 missing HyRE passages, 0 answer artifacts, and 0 think
  tags. The first script attempt was interrupted after it began BarExam
  retrieval and wrote only 2 probe retrieval rows. A retrieval-only resume was
  launched at 2026-05-20 08:40 CDT with `RUN_GENERATION=0` and completed at
  08:59 CDT. All q500/qfull retrieval probe caches have dup=0, short=0, and
  empty=0. The real-passage exemplar improves canonical Gemma Snap-HyRE
  retrieval on all four probe slices, but it remains below raw retrieval on
  Legal-Link-EU and MASLegalBench, where raw questions contain source anchors.
- Some Llama 8B rows have near-cap output token counts, but the checked signed
  rows preserve exact final answer lines. Treat these as retry/near-cap caveats,
  not silent truncation failures.
- HousingQA has many `cross_encoder_doc_truncated_count` rows because
  `CROSS_ENCODER_MAX_CHARS=4096` caps only the cross-encoder reranking input in
  `rag_utils.rerank_with_cross_encoder`; `eval/eval_harness.py` still passes
  `doc.page_content` into the final evidence prompt. This is a reranker caveat,
  not LLM evidence truncation.

## Current Operational State

- `current_status.md` was refreshed manually at 2026-05-20 09:12 CDT. That
  snapshot showed 70/78 signed, 70/78 full-row complete, 3 active, 0 partial
  stale, and 5 not started. The active rows were HousingQA
  `groq-llama70b` `rag_rewrite` at 57.8%, HousingQA `or-gemma4-26b`
  `llm_only` at 23.7%, and HousingQA `or-gemma4-26b` `rag_simple` at 11.8%.
- HousingQA `groq-llama8b` `snap_hyre` repair tail is complete and signed:
  `logs/merged/eval_snap_hyre_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl`.
- HousingQA `groq-llama8b` `rag_rewrite` is complete and signed:
  `logs/merged/eval_rag_rewrite_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl`.
- Active main-grid rows after the 09:16 check are HousingQA
  `groq-llama70b` `rag_rewrite` tail in tmux
  `housing_70b_rag_rewrite_tail_20260520_0902`, HousingQA `or-gemma4-26b`
  `llm_only` tail in tmux `housing_gemma_llm_only_tail_parasail_20260520_0614`,
  and HousingQA `or-gemma4-26b` `rag_simple` tail in tmux
  `housing_gemma_rag_simple_tail_cloudflare_doccache_20260520_0634`.
- The recurring `current_status.md` monitor was stopped after a brief restart
  because it held about 1.4 GB RSS while the HousingQA Chroma job was resident
  and swap was full. Use manual `python3 scripts/update_current_status.py`
  snapshots until one active job frees memory.
- Completed probe: Gemma 26B q500/qfull real-passage exemplar retrieval scoring
  completed in tmux `exemplar_q500_retrieval_probe_resume2_20260520_0845`.
  Exemplar Snap-HyRE improved canonical Snap-HyRE on all four retrieval slices
  but did not beat raw retrieval on Legal-Link-EU or MASLegalBench. Treat it as
  a useful analysis variant, not a canonical method replacement.

## 09:12 Fresh Failure Spot Check

After refreshing `current_status.md`, I re-ran structural checks on the active
HousingQA tails:

| Active tail | Rows checked | Errors | Missing predictions | Empty/short retrieval | Truthy fallback | Think tags | Near-cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| HousingQA `groq-llama70b` `rag_rewrite` tail | 242 | 0 | 0 | 0 | 0 | 0 | 0 |
| HousingQA `or-gemma4-26b` `llm_only` tail | 1612 | 0 | 0 | n/a | 0 | 0 | 0 |
| HousingQA `or-gemma4-26b` `rag_simple` tail | 806 | 0 | 0 | 0 | 0 | 0 | 1 |

The one Gemma `rag_simple` near-cap row still had an intact final answer line,
so this is a caveat to keep watching, not a truncation failure.

I also re-sampled raw-vs-Snap retrieval-loss rows in the two places where
Snap-HyRE is weakest against raw `rag_simple`.

Legal-Link-EU / Gemma 26B:

- On the final merged Gemma 26B logs, there were 233 rows where raw retrieval
  hit the gold evidence but Snap-HyRE missed it.
- 45 of those were high-cost rows where raw retrieval was correct and
  Snap-HyRE both missed the evidence and answered incorrectly.
- Example:
  `complex_legallink_31998R1549R(01)_31998R1549_corrects__row0165`.
  Raw retrieval hit both the corrigendum source and corrected regulation
  target and answered `C`; Snap-HyRE generated a generic corrigendum query,
  retrieved unrelated corrigenda from other acts, and answered `A`.
- Example:
  `complex_legallink_31999D0767_31999D0002_implicitly_repeals__row0175`.
  Raw retrieval hit both act anchors and answered `D`; Snap-HyRE generated a
  generic incompatibility/repeal passage, retrieved unrelated repeal snippets,
  and answered `A`.
- This reinforces the anchor-loss diagnosis: raw question text contains act IDs
  and relation-specific anchors that Snap-HyRE sometimes abstracts away.

MASLegalBench / Gemma 26B:

- On the full Gemma 26B logs, there were 79 rows where raw retrieval found a
  same-source-document passage but Snap-HyRE did not.
- Only 2 of those were high-cost answer regressions because MAS final accuracy
  is near ceiling, but they are diagnostic.
- Example: `maslb_135bfe175427134f`. Raw retrieval found Clearview source
  passages and answered `C`; Snap-HyRE generated a broad GDPR
  technical/organizational-measures query, retrieved Article 32-like snippets
  from other notices, and answered `A`.
- Example: `maslb_a66d2c47e3ac628a`. Raw retrieval found Birthlink source
  passages and answered `D`; Snap-HyRE generated a broad accountability query,
  retrieved principles/accountability passages from other notices, and answered
  `B`.
- This reinforces the source-notice-loss diagnosis: the generated legal passage
  is semantically plausible but less faithful to the benchmark's source
  document boundary.

Conclusion from the refresh: we are not cooked operationally. The current
problem is a method-shape limitation. Snap-HyRE is a good query improver when
the raw question is a weak lexical query, as in BarExamQA and parts of
HousingQA. It is risky when the raw question already carries high-value
document anchors, as in Legal-Link-EU and MASLegalBench.

## Remaining Housing/Gemma Launch Notes

As of the 09:16 status refresh, the only not-started main-grid rows are
HousingQA `or-gemma4-26b` `golden_passage`, `golden_plus_neighbors`,
`rag_hyde`, `snap_hyre`, and `rag_rewrite`. HousingQA `or-gemma4-26b`
`llm_only` and `rag_simple` are active tails, and HousingQA `groq-llama70b`
`rag_rewrite` is an active tail.

Cache state:

- Present: `caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl`.
- Present: `caches/retrieval/full/housing_qfull_seed42_golden_neighbors_k10.jsonl`.
- Present: `caches/retrieval_doc/full/housing_qfull_seed42_raw_golden_k10_doc_cache.jsonl`.
- Missing: full HousingQA `or-gemma4-26b` `rag_hyde` generation/retrieval/doc
  caches.
- Missing: full HousingQA `or-gemma4-26b` `snap_hyre` generation/retrieval/doc
  caches.

Launch discipline:

- Do not start another job while the 70B `rag_rewrite` dynamic Housing run is
  resident; it is using about half the machine RAM and swap is effectively full.
- When one slot frees, prefer finishing/signing any completed active row first.
- After the 70B Chroma-heavy row frees memory, build and audit full Gemma
  Housing `rag_hyde` and `snap_hyre` generation/retrieval caches before
  launching those answer rows.
- Golden rows can use the existing raw/golden retrieval/doc cache path and do
  not require generation caches.
