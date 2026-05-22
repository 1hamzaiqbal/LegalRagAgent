# Retrieval Passage Exemplar Probe - 2026-05-20

Status: proposal / mini-ablation candidate. Not part of the canonical
comprehensive grid unless explicitly promoted after smoke tests.

Latest hardened probe snapshot:

- 2026-05-20, q20 per benchmark, `or-gemma4-26b`, actual corpus-passage
  exemplar excerpts with associated question, answer, and document id removed.
- Generation cache health for `snap_hyre_exemplar`: 20/20 rows on BarExamQA,
  HousingQA, Legal-Link-EU, and MASLegalBench; no errors, fallback keys, answer
  artifacts, think tags, parse failures, or missing snap letters.
- Retrieval cache health: 20/20 rows for every scored cache; no duplicate keys,
  missing ids, empty retrieval, or short rows.

| Benchmark | Metric | Raw question | Canonical `snap_hyre` | `snap_hyre_exemplar` |
|---|---:|---:|---:|---:|
| BarExamQA | Hit@5 / MRR@5 | 0.0000 / 0.0000 | 0.0500 / 0.0250 | 0.1500 / 0.0850 |
| HousingQA | Hit@5 / MRR@5 | 0.0000 / 0.0000 | 0.0500 / 0.0500 | 0.1000 / 0.1000 |
| Legal-Link-EU | Hit@5 / MRR@5 | 0.8500 / 0.7167 | 0.7500 / 0.5808 | 0.7500 / 0.7000 |
| MASLegalBench | same-source@5 / MRR@5 | 0.7000 / 0.5892 | 0.4000 / 0.2475 | 0.5000 / 0.2908 |

Interpretation: the real-passage exemplar improved over canonical
`snap_hyre` on all four q20 slices at k=5/MRR, but it did not beat raw-question
retrieval on Legal-Link-EU or MASLegalBench. Treat it as a promising tentative
Snap-HyRE variant that needs q20/q50 downstream answer checks before promotion
to the comprehensive grid.

Answer-side q20 check: BarExamQA `or-gemma4-26b` `snap_hyre_exemplar` reached
19/20 = 95.0% with zero errors, fallback keys, parse failures, answer
artifacts, or think tags. The single miss was `qa_nan_mbe_58`: the generated
query retrieved gold (`mbe_49`) and the snap prior selected the gold answer, but
the final answerer chose the wrong option. That is an evidence-utilization /
option-mapping miss, not a retrieval miss.

A q500 retrieval-scale probe has completed the LLM-generation side. The
generation caches are present and clean for BarExamQA q500, HousingQA q500
canonical Snap-HyRE, HousingQA q500 real-passage exemplar, Legal-Link-EU q500
real-passage exemplar, and MASLegalBench full real-passage exemplar: no errors,
parse failures, missing HyRE passages, answer artifacts, or think tags. The
first retrieval-build attempt was interrupted during the first raw BarExamQA
retrieval call after loading the embedding stack and wrote only two raw q500
retrieval rows. A retrieval-only resume was launched at 2026-05-20 08:40 CDT
with `RUN_GENERATION=0` so it should not make additional API calls. BarExamQA
q500 retrieval has now landed cleanly:

| Benchmark | Method | N | Hit@5 | MRR@5 | Hit@10 | Cache health |
|---|---|---:|---:|---:|---:|---|
| BarExamQA | raw question | 500 | 0.0160 | 0.0082 | 0.0280 | dup=0, short=0, empty=0 |
| BarExamQA | canonical Gemma Snap-HyRE | 500 | 0.1300 | 0.0644 | 0.1820 | dup=0, short=0, empty=0 |
| BarExamQA | real-passage exemplar Snap-HyRE | 500 | 0.1360 | 0.0628 | 0.2000 | dup=0, short=0, empty=0 |
| HousingQA | raw question | 500 | 0.0260 | 0.0172 | 0.0500 | dup=0, short=0, empty=0 |
| HousingQA | canonical Gemma Snap-HyRE | 500 | 0.0740 | 0.0425 | 0.1160 | dup=0, short=0, empty=0 |
| HousingQA | real-passage exemplar Snap-HyRE | 500 | 0.0840 | 0.0429 | 0.1220 | dup=0, short=0, empty=0 |
| Legal-Link-EU | raw question | 500 | 0.9000 | 0.7572 | 0.9540 | dup=0, short=0, empty=0 |
| Legal-Link-EU | canonical Gemma Snap-HyRE | 500 | 0.6820 | 0.5559 | 0.7720 | dup=0, short=0, empty=0 |
| Legal-Link-EU | real-passage exemplar Snap-HyRE | 500 | 0.7580 | 0.6257 | 0.8420 | dup=0, short=0, empty=0 |
| MASLegalBench | raw question same-source proxy | 303 | 0.7261 | 0.6277 | 0.7888 | dup=0, short=0, empty=0 |
| MASLegalBench | canonical Gemma Snap-HyRE same-source proxy | 303 | 0.3531 | 0.2182 | 0.6106 | dup=0, short=0, empty=0 |
| MASLegalBench | real-passage exemplar Snap-HyRE same-source proxy | 303 | 0.4257 | 0.2472 | 0.6832 | dup=0, short=0, empty=0 |

The q500 result confirms the main BarExam/Housing retrieval narrative, but the
exemplar gain over canonical Snap-HyRE is modest. On BarExamQA it is +0.6pp
Hit@5 and +1.8pp Hit@10, with slightly lower MRR@5. On HousingQA it is +1.0pp
Hit@5, +0.04pp MRR@5, and +0.6pp Hit@10. On Legal-Link-EU the exemplar signal
is more useful, improving canonical Snap-HyRE by +7.6pp Hit@5 and +7.0pp MRR@5,
but it still remains far below raw question retrieval because raw carries
source/target act anchors. On MASLegalBench it also improves canonical
Snap-HyRE by +7.3pp same-source@5, but remains far below raw because raw
question text preserves source-notice anchors. Overall, exemplar prompting looks
like a useful analysis variant, not a clean canonical replacement.

Implementation note for any q500 answer-side follow-up: `snap_hyre_exemplar`
and `rag_hyde_exemplar` now support strict HyRE generation-cache replay, and
`NO_SILENT_FALLBACK` treats supplied exemplar generation/retrieval caches as
required. This means a q500 answer probe can use the already-built q500
generation and retrieval caches instead of regenerating hidden queries or
falling back to dynamic retrieval. It still needs document hydration for the
retrieved ids, ideally through a retrieval document cache once the active
Housing Chroma-heavy runs free memory.

2026-05-20 15:10 CDT follow-up: prioritized tmux session
`exemplar_answer_gemma26_priority_20260520_151049` ran cached
`snap_hyre_exemplar` answer probes for BarExamQA q500. It stopped at 55/500
because the old artifact detector falsely treated ordinary prose like "it is a
fair representation" as answer-letter leakage. The parsed HyRE passage was
clean, so this was a guard false positive rather than a method leak. The guard
has been tightened in `eval/eval_harness.py` and `scripts/analyze_detail_flags.py`.
The prefix is useful as a provisional answer-side signal: on the same 55 labels,
`snap_hyre_exemplar` reached 51/55, versus 49/55 for canonical `snap_hyre`,
48/55 for `rag_hyde`, 48/55 for `rag_rewrite`, and 46/55 for `rag_simple`.

2026-05-20 15:44 CDT follow-up: resumed tmux session
`exemplar_answer_gemma26_priority_resume_20260520_154427` from BarExam row 54,
then Legal-Link-EU q500, then MASLegalBench full via
`scripts/local/run_snap_hyre_exemplar_answer_probe.sh`. The lower-priority
Housing Gemma `llm_only` job remains paused at 3,678 total detail rows to free
the OpenRouter/Gemma lane. The cache hydration preflight passed 20/20 cache
hits for all three retrieval caches. HousingQA is excluded from this queue
because the available q500 exemplar cache is unfiltered provenance, while the
main Housing matrix now requires state-filtered retrieval.

Raw-retrieval diagnosis from the signed/full caches:

- BarExamQA `rag_simple` is not a harness failure. `scripts/audit_retrieval_id_alignment.py`
  found 1149/1149 unique full-row gold ids in Chroma. On the full corpus, raw
  question retrieval has Hit@1 0.0033 / Hit@5 0.0142 / Hit@10 0.0218. The gold
  ids are all `mbe_*`, but raw retrieval top-1 is `caselaw_*` on 1063/1195 rows
  and at least one `caselaw_*` appears in the top five on 1182/1195 rows. The
  top five are all case-law passages on 881/1195 rows. In the same q500 slice
  used for the exemplar probe, raw Hit@5 is 0.0160 while canonical Gemma
  Snap-HyRE Hit@5 is 0.1300 and Gemma HyDE Hit@5 is 0.1220. The practical
  failure mode is that MBE fact patterns embed like ordinary litigation facts,
  while the gold passages are compact doctrine/rule statements.
- HousingQA `rag_simple` is also a retrieval-shape problem, not silent
  fallback. `scripts/audit_retrieval_id_alignment.py` found 990/990 unique
  full-row gold ids in Chroma. On the full corpus, raw question retrieval has
  Hit@1 0.0079 / Hit@5 0.0282 / Hit@10 0.0511. Only 2.3% of rows have a
  same-state statute at rank 1, only 9.4% have any same-state statute in the
  top five, and 90.6% both miss the gold id and have no same-state statute in
  the top five. Dense retrieval is matching generic landlord-tenant language
  across jurisdictions instead of preserving the requested state. On the q500
  slice, raw Hit@5 is 0.0260; Llama 70B Snap-HyRE reaches 0.0980 and Llama 70B
  HyDE reaches 0.1740. HyDE improves this by naming state-specific statutory
  hooks more often; Snap-HyRE helps over raw but still trails HyDE on exact
  statute matching.

## Failure Notes

The q20 misses do not indicate a silent fallback or cache bug. They show two
different retrieval regimes:

- BarExamQA and HousingQA raw-question retrieval is weak because the raw query
  often embeds fact-pattern wording or generic statutory terms. BarExam raw
  retrieved case paragraphs about the fact pattern while the gold was compact
  black-letter doctrine. Housing raw retrieved cross-jurisdiction statutes with
  similar wording while missing the specific state statute. The real-passage
  exemplar helped by pushing Snap-HyRE toward doctrine/statute-shaped queries.
- Legal-Link-EU and MASLegalBench raw-question retrieval is already strong
  because the question text contains document identifiers, relation names, or
  case/source cues. Generated Snap queries can abstract away those anchors and
  lose some raw retrieval wins. The exemplar recovers some MRR but should not be
  described as dominating raw retrieval on those datasets.
- BarExamQA q20 answer-side probe reached 19/20. The single miss had improved
  retrieval and a correct snap prior, but the final answerer selected the wrong
  option. That is evidence-utilization / option-mapping risk, not retrieval
  failure.

## Motivation

Current Snap-HyRE already sometimes improves retrieval by converting the
question into a law-like passage before embedding search. A small prompt
variant could test whether showing the generator one dataset-specific example
of a useful retrievable passage improves that behavior.

The goal is retrieval exposure, not answer leakage:

- Teach the model what a useful corpus passage looks like.
- Do not provide row-specific evidence.
- Do not provide gold labels, gold passages, or answer letters for the current
  row.
- Keep the method clearly separate from canonical `snap_hyre`.

## Candidate Variant

Implemented probe mode: `snap_hyre_exemplar`.

Prompt change:

- Add one fixed, dataset-specific real corpus-style passage before the
  Snap-HyRE generation task.
- The passage has the associated question, answer, and document id removed.
- For q20/q50 mini runs, use samples that do not include the exemplar's
  associated question when that association is known.
- The final generated HyRE passage must still be based only on the current
  question and snap reasoning.
- Retrieval and final answer stages remain unchanged.

Control variant:

- `rag_hyde_exemplar`, using the same style signal but no snap reasoning.
- This separates "better passage style" from "snap-conditioned reasoning".

Implementation guard:

- The signal uses one real passage-style exemplar per dataset, not retrieved
  passages for the current row and not row-specific answer evidence.
- Canonical `rag_hyde` and `snap_hyre` prompts are unchanged.
- These modes are intentionally absent from `current_status.md` and the main
  comprehensive grid unless explicitly promoted later.

## Initial Log Examples

These examples were pulled from existing signed/full logs to show why the
variant is plausible.

### BarExamQA, Gemma

Row: `qa_nan_mbe_995`.

Raw `rag_simple` missed gold and answered wrong. Its top passages were generic
back-injury medical-history facts.

`snap_hyre` generated this retrieval query:

> Statements made by a declarant for the purpose of diagnosing or treating the
> declarant's medical condition are admissible as an exception to the hearsay
> rule...

That retrieved the medical-diagnosis hearsay exception, including `mbe_2703`,
and answered correctly.

### Legal-Link-EU, Llama 70B

Row:
`complex_legallink_32023R2081_article_2f_32014R0833_extends_application__row1115`.

Raw `rag_simple` retrieved generic purchase/intervention material and missed
the gold-linked sanctions regulation.

`snap_hyre` generated a query about EU restrictive measures, Russian entities,
subsidiaries, transferable securities, and maturity period. It retrieved
Council Regulation (EU) No 833/2014 and answered correctly.

### HousingQA, Llama 70B

Row: `hqa_Wisconsin_8622`.

Raw `rag_simple` retrieved generic restitution material from other
jurisdictions and answered wrong.

`snap_hyre` generated a warrant-of-restitution legal-definition passage. It
retrieved Wisconsin `799.44 Order for judgment; writ of restitution` and
answered correctly.

## Suggested Mini Test

Run a retrieval-only or answer-light q20/q50 probe before any full launch:

- Models: `or-gemma4-26b` first.
- Datasets: BarExamQA, Legal-Link-EU, HousingQA, MASLegalBench.
- Modes: `rag_hyde`, `rag_hyde_exemplar`, `snap_hyre`, `snap_hyre_exemplar`.
- Metrics: Hit@5 / MRR@5 where gold ids exist, source-proxy retrieval for
  MASLegalBench, answer accuracy only if the generated caches look clean.

Promotion gate:

- No fallback keys.
- No answer artifacts in generated passages.
- No think tags.
- No parse failures.
- Retrieval improves over canonical `snap_hyre` on at least two datasets or
  gives a clearly interpretable win without hurting answer accuracy in a q50
  answer probe.
