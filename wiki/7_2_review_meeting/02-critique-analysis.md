---
title: Critique Analysis - ICML AI4Law Rejection
type: meeting-doc
tags: [meeting, rejection, critique, scope, ai4law]
created: 2026-07-02
date: 2026-07-02
---

# Critique analysis

This dossier maps the official reviews in `assets/primary/feedback.md` to the
C1-C12 taxonomy in [[icml-ai4law-2026-rejection]], then asks what is true now.
The quoted review excerpts below are deliberately short; the full verbatim
reviews are preserved in `assets/primary/feedback.md`.

Status labels:
- **FIXED**: post-submission evidence directly answers the critique.
- **FIXABLE**: the critique can be retired by rewriting, adding a runnable
  baseline, or doing a targeted experiment.
- **REMAINS**: the problem is still substantively open for a resubmission.

## C1 - Legal reasoning and practitioner process

> "The proposed approach focuses primarily on retrieval optimization
> techniques rather than legal reasoning principles or legal practitioners'
> actual analytical processes." — Reviewer 7gGP

Reviewer right? Right for the submitted workshop fit: the paper used legal
motivation but evaluated an IR pipeline, not a practitioner-grounded legal
reasoning model [source: `assets/primary/feedback.md`; taxonomy:
[[icml-ai4law-2026-rejection]]].

Status now: **REMAINS**. The judge reranking line is the first evidence with
real legal-judgment flavor: BarExamQA trained judge Hit@5 `20.6%` vs CE
`3.8%`, with McNemar `p=1.4e-17`; Housing trained judge Hit@5 `55.0%` vs CE
`38.2%`, with `p=2.5e-23` [sources: [[judge-pilot-v0-results]],
[[judge-pilot-housing]]]. But no manuscript exists that reframes the work as
expert legal relevance judgment rather than query optimization.

What fully retires it: either choose an IR/NLP venue where legal-practitioner
process is not the central promise, or write a legal-venue paper around
learned legal relevance judgment with a lawyer/process connection and human
label plan.

## C2 - Legal-NLP literature grounding

> "The literature review is insufficient for a Legal NLP paper. Neither the
> benchmarks they used nor the baselines reflect prior research in the legal
> NLP community." — Reviewer 7gGP

Reviewer right? Right. The submitted related work missed legal generated-query
and legal passage-retrieval papers that reviewers expected [source:
`assets/primary/feedback.md`; [[icml-ai4law-2026-rejection]]].

Status now: **FIXABLE**. The wiki now has source pages for KoBLEX/ParSeR and
GuRE, and the benchmark-source page [[zheng-cslaw]] shows the benchmark
authors had already run legal generative expansion on these datasets. That
repairs the knowledge base, not the submitted manuscript.

What fully retires it: a rewritten related-work section that cites
[[koblex-parser]], [[gure]], [[zheng-cslaw]], [[hyde]], [[query2doc]], and the
legal-RAG benchmark family, then positions SCOPE as a zero-shot generated-query
member rather than a standalone legal method.

## C3 - SCOPE as HyDE-in-legal

> "The method is essentially an application of HyDE to the legal domain.
> However, it does not convincingly incorporate legal expertise, legal
> reasoning patterns, or practitioner perspectives." — Reviewer 7gGP

Reviewer right? Partially right: SCOPE is in the generated-query family; the
submitted structural delta was snap-conditioning and discard, not a new
retrieval paradigm [source: PDF Sections 2-3; [[scope]]].

Status now: **FIXABLE**. The method-superiority framing is dead, but the
mechanism framing is stronger: gold-affinity movement tracks retrieval gain
with pooled Spearman about `0.44`, and the same mechanism clears the
three-retriever check with means `0.342`, `0.354`, and `0.387` [sources:
[[affinity-margin-mechanism]], [[three-retriever-generality]]].

What fully retires it: drop "new method beats HyDE" as the primary claim.
Lead with "when generated expansion helps or hurts" and treat SCOPE/HyDE as
mechanism probes.

## C4 - LLM-generated pseudo-documents may fabricate

> "The generated hypothetical documents themselves appear likely to inherit
> the limitations of the underlying LLM, including superficial reasoning and
> potentially unsupported or fabricated legal content." — Reviewer 7gGP
> (also: "The paper highlights the limitations of LLMs in legal tasks, yet
> relies on the LLMs to generate hypothetical documents. This creates a
> inconsistency at the core of the approach.")

Reviewer right? Partially right as a safety concern, but wrong as the main
explanation for the retrieval effect now that the falsification work exists.

Status now: **FIXED** as an evidence question. Factuality was weak as a
failure predictor compared with geometry: in the full independent-judge wave,
factuality AUC was `0.548`, geometry AUC was `0.823`, and the joint model
added only `+0.003` beyond geometry [source: [[factuality-falsification]]].
The leakage audit also shows the weak-query lift survives without
gold-entailed sentences: strict all-unmatched BarExamQA rows had any-SCOPE
Hit@5 `10.5%` vs raw `1.5%`, with McNemar `p=1.1e-20` [source:
[[leakage-audit-barexam]]].

What fully retires it: include both the factuality-vs-geometry result and the
matched/unmatched leakage audit in the resubmission, with caveats about judge
coverage and BarExamQA's factuality-geometry correlation.

## C5 - Gains are marginal

> "The reported performance gains are marginal. [...] The observed
> improvements appear limited to minor variations." — Reviewer 7gGP

Reviewer right? Right for downstream answers in the submitted framing;
overstated for retrieval once benchmark-native calibration is included.

Status now: **FIXABLE**. Retrieval-side, the BarExamQA lift is not trivial:
SCOPE's submitted BarExam Hit@5 values were `9.5%`-`12.1%` against raw
`1.4%` [source: PDF Table 11], and [[zheng-cslaw]] reports the benchmark
authors' best Historical MBE Recall@10 as `6.95` on their structured expansion
setup. Answer-side, the critique remains real: the submitted BarExam average
answer change was only `72.3` LLM-only to `72.9` SCOPE [source:
[[scope-paper-2026]]]. Post-submission answer conversion is now modeled:
BarExamQA/70B judge evidence reached Hit@5 `20.6%` but answer accuracy
`75.2%`, below llm-only `77.7%` and nonsignificant; HousingQA/70B converted,
with judge evidence `65.6%` vs llm-only `54.2%`, `p=5.5e-08` [source:
[[judge-answer-conversion]]].

What fully retires it: make retrieval exposure, selection, and answer
conversion separate claims. Do not sell BarExamQA/70B as answer-positive.

## C6 - Uncited near-twin ParSeR/KoBLEX

> "This is the same idea as ParSeR in KoBLEX (EMNLP 2025), which generates
> 'parametric provisions' [...] The mapping is close to one-to-one [...] But
> the paper cites neither KoBLEX nor any legal generated-query method of this
> kind, so the novelty claim is untested against the closest prior art."
> — Reviewer oSUu, W1

Reviewer right? Right. [[koblex-parser]] confirms the core overlap:
parametric statute-like text is generated as a retrieval query in legal QA.

Status now: **FIXABLE** but not fixed in manuscript. The KoBLEX page reports
ParSeR's GPT-4o provision-retrieval F-1 `59.41` vs one-time retrieval `21.50`,
and answer-side LF-Eval `67.26` vs `36.45` [source: [[koblex-parser]]]. Those
numbers are stronger than the submitted SCOPE answer story and must change
the novelty claim.

What fully retires it: cite KoBLEX as closest prior art, concede overlap on
generated legal provisions, and position the surviving contribution as
mechanism/regime/selection analysis rather than pseudo-document novelty.

## C7 - Snap-conditioning not shown to beat HyDE

> "The novelty that does remain, the snap-answer conditioning that separates
> SCOPE from HyDE (Eq. 3 vs Eq. 6), shows no measurable benefit [...] No
> significance test is reported for any SCOPE-versus-HyDE comparison [...]
> The method is thus squeezed between a near-identical uncited legal method
> and a general baseline it does not improve on." — Reviewer oSUu, W1

Reviewer right? Right. The submitted paper did not prove that the snap step
was doing meaningful work beyond HyDE [source: PDF Table 6; feedback review].

Status now: **FIXED** as a diagnostic, but negative for the original claim.
The compiled answer ledger finds `13/16` full-N SCOPE-vs-HyDE pairs
nonsignificant. The significant answer cells split direction: Legal-Link-EU
Gemma is pro-snap by `+4.17pp` with `p=0.00361`, while HousingQA Llama-70B is
pro-HyDE by `-6.45pp` unfiltered with `p=1.4e-28` and `-2.57pp`
state-filtered with `p=1.7e-06` [source: [[snap-vs-hyde-ledger]]]. Retrieval
significance also cuts against the old BarExam snap-win story: BarExamQA is
SCOPE approximately equal to HyDE in `3/4` tested models, while BEIR shows
the surviving snap benefit is drift robustness, with `19/20` strong-query
cells significant and gains of `+16` to `+45pp` over HyDE [source:
[[snap-vs-hyde-ledger]], [[beir-phase1]]].

What fully retires it: stop claiming snap beats HyDE on answers. Reframe as
answer-equivalent on weak queries and more drift-robust on strong queries,
then run the C12 ablations to explain why.

## C8 - Missing trained-rewriter and distributional analysis

> "GuRE (Kim et al., NLLP 2025) trains a generative query rewriter to
> mitigate vocabulary mismatch in legal passage retrieval, the same problem
> framed the same way [...] SCOPE neither compares against a trained query
> rewriter nor offers any distributional characterization of its corpora."
> — Reviewer oSUu, W2

Reviewer right? Right. GuRE and corpus-frequency analysis are exactly the
legal-IR comparator and analysis layer the paper lacked [source:
`assets/primary/feedback.md`; [[gure]]].

Status now: **REMAINS**. The source page is written, but no GuRE-style
baseline has been run on BarExamQA/HousingQA, and no long-tail or jurisdiction
breakdown has been added to a manuscript. GuRE's own BM25 nDCG@10 jump from
`15.33` to `47.69` on LePaRD shows the supervised tier is a serious skyline,
not a citation formality [source: [[gure]]].

What fully retires it: run or explicitly scope a GuRE-style supervised skyline,
and add passage-frequency/jurisdiction stratification for where generated
queries help or hurt.

## C9 - Headline used the weakest answer baseline

> "raw-question RAG is the worst method on BarExamQA (average 69.0, Table 1),
> below LLM-only (72.3) [...] An eightfold Hit@5 improvement (1.4% to about
> 12%) moves average answer accuracy only from 72.3 to 72.9, indicating
> answers are driven by parametric knowledge, not retrieved evidence."
> — Reviewer oSUu, W3

Reviewer right? Right. Raw-question RAG was below LLM-only on submitted
BarExamQA, so the paper's answer framing overstated the result [source:
PDF Table 1; [[scope-paper-2026]]].

Status now: **REMAINS** for any resubmission until the tables are rewritten.
The submitted table gives BarExam averages of `69.0` for raw-question RAG,
`72.3` for LLM-only, and `72.9` for SCOPE [source: PDF Table 1]. Full-N MedQA
reinforces the law: Llama-70B llm_only was `85.6%`, raw RAG hurt at `83.1%`,
HyDE was `85.2%`, and SCOPE was `86.1%`, with SCOPE vs llm_only only
`+0.55pp` and nonsignificant [source: [[medqa-fulln-matrix]]].

What fully retires it: make LLM-only the primary answer baseline everywhere;
use raw RAG as a retrieval-policy comparator, not the denominator for the
headline answer claim.

## C10 - HousingQA regression was spun as parity

> "On HousingQA the method regresses rather than reaches 'parity': SCOPE is
> 59.0 and 59.6 versus raw-question RAG 62.3 and 62.1 (Table 1; pooled
> -2.9pp, Table 6), and is the weakest non-LLM method in that column."
> — Reviewer oSUu, W3

Reviewer right? Right. The submitted HousingQA answer rows were worse for
SCOPE than raw-question RAG [source: PDF Table 1].

Status now: **FIXABLE**. The regression now has a coherent regime account:
strong-query expansion drifts because the raw query is already corpus-shaped
[sources: [[query-drift]], [[weak-vs-strong-query-regime]], [[beir-phase1]]].
The Housing judge results also show how to handle the strong regime: pool plus
trained judge produced Hit@5 `55.0%` and answer accuracy `65.6%`, beating
llm_only `54.2%` and CE evidence `61.8%` on the same held-out subset [sources:
[[judge-pilot-housing]], [[judge-answer-conversion]]].

What fully retires it: present Housing as the predicted failure mode of
ungated expansion, then show the selector/pooling fix. Do not call the
submitted SCOPE rows parity.

## C11 - Rigor gaps: incomplete matrix, intervals, cost accounting

> "the evaluation matrix is one-quarter empty (31 of 42 cells, HousingQA 10
> of 21 [...]) and confidence intervals are reported on no answer-accuracy
> number [...] the 'leading answer-token efficiency' claim (Section 5.5,
> Table 3) excludes first-stage query-generation tokens [...] which flatters
> SCOPE because its first call emits both a snap answer and a passage."
> — Reviewer oSUu, Comments

Reviewer right? Right. The submitted appendix disclosed `31/42` signed cells,
and Table 3 excluded first-stage generation tokens [source: PDF Appendix A.1
and Table 3].

Status now: **FIXABLE**. Retrieval-side significance has improved: the
compiled sweep reports `97/128` retrieval pairs significant, and the
snap-vs-HyDE ledger now separates answer-side and retrieval-side tests
[source: [[snap-vs-hyde-ledger]]]. But the full revised answer matrix with
confidence intervals, total-pipeline token accounting, and inclusion rules
does not exist yet.

What fully retires it: every headline answer delta gets paired McNemar and
confidence intervals; every retrieval delta gets paired testing or bootstrap
intervals; all token tables include stage-one and stage-two costs.

## C12 - Guardrail asserted but not ablated

> "the confirmation-bias guardrail (Eq. 7) is asserted rather than tested; an
> ablation that passes the snap answer into the answer call and shows it
> hurts would convert the rationale into evidence." — Reviewer oSUu, Comments

Reviewer right? Right. The submitted paper argued that withholding the snap
answer avoided anchoring, but did not test the alternative [source: PDF
Section 3.3; `assets/primary/feedback.md`].

Status now: **REMAINS**. Adjacent evidence exists: [[zheng-cslaw]] reports
rollout-as-pseudo-passage can hurt BarExamQA answers, and [[scope]] notes the
repo's sanitizer finding that showing the answer letter hurt. But the exact
C12 ablation has not been run.

What fully retires it: run the small ablation trio: pass `a0` to the answer
call, keep or concatenate the raw question Query2doc-style, and generate a
ParSeR-style conclusion-banned passage. Report both retrieval margin and
answer outcomes.

## Scorecard

| Critique | Reviewer verdict | Status now | Evidence link |
|---|---|---|---|
| C1 legal reasoning/process | right | REMAINS | [[judge-pilot-v0-results]], [[judge-pilot-housing]] |
| C2 legal-NLP grounding | right | FIXABLE | [[koblex-parser]], [[gure]], [[zheng-cslaw]] |
| C3 HyDE-in-legal | partially right | FIXABLE | [[affinity-margin-mechanism]], [[three-retriever-generality]] |
| C4 fabrication risk | partially right, mechanism wrong | FIXED | [[factuality-falsification]], [[leakage-audit-barexam]] |
| C5 marginal gains | right for answers, weak for retrieval | FIXABLE | [[judge-answer-conversion]], [[zheng-cslaw]] |
| C6 ParSeR overlap | right | FIXABLE | [[koblex-parser]] |
| C7 snap vs HyDE | right | FIXED | [[snap-vs-hyde-ledger]], [[beir-phase1]] |
| C8 trained rewriter/distribution | right | REMAINS | [[gure]] |
| C9 weakest baseline | right | REMAINS | [[scope-paper-2026]], [[medqa-fulln-matrix]] |
| C10 Housing regression | right | FIXABLE | [[query-drift]], [[judge-pilot-housing]] |
| C11 rigor/cost gaps | right | FIXABLE | [[snap-vs-hyde-ledger]] |
| C12 guardrail ablation | right | REMAINS | [[scope]], [[zheng-cslaw]] |

Sources: `assets/primary/feedback.md`, [[icml-ai4law-2026-rejection]], and
the evidence pages linked in the table.

## Ranked top 3 things that must change

1. **The flagship claim must change from method win to mechanism law.**
   The submitted "SCOPE beats raw RAG" story is too weak because raw RAG is
   often the wrong answer baseline and SCOPE is not reliably better than HyDE.
   The stronger claim is in [[thesis-v2]]: expansion helps when query-gold
   margin is low, selection binds after pooling, and answer conversion depends
   on reader-task evidence value.

2. **The legal-prior-art section must be rebuilt before any legal venue.**
   KoBLEX/ParSeR, GuRE, and Zheng et al. are not optional citations. They
   define the prior art, the supervised skyline, and the benchmark-native
   calibration. Without them, C2, C6, and C8 return immediately.

3. **The answer-side story must become honest and predictive.**
   On parametric-strong multiple-choice tasks, better evidence can fail to
   improve answers: BarExamQA/70B judge evidence had higher Hit@5 but lower
   answer accuracy than llm_only [source: [[judge-answer-conversion]]]. On
   evidence-valuable statutory entailment, better evidence converts. A
   resubmission should lead with that law, not hide negative answer cells.

## Meeting readout

The rejection was justified against the submitted paper. It does not kill the
research program. It kills the method-novelty framing and the raw-RAG
headline. The live path is a mechanism and selection paper, with a legal
companion only after the prior art, baselines, and ablations are actually in
place.

