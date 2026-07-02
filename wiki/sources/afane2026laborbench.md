---
title: Benchmarking Legal RAG - AI Statutory Surveys (CSLAW 2026)
type: source
tags: [legal-rag-benchmarks, statutory-retrieval, evaluation-validity, multi-jurisdictional, commercial-legal-ai]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2603.03300
local: references/afane2026laborbench.pdf
authors: Afane et al.
year: 2026
venue: CSLAW '26 (ACM Symposium on Computer Science and Law), Berkeley, March 2026
code: none
---

# Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys

## TL;DR
Afane, Hariri, Ouyang & Ho (Stanford RegLab — the same lab behind BarExamQA/HousingQA, cited here as their "reasoning-focused legal retrieval benchmark") evaluate statutory-survey RAG on LaborBench: 1,647 boolean questions about state unemployment-insurance law across all 50 states, ground-truthed by a DOL attorney compilation. A structure-aware retrieval system (STARA: hierarchical statute parsing, definition/cross-reference augmentation, RegEx pre-filtering + LLM classification) hits 83% accuracy vs 66% for the best generic RAG from prior work, while Westlaw AI (58%) and Lexis+ AI (64%, recall 0.29) do *worse than standard RAG*. Manual verification then shows 135 of STARA's 181 apparent false positives are real statutes the DOL attorneys missed, lifting corrected accuracy to 92% — the expert "ground truth" itself is materially incomplete.

## Key claims / numbers
- Table 1: majority-class baseline acc 0.50 / F1 0.67; generic RAG 0.66 / 0.67; Westlaw AI 0.58 / 0.64; Lexis+ AI 0.64 / 0.41; STARA 0.83 / 0.81; STARA-corrected 0.92 / 0.91 (precision 0.94). *Our-relevance:* on boolean statutory tasks a yes-everything baseline gets F1 0.67 — the exact baseline-framing trap reviewers flagged in our HousingQA (yes/no) column (C9, C10).
- False-positive audit: 75% (135/181) of STARA's apparent FPs were legitimate DOL omissions vs 10% (5/47 examined) for Westlaw; Westlaw produced 596 FPs, Lexis+ 458 FNs. *Our-relevance:* gold labels in expert-built legal benchmarks are fallible, so "gold retrieved but wrong / gold missing but correct" cells (our C11 matrix) may partly reflect label error — an argument we can use and must also guard against ([[expert-judgment-replication]]).
- STARA's wins come from *structure*, not query generation: preserving statutory hierarchy, attaching definitions/cross-references/parent provisions, targeted corpus scoping; searching complete codes "consistently retrieved correct UI statutes." *Our-relevance:* this is what "incorporating legal expertise" concretely looks like to this community — a direct answer-shape for C1/C3 that SCOPE lacks.
- Commercial platform pathologies: Westlaw's 300-character query limit forces context destruction; Lexis+ favours speed over completeness (recall 0.29) and flips answers between query modes (National Guard exclusion question: 12.5% consistency; overall 50% consistency across modes, 12/24 answers changed). *Our-relevance:* deployed legal AI fails on retrieval coverage and stability, supporting our framing that retrieval exposure is a first-class outcome (C1, C5).
- Design principles (Sec. 5.1): precise question specification (temporal scope, expired provisions, exceptions); domain expertise in statutory interpretation; strategic corpus selection; transparent retrieval and citation; recognition of non-statutory authorities (regulations/policy that pure statute search structurally misses — STARA's Utah/Oregon/Nevada false negatives). *Our-relevance:* a checklist the AI4Law community will judge legal-RAG papers by; several items map onto our jurisdiction state-filter finding on HousingQA (C1, C8).
- Cost: STARA needs ~3.3 hours per full fifty-state survey question (days for the whole benchmark) vs DOL's 6 months of attorney time. *Our-relevance:* honest compute/token accounting presented as a feature, the C11 standard.

## Bearing on the review
- **C2 (insufficient legal-NLP grounding)**: CSLAW/LaborBench/STARA (plus Hariri & Ho 2025, arXiv:2508.19365, the LaborBench corpus paper) is now core legal-RAG canon from the *same lab as our benchmarks*; not citing it in a revision would repeat the exact failure mode of the rejection.
- **C10 (HousingQA regression framed as parity)**: their result that structure/metadata-aware retrieval — not generated queries — is what moves boolean statutory tasks supports reframing HousingQA honestly: strong-query statutory entailment is a regime where SCOPE-style expansion is the wrong tool and jurisdiction filtering/structure is the right one ([[regime-routing]]).
- **C8 (no corpus-level distributional analysis)**: their per-state gap analysis (Fig. 6: 0-9 missed findings per state; Michigan 9/31 questions) is the model for the jurisdiction-breakdown analysis reviewers asked us for.
- **C9 (weak-baseline framing)**: they report a majority-class baseline; our HousingQA table must too.

## Differentiation
No overlap in method: they do not study query generation, HyDE-style expansion, or query-performance prediction — STARA is retrieval-side engineering plus classification, and they run no mechanism analysis. Different task shape (50-state survey enumeration vs per-question QA). Where we are exposed rather than pre-empted: they set the community's bar for what counts as a serious statutory-RAG evaluation (expert-audited errors, corrected ground truth, commercial comparators), and our 31/42-cell matrix with no CIs falls short of it. Their ground-truth-fallibility finding cuts both ways for us: it excuses some noise in gold-based Hit@k, but obliges us to spot-audit HousingQA/BarExamQA gold labels before leaning on retrieval-exposure claims.

## Links
[[scope]], [[legal-rag-benchmarks]], [[weak-vs-strong-query-regime]], [[regime-routing]], [[expert-judgment-replication]], [[vocabulary-gap]], [[answer-conversion-gap]], [[icml-ai4law-2026-rejection]]; siblings: [[yoon2025leakage]], [[li2026legalmalr]]

## Raw source
references/afane2026laborbench.pdf (arXiv:2603.03300v1 / CSLAW '26, read pages 1-14: abstract, method, Tables 1-7, Figures 2-6, discussion; appendix statute figures skimmed)
