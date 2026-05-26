# SCOPE Query-Gold Gap Mechanism - 2026-05-25

## Scope

This results-lane analysis measures the query-to-gold-passage gap directly using the real retrieval model families: MiniLM cross-encoder scores and gte-large bi-encoder cosine. It uses signed raw/SCOPE caches, canonical `snap_hyre` generated-passage caches, and fetched gold passage text. No files under `paper/` were edited.

- Multi-gold rows use the maximum score over the gold set for raw and SCOPE independently.
- CE inputs use `CROSS_ENCODER_MAX_CHARS=4096` by default, matching the retrieval-cache reranker cap used in these analyses.
- Outcomes are signed as SCOPE minus raw: retrieval delta is Hit@5 movement; answer delta is exact-answer correctness movement.

## Summary

| Dataset | Model | N | Multi-gold rows | Mean CE delta | CE delta > 0 | Mean cos delta | Cos delta > 0 | Mean retrieval delta | Mean answer delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | 1195 | 0.0% | 4.266 | 78.6% | 0.080 | 77.3% | 8.1% | 2.4% |
| BarExamQA | Gemma 4 26B | 1195 | 0.0% | 3.885 | 75.4% | 0.100 | 85.5% | 10.6% | 4.0% |
| BarExamQA | Groq Llama 70B | 1195 | 0.0% | 4.760 | 86.0% | 0.094 | 84.8% | 9.6% | 5.2% |
| BarExamQA | Pooled | 3585 | 0.0% | 4.304 | 80.0% | 0.091 | 82.5% | 9.5% | 3.9% |
| HousingQA state-filtered | Groq Llama 8B | 6853 | 48.8% | 3.219 | 81.2% | 0.032 | 69.5% | -7.4% | -3.3% |
| HousingQA state-filtered | Gemma 4 26B | 6853 | 48.8% | 2.990 | 79.2% | 0.032 | 70.8% | 1.1% | -1.1% |
| HousingQA state-filtered | Groq Llama 70B | 6853 | 48.8% | 3.452 | 82.5% | 0.020 | 60.5% | -13.8% | -2.5% |
| HousingQA state-filtered | Pooled | 20559 | 48.8% | 3.221 | 80.9% | 0.028 | 66.9% | -6.7% | -2.3% |

## Correlations

H1 expects low `CE(raw,gold)` / `cos(raw,gold)` to associate with SCOPE gains, so negative correlations with SCOPE-minus-raw retrieval delta support H1. H2 expects positive deltas to associate with retrieval gain. H3 asks whether any axis predicts answer delta.

| Dataset | Model | Axis | N | Pearson retrieval | Spearman retrieval | Pearson answer | Spearman answer |
|---|---|---|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | CE(raw, gold) | 1195 | -0.009 | -0.011 | 0.004 | 0.010 |
| BarExamQA | Groq Llama 8B | cos(raw, gold) | 1195 | -0.014 | -0.013 | -0.012 | -0.013 |
| BarExamQA | Groq Llama 8B | CE(scope, gold) - CE(raw, gold) | 1195 | 0.342 | 0.343 | 0.008 | 0.012 |
| BarExamQA | Groq Llama 8B | cos(scope, gold) - cos(raw, gold) | 1195 | 0.351 | 0.340 | 0.029 | 0.037 |
| BarExamQA | Gemma 4 26B | CE(raw, gold) | 1195 | 0.027 | 0.020 | -0.031 | -0.020 |
| BarExamQA | Gemma 4 26B | cos(raw, gold) | 1195 | 0.080 | 0.083 | -0.014 | -0.011 |
| BarExamQA | Gemma 4 26B | CE(scope, gold) - CE(raw, gold) | 1195 | 0.342 | 0.354 | 0.059 | 0.060 |
| BarExamQA | Gemma 4 26B | cos(scope, gold) - cos(raw, gold) | 1195 | 0.314 | 0.318 | 0.051 | 0.057 |
| BarExamQA | Groq Llama 70B | CE(raw, gold) | 1195 | 0.023 | 0.024 | 0.023 | 0.027 |
| BarExamQA | Groq Llama 70B | cos(raw, gold) | 1195 | -0.018 | -0.005 | -0.032 | -0.028 |
| BarExamQA | Groq Llama 70B | CE(scope, gold) - CE(raw, gold) | 1195 | 0.315 | 0.327 | -0.021 | -0.026 |
| BarExamQA | Groq Llama 70B | cos(scope, gold) - cos(raw, gold) | 1195 | 0.325 | 0.327 | -0.007 | -0.016 |
| BarExamQA | Pooled | CE(raw, gold) | 3585 | 0.014 | 0.012 | -0.001 | 0.007 |
| BarExamQA | Pooled | cos(raw, gold) | 3585 | 0.017 | 0.023 | -0.018 | -0.016 |
| BarExamQA | Pooled | CE(scope, gold) - CE(raw, gold) | 3585 | 0.331 | 0.340 | 0.017 | 0.017 |
| BarExamQA | Pooled | cos(scope, gold) - cos(raw, gold) | 3585 | 0.330 | 0.330 | 0.026 | 0.028 |
| HousingQA state-filtered | Groq Llama 8B | CE(raw, gold) | 6853 | -0.230 | -0.243 | -0.027 | -0.024 |
| HousingQA state-filtered | Groq Llama 8B | cos(raw, gold) | 6853 | -0.188 | -0.253 | 0.005 | -0.005 |
| HousingQA state-filtered | Groq Llama 8B | CE(scope, gold) - CE(raw, gold) | 6853 | 0.457 | 0.444 | 0.059 | 0.055 |
| HousingQA state-filtered | Groq Llama 8B | cos(scope, gold) - cos(raw, gold) | 6853 | 0.387 | 0.375 | 0.032 | 0.030 |
| HousingQA state-filtered | Gemma 4 26B | CE(raw, gold) | 6853 | -0.211 | -0.214 | -0.033 | -0.032 |
| HousingQA state-filtered | Gemma 4 26B | cos(raw, gold) | 6853 | -0.116 | -0.172 | 0.003 | -0.006 |
| HousingQA state-filtered | Gemma 4 26B | CE(scope, gold) - CE(raw, gold) | 6853 | 0.530 | 0.504 | 0.088 | 0.082 |
| HousingQA state-filtered | Gemma 4 26B | cos(scope, gold) - cos(raw, gold) | 6853 | 0.401 | 0.395 | 0.065 | 0.062 |
| HousingQA state-filtered | Groq Llama 70B | CE(raw, gold) | 6853 | -0.302 | -0.320 | -0.036 | -0.035 |
| HousingQA state-filtered | Groq Llama 70B | cos(raw, gold) | 6853 | -0.238 | -0.306 | -0.009 | -0.018 |
| HousingQA state-filtered | Groq Llama 70B | CE(scope, gold) - CE(raw, gold) | 6853 | 0.443 | 0.440 | 0.083 | 0.080 |
| HousingQA state-filtered | Groq Llama 70B | cos(scope, gold) - cos(raw, gold) | 6853 | 0.329 | 0.308 | 0.086 | 0.085 |
| HousingQA state-filtered | Pooled | CE(raw, gold) | 20559 | -0.245 | -0.257 | -0.031 | -0.030 |
| HousingQA state-filtered | Pooled | cos(raw, gold) | 20559 | -0.179 | -0.242 | -0.000 | -0.010 |
| HousingQA state-filtered | Pooled | CE(scope, gold) - CE(raw, gold) | 20559 | 0.468 | 0.453 | 0.074 | 0.070 |
| HousingQA state-filtered | Pooled | cos(scope, gold) - cos(raw, gold) | 20559 | 0.379 | 0.366 | 0.059 | 0.056 |

## CE Delta Binned Curve

Quintiles are pooled across dataset/model rows and sorted by `CE(scope,gold) - CE(raw,gold)`.

| Bin | N | CE delta median | CE delta range | CE delta > 0 | Cos delta median | Net retrieval delta | Net answer delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4829 | -1.515 | -15.575-0.112 | 4.0% | -0.004 | -36.4% | -6.4% |
| 2 | 4829 | 1.199 | 0.113-2.181 | 100.0% | 0.020 | -16.1% | -2.5% |
| 3 | 4828 | 3.184 | 2.181-4.203 | 100.0% | 0.034 | -4.5% | -0.9% |
| 4 | 4829 | 5.343 | 4.203-6.740 | 100.0% | 0.050 | 8.7% | 0.4% |
| 5 | 4829 | 8.667 | 6.741-17.171 | 100.0% | 0.087 | 26.9% | 2.6% |

## Masked-LM Pseudo-Perplexity

Blocked/provisional: no candidate masked-LM was available in the local Hugging Face cache under HF_HUB_OFFLINE. Attempted: prajjwal1/bert-tiny: OSError, distilbert-base-uncased: OSError, bert-base-uncased: OSError, google/bert_uncased_L-2_H-128_A-2: OSError.

## Top CE-Delta Examples

| Dataset | Model | Label | CE delta | Cos delta | Outcomes | Raw question | SCOPE passage | Gold passage |
|---|---|---|---:|---:|---|---|---|---|
| HousingQA state-filtered | Groq Llama 8B | `hqa_Florida_4051` | 17.171 | 0.146 | ret 0->1; ans 1->1 | Are eviction cases first heard in justice of the peace court? | Florida's county courts have jurisdiction over civil cases, including those involving the possession of real property, such as eviction p... | 34.011 Jurisdiction in landlord and tenant cases.—(1) The county court shall have jurisdiction concurrent with the circuit court to consi... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Oklahoma_4427` | 17.001 | 0.205 | ret 0->1; ans 1->1 | Are eviction cases first heard in court of common pleas? | Oklahoma district courts have jurisdiction over eviction proceedings, also known as forcible entry and detainer actions. These courts hea... | \|The district court shall have jurisdiction to try all actions for the forcible entry and detention, or detention only, of real property... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Hawaii_4092` | 16.924 | 0.090 | ret 0->1; ans 1->1 | Are eviction cases first heard in superior court? | The district courts of the State have jurisdiction in all civil actions and proceedings, except as otherwise provided by law, including a... | §604-5 Civil jurisdiction. (a) Except as otherwise provided, the district courts shall have jurisdiction in all civil actions where the d... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Michigan_4237` | 16.709 | 0.185 | ret 0->1; ans 1->1 | Are eviction cases first heard in justice court? | Michigan district courts have jurisdiction over eviction cases, which are summary proceedings to recover possession of premises. The cour... | 600.5704 Jurisdiction. Sec. 5704. The district court, municipal courts and the common pleas court of Detroit have jurisdiction over summa... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Oklahoma_4428` | 16.663 | 0.127 | ret 0->1; ans 0->1 | Are eviction cases first heard in justice of the peace court? | Eviction proceedings in Oklahoma are generally governed by the Oklahoma Residential Landlord and Tenant Act, which provides that eviction... | \|The district court shall have jurisdiction to try all actions for the forcible entry and detention, or detention only, of real property... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Missouri_4942` | 16.175 | 0.169 | ret 0->1; ans 1->1 | Primary methods of service are defined as those methods that must be attempted before a secondary method is permitted. Is delivery by com... | Service of process may be made by delivering a copy of the summons and petition to the defendant personally, or by leaving a copy at the... | Effective - 28 Aug 1988 506.150. Summons and petition, how served — service by mail, authorized when — notice by mail and acknowledgment... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Indiana_4133` | 16.142 | 0.222 | ret 0->1; ans 1->0 | Are eviction cases first heard in justice court? | In Indiana, the small claims court has jurisdiction over civil actions where the amount in controversy does not exceed $6,000, including... | Sec. 4. (a) This section applies after June 30, 2021.(b) The small claims docket has jurisdiction over the following:(1) Civil actions in... |
| BarExamQA | Groq Llama 8B | `qa_EVIDENCE_mbe_658` | 16.105 | 0.303 | ret 0->1; ans 1->1 | Daggett was prosecuted for murder of Vales, whose body was found one morning in the street near Daggett's house. The state calls Witt, a... | A statement relating to a startling event or condition made while the declarant was under the stress of excitement caused by the event or... | The excited utterance exception excludes from the general hearsay rule “[a] statement relating to a startling event or condition made whi... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Oklahoma_4422` | 16.049 | 0.210 | ret 0->1; ans 1->1 | Are eviction cases first heard in municipal court? | Forcible entry and detainer actions in Oklahoma are governed by statute and are typically heard in district court. The district court has... | \|The district court shall have jurisdiction to try all actions for the forcible entry and detention, or detention only, of real property... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Colorado_4702` | 15.935 | 0.132 | ret 0->1; ans 1->1 | Primary methods of service are defined as those methods that must be attempted before a secondary method is permitted. Is delivery by com... | Service of process may be made by delivering a copy of the summons and complaint to the defendant personally or by leaving a copy at the... | Such summons may be served by personal service as in any civil action. A copy of the complaint must be served with the summons. If person... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Florida_4050` | 15.876 | 0.154 | ret 0->1; ans 1->1 | Are eviction cases first heard in court of common pleas? | Florida's county courts have jurisdiction to hear cases involving eviction and other landlord-tenant disputes. The Florida Rules of Civil... | 34.011 Jurisdiction in landlord and tenant cases.—(1) The county court shall have jurisdiction concurrent with the circuit court to consi... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_California_5353` | 15.829 | 0.189 | ret 0->1; ans 1->1 | Secondary methods of service are defined as those methods that may be used if the primary method is unsuccessful. Is personal service a p... | Service of summons in unlawful detainer actions may be made by mail or by posting and mailing a copy of the summons and complaint to the... | 415.45. (a) A summons in an action for unlawful detainer of real property may be served by posting if upon affidavit it appears to the sa... |
| BarExamQA | Groq Llama 8B | `qa_nan_mbe_24` | 15.785 | 0.391 | ret 0->0; ans 1->0 | Pace sues Def Company for injuries suffered when Pace's car collided with Def Company's truck. Def's general manager prepared a report of... | The attorney-client privilege protects confidential communications made for the purpose of facilitating legal representation, including t... | The attorney-client privilege protects communications made in confidence by a client and a client's employees to an attorney, acting as a... |
| HousingQA state-filtered | Gemma 4 26B | `hqa_Oklahoma_4429` | 15.771 | 0.207 | ret 0->1; ans 1->1 | Are eviction cases first heard in magistrates court? | Forcible entry and detainer actions are civil proceedings used to recover possession of real property. In Oklahoma, jurisdiction over the... | \|The district court shall have jurisdiction to try all actions for the forcible entry and detention, or detention only, of real property... |
| BarExamQA | Gemma 4 26B | `qa_nan_mbe_318` | 15.600 | 0.290 | ret 0->0; ans 1->1 | Drew was tried for the July 21 murder of Victor. Drew called William to testify that on July 20 Drew said that he was about to leave that... | A statement of the declarant's then-existing state of mind, such as intent, plan, or motive, is admissible as an exception to the hearsay... | A statement of a declarant's then existing state of mind, emotion, sensation, or physical condition is an exception to the Hearsay Rule;... |

## Bottom CE-Delta Examples

| Dataset | Model | Label | CE delta | Cos delta | Outcomes | Raw question | SCOPE passage | Gold passage |
|---|---|---|---:|---:|---|---|---|---|
| HousingQA state-filtered | Gemma 4 26B | `hqa_Arkansas_515` | -15.575 | -0.239 | ret 1->0; ans 1->0 | Is it specified whether the law requires a tenant to pay a bond in order to appeal an eviction judgment? | The Arkansas Rules of Appellate Procedure govern the process for seeking review of a trial court's decision. There is no statutory requir... | (a) An appeal in an eviction case will not stay eviction unless at the time of appealing the tenant shall give an appeal bond as in other... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Texas_7225` | -12.573 | -0.109 | ret 1->0; ans 0->0 | In an eviction action, can a tenant rebut/raise the defense that eviction is unlawful because it was motivated by a legally recognized fo... | The Texas Fair Housing Act prohibits discrimination in the sale, rental, and financing of housing based on certain protected characterist... | Sec. 94.256. EVICTION SUITS. In an eviction suit, retaliation by the landlord under Section 94.251 is a defense and a rent deduction lawf... |
| HousingQA state-filtered | Gemma 4 26B | `hqa_New Hampshire_444` | -12.250 | -0.107 | ret 1->0; ans 1->1 | Does the law allow a landlord to include a lease provision whereby a tenant waives their right to notice? | New Hampshire statutes protect tenants from unlawful eviction practices and require landlords to provide specific notice periods before i... | 356-C:7 Waiver Prohibited. –\|\|No lease or rental agreement, oral or written, shall contain any provision by which the tenant prospectiv... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Wyoming_8829` | -11.202 | -0.060 | ret 1->0; ans 0->0 | Does the law require that writs of execution not be executed on weekends? | In Wyoming, the execution of judgments is governed by the Wyoming Rules of Civil Procedure, specifically Rule 69, which outlines the proc... | 1-21-1013. Writ of restitution; execution and return.\| Unless the defendant takes an appeal, the officer shall execute the writ of resti... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Tennessee_463` | -11.061 | -0.084 | ret 1->0; ans 1->1 | Is it specified whether the law allows a landlord to include a lease provision whereby a tenant waives their right to notice? | Tennessee courts have consistently upheld the validity of contractual provisions that waive statutory rights, provided the waiver is clea... | The landlord and tenant may include in a rental agreement, terms and conditions not prohibited by this chapter or other rule of law inclu... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Nevada_9280` | -10.954 | -0.050 | ret 1->0; ans 0->0 | Are court records for eviction cases which result in judgement for the tenant automatically made inaccessible? | Nevada Revised Statutes Section 11.085 provides that certain information, including social security numbers and financial account numbers... | 1. In any action for summary eviction pursuant to NRS 40.253, 40.254 or 40.2542, the eviction case court file is sealed automatically and... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Illinois_2760` | -10.771 | -0.125 | ret 1->0; ans 0->1 | Is it unlawful to evict a tenant because of their immigration status? | The Illinois Human Rights Act prohibits discrimination in housing based on national origin, which includes an individual's ancestry, cult... | Sec. 9-106.3. Affirmative defenses for retaliation on the basis of immigration status.(a) It is an affirmative defense to an action maint... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Oregon_319` | -10.399 | 0.008 | ret 1->0; ans 0->1 | In eviction proceedings, does the filing of an appeal stay the execution of a writ? | The Oregon Rules of Civil Procedure and the Oregon Revised Statutes govern the process of eviction and appeal. A stay of execution may be... | The filing of a notice of appeal does not automatically stay the judgment that is the subject of the appeal. A party may seek to stay a j... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_New Hampshire_443` | -10.318 | -0.135 | ret 1->0; ans 0->0 | Is it specified whether the law allows a landlord to include a lease provision whereby a tenant waives their right to notice? | New Hampshire Revised Statutes Annotated (RSA) 540:1-a requires that any notice given by a landlord must be in writing and specify the re... | 540:28 Lease Provisions. –\|\|No lease or rental agreement, oral or written, shall contain any provision by which a tenant waives any of... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_Texas_7232` | -9.504 | -0.101 | ret 1->0; ans 0->0 | In an eviction action, can a tenant rebut/raise the defense that the landlord refused a voucher? | The Texas Fair Housing Act prohibits discrimination against tenants based on their source of income, including government vouchers. A lan... | Sec. 94.256. EVICTION SUITS. In an eviction suit, retaliation by the landlord under Section 94.251 is a defense and a rent deduction lawf... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_West Virginia_474` | -9.167 | -0.145 | ret 0->0; ans 0->0 | Does the law allow a landlord to include a lease provision whereby a tenant waives their right to notice? | A waiver of a statutory right must be voluntary, knowing, and intelligent. A waiver is not voluntary if it is induced by fraud, duress, o... | A tenancy from year to year may be terminated by either party giving notice in writing to the other, at least three months prior to the e... |
| HousingQA state-filtered | Groq Llama 8B | `hqa_District of Columbia_144` | -9.080 | -0.067 | ret 1->0; ans 1->0 | Does the law directly regulate the amount a landlord can charge as a fee for late rent? | The Rent Control Act of 1985, as amended, sets forth the rules and regulations governing rent increases and other fees in the District of... | (a) Pursuant to subsection (b) of this section, a housing provider may charge a late fee of no more than 5% of the full amount of rent du... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Texas_7225` | -9.020 | -0.078 | ret 1->0; ans 0->0 | In an eviction action, can a tenant rebut/raise the defense that eviction is unlawful because it was motivated by a legally recognized fo... | The Texas Fair Housing Act prohibits discrimination in housing based on race, color, national origin, religion, sex, familial status, or... | Sec. 94.256. EVICTION SUITS. In an eviction suit, retaliation by the landlord under Section 94.251 is a defense and a rent deduction lawf... |
| HousingQA state-filtered | Groq Llama 70B | `hqa_Arkansas_515` | -8.965 | -0.061 | ret 1->0; ans 1->1 | Is it specified whether the law requires a tenant to pay a bond in order to appeal an eviction judgment? | In Arkansas, the law governing appeals of eviction judgments is outlined in the Arkansas Code. According to the relevant statutes, a tena... | (a) An appeal in an eviction case will not stay eviction unless at the time of appealing the tenant shall give an appeal bond as in other... |
| HousingQA state-filtered | Gemma 4 26B | `hqa_Texas_7220` | -8.837 | -0.092 | ret 1->0; ans 1->0 | In an eviction action, can a tenant rebut/raise the defense that the property is uninhabitable? | A landlord has a statutory duty to repair or remedy conditions that materially affect the physical health or safety of an ordinary tenant... | Sec. 92.335. EVICTION SUITS. In an eviction suit, retaliation by the landlord under Section 92.331 is a defense and a rent deduction lawf... |

## Reading

- H1 is mixed. Low raw query-gold alignment predicts SCOPE retrieval benefit on HousingQA (Spearman -0.257 CE, -0.242 cosine), but the raw-alignment score alone is near-null on BarExamQA (0.012 CE, 0.023 cosine).
- H2 is the cleaner mechanism result. The movement toward gold predicts retrieval gain in both datasets: CE delta Spearman is 0.340 on BarExamQA and 0.453 on HousingQA; cosine delta is 0.330 and 0.366, respectively. Pooled CE/cosine deltas are 0.436 / 0.368.
- H3 is weak: answer-delta correlations are small across all four axes. The largest pooled answer Spearman in this report is 0.066, so query-gold alignment explains retrieval movement better than downstream answer movement.
- Mechanism read: the non-circular query-gold gap is a better explanation than unigram perplexity. SCOPE helps when the raw query is far from the gold passage and when the generated passage increases cross-encoder affinity to that gold passage; answer conversion still depends on whether the repaired evidence is useful rather than distracting.

## Sources

- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_groq-llama8b_snap_hyre.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_groq-llama70b_snap_hyre.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
- `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`
- `logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl`
- `caches/hyre/full/housing_qfull_seed42_groq-llama8b_snap_hyre.jsonl`
- `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/hyre/full/housing_qfull_seed42_groq-llama70b_snap_hyre.jsonl`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_scope_gap_mechanism.py \
  --output docs/generated/scope_gap_mechanism_2026-05-25.md
```
