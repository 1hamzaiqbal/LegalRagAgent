# Golden-passage paradox audit — Gemma 4 26B-A4B BarExam Tier 3 N=1195

Meeting 2026-04-27 #1 ask: explain why `golden_passage` (78.66%) is BELOW `llm_only` (79.75%) 
when both are run on the same model with the same fact pattern, and where the lift to 
`rag_snap_hyde` (81.17%) actually comes from.

## TL;DR — paradox explained

1. **The gold-passage injection is roughly symmetric, not monotone.** Across 1195 paired
   questions, injecting the labeled gold passage *flipped* 96 previously-correct answers to
   wrong and 83 previously-wrong answers to correct, for a net **-1.09pp** vs `llm_only`.
   The "gold" passage is therefore **not an oracle ceiling**; it is approximately a coin flip
   with a slight net cost on this benchmark/model combo.

2. **The dominant failure mode is anchoring, not insufficient evidence.** Gold-passage length
   is essentially identical across paradox / golden-win / outcome-match buckets
   (avg ≈ 420-440 chars, median 500, no length-bucket separation). What separates them is
   *which doctrine the gold passage anchors to* — when the gold passage cites a related-but-
   wrong rule (or a partial rule), the model follows the passage instead of its priors.

3. **`rag_snap_hyde` recovers from the gold-passage anchor 65.6% of the time.** In the 96
   paradox cases, snap-first reasoning lands the *same answer as `llm_only`* in 63/96 cases
   (the snap dominates the prior even when retrieval surfaces the same misleading passage).
   In 28/96 (29.2%) snap_hyde itself anchors to the same wrong answer as the gold-injected
   model — i.e., the gold passage is genuinely misleading enough to fool snap+HyDE too.

4. **`rag_snap_hyde`'s +3.10pp lift over `rag_simple` is a 124-win / 87-loss split**, not a
   monotone improvement. The lift is real but mechanism-driven: snap-first reasoning is a
   stronger anchor than any single retrieved passage, *including* the gold passage itself.

### Implications for paper language

- **Stop calling `golden_passage` an "oracle ceiling".** Rename to **"single gold-passage
  control"** or **"provided-snippet control"** in tables and prose. The reglab gold label
  is one plausible authority, not a sufficient ground-truth context.
- The BarExam `rag_snap_hyde` win is not "retrieval beats no-retrieval". It is **"snap-anchored
  retrieval beats both raw retrieval and forced gold-context"**. That is a stronger and more
  honest claim than the headline "RAG helps legal MC".
- For the snap-vs-evidence-source mechanism question, the cleanest experiment is `snap_only_in_final`
  (80.59%) ≈ `rag_snap_hyde` (81.17%) ≈ `llm_only` (79.75%) ≫ `golden_passage` (78.66%).
  Snap reasoning + *any* permissive evidence source ≥ gold passage forced-context.

Source data + code: `scripts/audit_golden_paradox.py`, run on the 4 paths listed below.

## Independent peer-review verification (Haiku subagent, blind to my conclusions)

A Haiku subagent independently re-ran the audit without seeing this doc. All headline numbers matched exactly (79.75 / 78.66 / 78.08 / 81.17). It confirmed: symmetric flip (96 hurt / 83 helped, 1.16:1), no length signal (identical medians of 500 chars across buckets), and the anchoring claim (87% same-wrong-pred between snap_hyde and golden_passage in the paradox-and-snap_hyde-failed subset, vs my own 84.8% measurement on the same data).

It also added two findings I had not run:

### 30-case manual deep-dive of paradox failure modes

| Bucket | Count | % | Description |
|---|---:|---:|---|
| (D) Model misapplied a relevant gold passage | 15 | 50% | Gold contains the right rule; model cited it but reached the wrong answer |
| (A) Gold passage topically irrelevant | 9 | 30% | Gold cites a different legal area entirely — dataset curation issue (reglab keyword-pairing) |
| (B) Distractor / partially-on-topic gold | 6 | 20% | Gold cites a related-but-wrong rule that misdirects the model |
| (C) Questionable label | 0 | 0% | Reglab's "correct" answer looked debatable |

Implication: ~30% of the paradox is upstream dataset-quality (reglab gold labels include topically-mismatched passages), not a model failure. The remaining 70% is a real model-vs-context interaction (anchoring/misapplication).

### Cross-dataset direction reversal — golden_passage HELPS on MuSiQue

Peer review paired Llama 70b MuSiQue `golden_passage` (N=30) against `rag_simple` (N=30) and found golden_passage **+40pp above** rag_simple. The exact magnitude is suspect because the N=30 rag_simple comparator (6.7%) is far below our cited N=200 Llama 70b MuSiQue rag_simple baseline (27.5%) — likely an early/unrepresentative log. **But the qualitative direction is meaningful**: on multi-hop factoid QA where the model has weak priors, gold passage helps materially; on legal MC where the model has strong priors, gold passage hurts.

The cleanest framing for the paper:

> **Gold-passage usefulness is inversely correlated with model prior strength on the task.**
> When the model already "knows" the answer, injected gold context is noise (BarExam: -1.09pp).
> When the model needs facts it doesn't have, injected gold context is signal (MuSiQue: +Xpp).
> Either way, the "oracle ceiling" framing is wrong — golden_passage is a noisy single-snippet
> control whose net direction depends on the task.

To strengthen this for citation, re-run paired golden_passage vs llm_only and golden_passage vs rag_simple on Llama 70b MuSiQue at N=200 with the same question set as our existing Tier 2 logs.


Logs paired:

- `llm_only` → `logs/eval_llm_only_cluster-vllm_20260426_0027_detail.jsonl` (N=1195)
- `golden_passage` → `logs/eval_golden_passage_cluster-vllm_20260426_0224_detail.jsonl` (N=1195)
- `rag_simple` → `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl` (N=1195)
- `rag_snap_hyde` → `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` (N=1195)

Common idx across all 4 logs: **1195**

## Paired transition tables

### llm_only vs golden_passage

- Paired N: 1195
- llm_only acc: 79.75% | golden_passage acc: 78.66% | Δ (golden_passage - llm_only): -1.09pp

|  | golden_passage right | golden_passage wrong |
|---|---:|---:|
| **llm_only right** | 857 | 96 |
| **llm_only wrong** | 83 | 159 |


### rag_simple vs golden_passage

- Paired N: 1195
- rag_simple acc: 78.08% | golden_passage acc: 78.66% | Δ (golden_passage - rag_simple): +0.59pp

|  | golden_passage right | golden_passage wrong |
|---|---:|---:|
| **rag_simple right** | 841 | 92 |
| **rag_simple wrong** | 99 | 163 |


### golden_passage vs rag_snap_hyde

- Paired N: 1195
- golden_passage acc: 78.66% | rag_snap_hyde acc: 81.17% | Δ (rag_snap_hyde - golden_passage): +2.51pp

|  | rag_snap_hyde right | rag_snap_hyde wrong |
|---|---:|---:|
| **golden_passage right** | 860 | 80 |
| **golden_passage wrong** | 110 | 145 |


### llm_only vs rag_snap_hyde

- Paired N: 1195
- llm_only acc: 79.75% | rag_snap_hyde acc: 81.17% | Δ (rag_snap_hyde - llm_only): +1.42pp

|  | rag_snap_hyde right | rag_snap_hyde wrong |
|---|---:|---:|
| **llm_only right** | 875 | 78 |
| **llm_only wrong** | 95 | 147 |


### rag_simple vs rag_snap_hyde

- Paired N: 1195
- rag_simple acc: 78.08% | rag_snap_hyde acc: 81.17% | Δ (rag_snap_hyde - rag_simple): +3.10pp

|  | rag_snap_hyde right | rag_snap_hyde wrong |
|---|---:|---:|
| **rag_simple right** | 846 | 87 |
| **rag_simple wrong** | 124 | 138 |


## Prediction distributions (paired N)

| Letter | golden_passage | llm_only |
|---|---:|---:|
| `A` | 262 | 271 |
| `B` | 281 | 288 |
| `C` | 327 | 326 |
| `D` | 325 | 310 |

## The paradox: golden_passage failed while llm_only succeeded

Total paradox cases: **96**

These are the cases the meeting flagged: gold passage was injected into the prompt, 
but the model did worse than with no context at all. Either the gold passage isn't 
sufficient, the model was distracted/anchored by it, or the gold label is questionable.

### Sample paradox cases

**1. idx=mbe_1031** (correct='C', golden→'D', llm_only→'C', snap_hyde→'B')

- **Q**: Mom rushed her eight-year-old daughter, Child, to the emergency room at Hospital after Child fell off her bicycle and hit her head on a sharp rock. The wound caused by the fall was extensive and bloody. Mom was permitted to remain in the treatment room, and held Child's hand while the emergency room physician cleaned and sutured the wound. During the procedure, Mom said that she was feeling fai...
- **Gold passage** (500 chars): As noted previously, the imposition of a duty to use reasonable care to prevent injury to another (as distinguished from using reasonable care while engaged in an activity that poses a risk of harm to others) is an exception to the general rule of nonliability for nonfeasance.1   A duty of care will be imposed upon the defendant to use reasonable care to prevent injury to the plaintiff if (a) t...
- **gold_idx**: `mbe_790` | **gold_retrieved (snap_hyde)**: False

**2. idx=mbe_1032** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: For this question only, assume that Conglomerate orally approved the contract, but that Shareholder changed her mind and refused to consummate the sale on two grounds: (1) when the agreement was made there was no consideration for her promise to sell; and (2) Conglomerate's approval of the contract was invalid. If Buyer sues Shareholder for breach of contract, is Buyer likely to prevail?
- **Gold passage** (500 chars): The concept of mutuality of obligation requires that both parties to a contract be bound by the terms of the contract. See, e.g., Floss, 211 F.3d at 315–16. It goes hand in hand with the concepts of “consideration” and the “illusory promise” (which is basically an empty promise: promising to do one thing while, at the same time, expressly retaining the right to change one's mind). See id. at 31...
- **gold_idx**: `mbe_2705` | **gold_retrieved (snap_hyde)**: False

**3. idx=mbe_1037** (correct='D', golden→'A', llm_only→'D', snap_hyde→'A')

- **Q**: Corp, a corporation, owned Blackacre in fee simple, as the real estate records showed. Corp entered into a valid written contract to convey Blackacre to Barbara, an individual. At closing, Barbara paid the price in full and received an instrument in the proper form of a deed, signed by duly authorized corporate officers on behalf of Corp, purporting to convey Blackacre to Barbara. Barbara did n...
- **Gold passage** (500 chars): When a creditor records a judgment in the Florida public records, the recording results in a claim against the purported homestead property. As a claim, the recorded judgment asserts a “right to payment,” albeit a contingent one, subject to the debtor's ability to successfully assert homestead rights under Florida law. See Fla. Stat. § 222.01. Consequently, a recorded judgment, irrespective of ...
- **gold_idx**: `mbe_793` | **gold_retrieved (snap_hyde)**: False

**4. idx=mbe_1040** (correct='D', golden→'B', llm_only→'D', snap_hyde→'A')

- **Q**: Mrs. Pence sued Duarte for shooting her husband from ambush. Mrs. Pence offers to testify that, the day before her husband was killed, he described to her a chance meeting with Duarte on the street in which Duarte said, "I'm going to blow your head off one of these days." The witness's testimony concerning her husband's statement is
- **Gold passage** (212 chars): Hearsay is a statement, including both oral assertions and nonverbal conduct intended as an assertion, made by a person not currently testifying and offered in evidence to prove the truth of the matter asserted.
- **gold_idx**: `mbe_794` | **gold_retrieved (snap_hyde)**: False

**5. idx=mbe_1041** (correct='C', golden→'B', llm_only→'C', snap_hyde→'D')

- **Q**: The state of Brunswick enacted a statute providing for the closure of the official state records of arrest and prosecution of all persons acquitted of a crime by a court or against whom criminal charges were filed and subsequently dropped or dismissed. The purpose of this statute is to protect these persons from further publicity or embarrassment relating to those state proceedings. However, th...
- **Gold passage** (500 chars): [W]e have held that the right to publish is central to the First Amendment and basic to the existence of constitutional democracy. Grosjean, supra, at 250, 56 S.Ct. at 449; New York Times, supra, 376 U.S. at 270, 84 S.Ct. at 720.   A corollary of the right to publish must be the right to gather news. The full flow of information to the public protected by the free-press guarantee would be sever...
- **gold_idx**: `mbe_2708` | **gold_retrieved (snap_hyde)**: False

**6. idx=mbe_106** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: Pemberton's counsel seeks to introduce Helper's written statement that Edwards, Mammoth's driver, had left his glasses (required by his operator's license) at the truck Stop which they had left five minutes before the accident. The judge should rule the statement admissible only if
- **Gold passage** (500 chars): The defendant next argues that, even if the sketch is hearsay, the trial justice erred in excluding it because the sketch was admissible under Rule 804(b)(5) or the “catch-all” exception. “Under Rule 804(b)(5)(B), an out-of-court statement made by an unavailable witness can be admitted to prove the truth of the matter asserted, if, among other requirements, ‘the statement is more probative on t...
- **gold_idx**: `mbe_2615` | **gold_retrieved (snap_hyde)**: False

**7. idx=mbe_1079** (correct='A', golden→'D', llm_only→'A', snap_hyde→'A')

- **Q**: By a writing, Oner leased his home, Blackacre, to Tenn for a term of three years, ending December 31 of last year, at the rent of $1,000 per month. The lease provided that Tenn could sublet and assign. Tenn lived in Blackacre for one year and paid the rent promptly. After one year, Tenn leased Blackacre to Agrit for one year at a rent of $1,000 per month. Agrit took possession of Blackacre and ...
- **Gold passage** (206 chars): A judgment in unlawful detainer declaring the forfeiture of the lease or agreement under which real property is held shall not relieve the lessee from liability pursuant to Section 1951.2 of the Civil Code.
- **gold_idx**: `mbe_824` | **gold_retrieved (snap_hyde)**: False

**8. idx=mbe_1087** (correct='A', golden→'C', llm_only→'A', snap_hyde→'C')

- **Q**: A written construction contract began with the following recital: "This Agreement, between Land, Inc. (hereafter called 'Owner'), and Builder, Inc., and Boss, its President (hereafter called 'Contractor'), witnesseth:" The signatures to the contract appeared in the following format: LAND, INC. By /s/ Oscar Land President BUILDER, INC. By /s/ George Mason Vice President /s/ Mary Boss, President ...
- **Gold passage** (500 chars): A written agreement is ambiguous when a plain reading of the contract could result in more than one reasonable interpretation. See also Metric Constructors, Inc. v. NASA, 169 F.3d 747, 751 (Fed.Cir.1999); Grumman Data Sys. Corp. v. Dalton, 88 F.3d 990, 997 (Fed.Cir.1996); A–Transport Northwest Co. v. United States, 36 F.3d 1576, 1584 (Fed.Cir.1994) (“A contract is ambiguous only when it is susc...
- **gold_idx**: `mbe_830` | **gold_retrieved (snap_hyde)**: False

**9. idx=mbe_1100** (correct='D', golden→'A', llm_only→'D', snap_hyde→'D')

- **Q**: Olivia owned Blackacre, her home. Her daughter, Dawn, lived with her and always referred to Blackacre as "my property." Two years ago, Dawn, for a valuable consideration, executed and delivered to Bruce an instrument in the proper form of a warranty deed purporting to convey Blackacre to Bruce in fee simple, reserving to herself an estate for two years in Blackacre. Bruce promptly and properly ...
- **Gold passage** (276 chars): A purchaser is bound by every recital, reference, and reservation contained in or fairly disclosed by any instrument that forms an essential link in the chain of title under which the purchaser claims. See Westland Oil Dev. Corp. v. Gulf Oil, 637 S.W.2d 903, 908 (Tex. 1982).
- **gold_idx**: `mbe_839` | **gold_retrieved (snap_hyde)**: False

**10. idx=mbe_1102** (correct='D', golden→'A', llm_only→'D', snap_hyde→'D')

- **Q**: The warden of State Prison prohibits the photographing of the face of any prisoner without the prisoner's consent. Photographer, a news photographer, wanted to photograph Mobster, a notorious organized crime figure incarcerated at State Prison. To circumvent the warden's prohibition, Photographer flew over the prison exercise yard and photographed Mobster. Prisoner, who was imprisoned for a tec...
- **Gold passage** (492 chars): To satisfy the first element of an IIED claim, a plaintiff must demonstrate that the defendant either “desired to inflict severe emotional distress, knew that such distress was certain or substantially certain to result from his conduct, or acted recklessly in deliberate disregard of a high degree of probability that emotional distress would follow.” Interphase Garment Solutions, LLC, 566 F.Sup...
- **gold_idx**: `mbe_840` | **gold_retrieved (snap_hyde)**: False

**11. idx=mbe_1112** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: Ven owned Goldacre, a tract of land, in fee simple. Ven and Pur entered into a written agreement under which Pur agreed to buy Goldacre for $100,000, its fair market value. The agreement contained all the essential terms of a real estate contract to sell and buy, including a date for closing. The required $50,000 down payment was made. The contract provided that in the event of Pur's breach, Ve...
- **Gold passage** (500 chars): To establish a claim for conversion, a plaintiff must prove that (1) the plaintiff owned or had possession of the property or entitlement to possession; (2) the defendant unlawfully and without authorization assumed and exercised control over the property to the exclusion of, or inconsistent with, the plaintiff's rights as an owner; (3) the plaintiff demanded return of the property; and (4) the...
- **gold_idx**: `mbe_848` | **gold_retrieved (snap_hyde)**: False

**12. idx=mbe_1121** (correct='B', golden→'D', llm_only→'B', snap_hyde→'D')

- **Q**: nan
- **Gold passage** (368 chars): To invoke the doctrine of res ipsa loquitur, the plaintiff must establish: “(1) the event [was] of a kind which ordinarily does not occur in the absence of someone's negligence; (2) it [was] caused by an agency or instrumentality within the exclusive control of the defendant; (3) it [was not] due to any voluntary action or contribution on the part of the plaintiff”.
- **gold_idx**: `mbe_2435` | **gold_retrieved (snap_hyde)**: False

**13. idx=mbe_1138** (correct='C', golden→'B', llm_only→'C', snap_hyde→'B')

- **Q**: Kontractor agreed to build a power plant for a public utility. Subbo agreed with Kontractor to lay the foundation for $200,000. Subbo supplied goods and services worth $150,000, for which Kontractor made progress payments aggregating $100,000 as required by the subcontract. Subbo then breached by refusing unjustifiably to perform further. Kontractor reasonably spent $120,000 to have the work co...
- **Gold passage** (500 chars): At common law, a breaching party could not obtain restitution for benefits conferred. The common law rule reflected a belief that breach was “morally unworthy conduct,” and that a breaching party should not benefit from his own wrong, see Lancellotti v. Thomas, 341 Pa.Super. 1, 491 A.2d 117, 118–19 (1985) (internal quotation marks omitted). In contrast to the common law rule, the Restatement ru...
- **gold_idx**: `mbe_2437` | **gold_retrieved (snap_hyde)**: False

**14. idx=mbe_1148** (correct='A', golden→'D', llm_only→'A', snap_hyde→'A')

- **Q**: Thirty years ago Able, the then-record owner of Greenacre, a lot contiguous to Blueacre, in fee simple, executed and delivered to Baker an instrument in writing which was denominated "Deed of Conveyance." In pertinent part it read, "Able does grant to Baker and her heirs and assigns a right-of-way for egress and ingress to Blueacre." If the quoted provision was sufficient to create an interest ...
- **Gold passage** (500 chars): “To establish a prescriptive easement, the claimant must prove by clear and convincing evidence: ‘(1) the continued and uninterrupted use or enjoyment of the right for a period of [twenty] years; (2) the identity of the thing enjoyed; and (3) the use [was] adverse under claim of right.’ ” Simmons v. Berkeley Elec. Coop., Inc., Op. No. 27674 (S.C. Sup. Ct. filed Nov. 2, 2016) (Shearouse Adv. Sh....
- **gold_idx**: `mbe_872` | **gold_retrieved (snap_hyde)**: False

**15. idx=mbe_115** (correct='B', golden→'A', llm_only→'B', snap_hyde→'A')

- **Q**: What was the probable legal effect of the following? I. Sawtooth's failure to object to Farquart's making no payments on November 1, December I , January 1, and February I II. Farquart's making payments in August through October without requiring a certificate from Builders.
- **Gold passage** (500 chars): An implied waiver may also arise where the party against whom waiver is asserted pursues a course of action or acts in such a way that demonstrates his intention to waive a right or is inconsistent with any intention other than waiving the right. Hahn v. County of Kane, 2013 IL App (2d) 120660, ¶ 11, 372 Ill.Dec. 66, 991 N.E.2d 373. However, “we must point out that ‘equitable estoppel’ and ‘wai...
- **gold_idx**: `mbe_93` | **gold_retrieved (snap_hyde)**: False

**16. idx=mbe_1161** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: Three years ago Adam conveyed Blackacre to Betty for $50,000 by a deed that provided: "By accepting this deed, Betty covenants for herself, her heirs and assigns, that the premises herein conveyed shall be used solely for residential purposes and, if the premises are used for nonresidential purposes, Adam, his heirs and assigns, shall have the right to repurchase the premises for the sum of one...
- **Gold passage** (364 chars): The rule against perpetuities is generally stated with deceptive simplicity as follows: “No interest is good unless it must vest, if at all, not later than twenty-one years after some life in being at the creation of the interest.” Iglehart v. Phillips, 383 So.2d 610, 614 (Fla.1980) (quoting John Chipman Gray, The Rule Against Perpetuities, § 201 (4th ed.1942)).
- **gold_idx**: `mbe_884` | **gold_retrieved (snap_hyde)**: False

**17. idx=mbe_117** (correct='C', golden→'D', llm_only→'C', snap_hyde→'C')

- **Q**: Doctor, a licensed physician, resided in her own home. The street in front of the home had a gradual slope. Doctor's garage was on the street level, with a driveway entrance from the street. At two in the morning Doctor received an emergency call. She dressed and went to the garage to get her car and found a car parked in front of her driveway. That car was occupied by Parker, who, while intoxi...
- **Gold passage** (500 chars): Kansas courts discussed assumption of risk as early as 1898. See Greef Bros. v. Brown, 7 Kan.App. 394, 51 P. 926 (1898). But in those early cases it was viewed as a “species of contributory negligence.” Greef Bros., 7 Kan.App. at 398, 51 P. 926 (discussing relationship between assumption of risk and contributory negligence). Contributory negligence is defined as “conduct on the part of the plai...
- **gold_idx**: `mbe_95` | **gold_retrieved (snap_hyde)**: False

**18. idx=mbe_1179** (correct='B', golden→'D', llm_only→'B', snap_hyde→'D')

- **Q**: Senator makes a speech on the floor of the United States Senate in which she asserts that William, a federal civil servant with minor responsibilities, was twice convicted of fraud by the courts of State X. In making this assertion, Senator relied wholly on research done by Frank, her chief legislative assistant. In fact, it was a different man named William and not William the civil servant, w...
- **Gold passage** (500 chars): Article I, Section 6, clause 1 of the Constitution reads, in part, [Senators and Representatives] shall in all Cases, except Treason, Felony and Breach of the Peace, be privileged … for any Speech or Debate in either House, they shall not be questioned in any other Place1 The Speech or Debate Clause protects the questioning of a Congressman and provides that a member of either house shall not b...
- **gold_idx**: `mbe_897` | **gold_retrieved (snap_hyde)**: False

**19. idx=mbe_1184** (correct='B', golden→'C', llm_only→'B', snap_hyde→'B')

- **Q**: Martin, the owner in fee simple of Orchardacres, mortgaged Orchardacres to Marie to secure the payment of the loan she made to him. The loan was due at the end of the growing season of the year in which it was made. Martin maintained and operated an orchard on the land, which was his sole source of income. Halfway through the growing season, Martin experienced severe health and personal problem...
- **Gold passage** (500 chars): A mortgagee may become a mortgagee in possession where the mortgage itself gives the mortgagee the right to enter and “take possession of the mortgaged premises and receive the rents and profits” (Gomez v. Bobker, 124 A.D.2d 703, 704, 508 N.Y.S.2d 215; see e.g. Gasco Corp. & Gordian Group of Hong Kong v. Tosco Props., 236 A.D.2d 510, 653 N.Y.S.2d 687). A mortgagee in possession “takes the rents...
- **gold_idx**: `mbe_901` | **gold_retrieved (snap_hyde)**: True

**20. idx=mbe_132** (correct='C', golden→'B', llm_only→'C', snap_hyde→'B')

- **Q**: Patty sues Mart Department Store forpersonal injuries, alleging that while shopping she was knocked to the floor by a merchandise cart being pushed by Handy, a stock clerk, and that as a consequence her back was injured. Handy testified that Patty fell near the cartbut was not struck by it. Thirty minutes after Patty's fall, Handy, in accordance with regular practice at Mart, had filled out a p...
- **Gold passage** (500 chars): Hearsay is an out of court, unsworn, oral or written statement by a third person, which is offered for the truth of its content. Hearsay statements are inadmissible unless they fit into one of the recognized exceptions.” **6 State v. Gremillion, 542 So.2d 1074, 1077 (La.1989) (citation omitted). One such exception to the hearsay exclusionary rule applies to unavailable witnesses: A. Definition ...
- **gold_idx**: `mbe_2618` | **gold_retrieved (snap_hyde)**: False

## Lift mechanism: rag_snap_hyde won where golden_passage failed

Total cases where snap_hyde correct + golden_passage wrong: **110**

These are the cases driving the +2.51pp gap between snap_hyde and golden_passage. 
Inspect to see whether snap_hyde wins by retrieving better evidence than the gold passage, 
by ignoring the noisy gold passage, or by snap-reasoning around it.

### Sample snap_hyde-over-golden cases

**1. idx=mbe_0** (correct='B', golden→'D', llm_only→'D', snap_hyde→'B')

- **Q**: Paul then called Vic to testify that Dan's car did run the light. The trial judge should rule that Vic's testimony is
- **Gold passage** (366 chars): Rules of evidence determine what types of evidence is admissible, and the trial court judge applies these rules to the case. Generally, to be admissible, the evidence must be relevant) and not outweighed by countervailing considerations (e.g., the evidence is unfairly prejudicial, confusing, a waste of time, privileged, or, among other reasons, based on hearsay).
- **gold_idx**: `mbe_0` | **gold_retrieved (snap_hyde)**: False

**2. idx=mbe_1004** (correct='C', golden→'A', llm_only→'A', snap_hyde→'C')

- **Q**: By the terms of a written contract signed by both parties on January 15, M.B. Ram, Inc., agreed to sell a specific ICB personal computer to Marilyn Materboard for $3,000, and Materboard agreed to pick up and pay for the computer at Ram's store on February 1. Materboard unjustifiably repudiated on February 1. Without notifying Materboard, Ram subsequently sold at private sale the same specific c...
- **Gold passage** (419 chars): a repudiation is (1) a statement by the obligor to the obligee indicating that he or she will commit a breach that would of itself give the obligee a claim for damages for total breach or (2) a voluntary affirmative act which renders the obligor either unable or apparently unable to perform without such a breach. Hooker and Heft v. Estate of Weinberger, 203 Neb. 674, 279 N.W.2d 849 (1979); Rest...
- **gold_idx**: `mbe_769` | **gold_retrieved (snap_hyde)**: False

**3. idx=mbe_1009** (correct='B', golden→'D', llm_only→'D', snap_hyde→'B')

- **Q**: Karen was crossing Main Street at a crosswalk. John, who was on the sidewalk nearby, saw a speeding automobile heading in Karen's direction. John ran into the street and pushed Karen out of the path of the car. Karen fell to the ground and broke her leg. In an action for battery brought by Karen against John, will Karen prevail?
- **Gold passage** (302 chars): The rescue exception is based on the tort theory that if one undertakes to render aid to another or to warn a person in danger, one must exercise reasonable care. If a rescuer fails to exercise care and increases the risk of harm to those he is trying to rescue, he is liable for any damages he causes.
- **gold_idx**: `mbe_774` | **gold_retrieved (snap_hyde)**: False

**4. idx=mbe_1030** (correct='C', golden→'B', llm_only→'D', snap_hyde→'C')

- **Q**: Rachel, an antique dealer and a skilled calligrapher, crafted a letter on very old paper. She included details that would lead knowledgeable readers to believe the letter had been written by Thomas Jefferson to a friend. Rachel, who had a facsimile of Jefferson's autograph, made the signature and other writing on the letter resemble Jefferson's. She knew that the letter would attract the attent...
- **Gold passage** (182 chars): In California, ‘forgery consists either in the false making or alteration of a document without authority or the uttering (making use) of such a document with the intent to defraud.’
- **gold_idx**: `mbe_789` | **gold_retrieved (snap_hyde)**: True

**5. idx=mbe_1032** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: For this question only, assume that Conglomerate orally approved the contract, but that Shareholder changed her mind and refused to consummate the sale on two grounds: (1) when the agreement was made there was no consideration for her promise to sell; and (2) Conglomerate's approval of the contract was invalid. If Buyer sues Shareholder for breach of contract, is Buyer likely to prevail?
- **Gold passage** (500 chars): The concept of mutuality of obligation requires that both parties to a contract be bound by the terms of the contract. See, e.g., Floss, 211 F.3d at 315–16. It goes hand in hand with the concepts of “consideration” and the “illusory promise” (which is basically an empty promise: promising to do one thing while, at the same time, expressly retaining the right to change one's mind). See id. at 31...
- **gold_idx**: `mbe_2705` | **gold_retrieved (snap_hyde)**: False

**6. idx=mbe_106** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: Pemberton's counsel seeks to introduce Helper's written statement that Edwards, Mammoth's driver, had left his glasses (required by his operator's license) at the truck Stop which they had left five minutes before the accident. The judge should rule the statement admissible only if
- **Gold passage** (500 chars): The defendant next argues that, even if the sketch is hearsay, the trial justice erred in excluding it because the sketch was admissible under Rule 804(b)(5) or the “catch-all” exception. “Under Rule 804(b)(5)(B), an out-of-court statement made by an unavailable witness can be admitted to prove the truth of the matter asserted, if, among other requirements, ‘the statement is more probative on t...
- **gold_idx**: `mbe_2615` | **gold_retrieved (snap_hyde)**: False

**7. idx=mbe_1066** (correct='A', golden→'D', llm_only→'D', snap_hyde→'A')

- **Q**: Employer retained Doctor to evaluate medical records of prospective employees. Doctor informed Employer that Applicant, a prospective employee, suffered from AIDS. Employer informed Applicant of this and declined to hire her. Applicant was shocked by this news and suffered a heart attack as a result. Subsequent tests revealed that Applicant in fact did not have AIDS. Doctor had negligently conf...
- **Gold passage** (468 chars): An invasion-of-privacy claim protects against four types of invasion of privacy: “(1) intrusion upon the plaintiff's seclusion or solitude or into his private affairs; (2) public disclosure of embarrassing private facts about the plaintiff; (3) publicity that places the plaintiff in a false light in the public eye; and (4) appropriation, for the defendant's advantage, of the plaintiff's name or...
- **gold_idx**: `mbe_815` | **gold_retrieved (snap_hyde)**: False

**8. idx=mbe_1075** (correct='B', golden→'C', llm_only→'C', snap_hyde→'B')

- **Q**: In a jurisdiction without a Dead Man's Statute, Parker's estate sued Davidson claiming that Davidson had borrowed from Parker $10,000, which had not been repaid as of Parker's death. Parker was run over by a truck. At the accident scene, while dying from massive injuries, Parker told Officer Smith to "make sure my estate collects the $10,000 I loaned to Davidson." Smith's testimony about Parker...
- **Gold passage** (427 chars): for an excited utterance to be admissible, the following requirements must be met: (1) there must have been an event startling enough to cause nervous excitement; (2) the statement must have been made before there was time to contrive or misrepresent; and (3) the statement must have been made while the person was under the stress of excitement caused by the startling event. See State v. Jano, 5...
- **gold_idx**: `mbe_821` | **gold_retrieved (snap_hyde)**: False

**9. idx=mbe_1079** (correct='A', golden→'D', llm_only→'A', snap_hyde→'A')

- **Q**: By a writing, Oner leased his home, Blackacre, to Tenn for a term of three years, ending December 31 of last year, at the rent of $1,000 per month. The lease provided that Tenn could sublet and assign. Tenn lived in Blackacre for one year and paid the rent promptly. After one year, Tenn leased Blackacre to Agrit for one year at a rent of $1,000 per month. Agrit took possession of Blackacre and ...
- **Gold passage** (206 chars): A judgment in unlawful detainer declaring the forfeiture of the lease or agreement under which real property is held shall not relieve the lessee from liability pursuant to Section 1951.2 of the Civil Code.
- **gold_idx**: `mbe_824` | **gold_retrieved (snap_hyde)**: False

**10. idx=mbe_1081** (correct='A', golden→'B', llm_only→'B', snap_hyde→'A')

- **Q**: Bill owned in fee simple Lot 1 in a properly approved subdivision, designed and zoned for industrial use. Gail owned the adjoining Lot 2 in the same subdivision. The plat of the subdivision was recorded as authorized by statute. Twelve years ago, Bill erected an industrial building wholly situated on Lot 1 but with one wall along the boundary common with Lot 2. The construction was done as auth...
- **Gold passage** (423 chars): Though an adjoining owner has no right of support in his neighbor's land for his buildings, unless he has acquired it by grant or otherwise, and the latter may excavate in his *1088 land so as to cause them to fall, without committing a trespass or taking away a property right, provided the adjacent soil would not have fallen of its own weight, he may nevertheless be liable in respect to his co...
- **gold_idx**: `mbe_825` | **gold_retrieved (snap_hyde)**: False

**11. idx=mbe_1085** (correct='D', golden→'A', llm_only→'A', snap_hyde→'D')

- **Q**: In a jurisdiction that has abolished the felony-murder rule, but otherwise follows the common law of murder, Sally and Ralph, both armed with automatic weapons, went into a bank to rob it. Ralph ordered all the persons in the bank to lie on the floor. When some were slow to obey, Sally, not intending to hit anyone, fired about 15 rounds into the air. One of these ricocheted off a stone column a...
- **Gold passage** (500 chars): Depraved heart murder and culpable negligence differ “simply by degree of mental state of culpability. In short, depraved-heart murder involves a higher degree of recklessness from which malice or deliberate design may be implied.” Windham v. State, 602 So.2d 798, 801 (Miss.1992). This distinction can be seen in the jury instructions by comparing the two instructions that defined depraved heart...
- **gold_idx**: `mbe_828` | **gold_retrieved (snap_hyde)**: False

**12. idx=mbe_1091** (correct='C', golden→'A', llm_only→'A', snap_hyde→'C')

- **Q**: Olive owned Blackacre, a single-family residence. Fifteen years ago, Olive conveyed a life estate in Blackacre to Lois. Fourteen years ago, Lois, who had taken possession of Blackacre, leased Blackacre to Trent for a term of 15 years at the monthly rental of $500. Eleven years ago, Lois died intestate leaving Ron as her sole heir. Trent regularly paid rent to Lois and, after Lois's death, to Ro...
- **Gold passage** (403 chars): (“[I]t is one of the essential properties of a lease that its duration shall be for a determinate period, shorter than the duration of the estate of the lessor, hence the estate demised is called a ‘term’, and necessarily implies a reversion. If the entire interest of the lessor is conveyed, in the whole or a portion of his land, the conveyance cannot therefore be properly regarded as a demise ...
- **gold_idx**: `mbe_2711` | **gold_retrieved (snap_hyde)**: False

**13. idx=mbe_1098** (correct='D', golden→'B', llm_only→'A', snap_hyde→'D')

- **Q**: For this question only, make the following assumptions. On March 1, Fixtures tendered 24 sets to Apartments and explained, "One of the 25 sets was damaged in transit from the manufacturer to us, but we will deliver a replacement within 5 days." Which of the following statements is correct?
- **Gold passage** (500 chars): If one party to a contract materially breaches that contract, the aggrieved party may cancel the contract and be relieved of its obligation under the contract. Curt Ogden Equip. Co. v. Murphy Leasing Co., Inc., 895 S.W.2d 604, 609 (Mo.App.1995). “A material breach is one where the breach relates to a vital provision (i.e., material term) of the agreement and cannot relate simply to a subordinat...
- **gold_idx**: `mbe_838` | **gold_retrieved (snap_hyde)**: False

**14. idx=mbe_1100** (correct='D', golden→'A', llm_only→'D', snap_hyde→'D')

- **Q**: Olivia owned Blackacre, her home. Her daughter, Dawn, lived with her and always referred to Blackacre as "my property." Two years ago, Dawn, for a valuable consideration, executed and delivered to Bruce an instrument in the proper form of a warranty deed purporting to convey Blackacre to Bruce in fee simple, reserving to herself an estate for two years in Blackacre. Bruce promptly and properly ...
- **Gold passage** (276 chars): A purchaser is bound by every recital, reference, and reservation contained in or fairly disclosed by any instrument that forms an essential link in the chain of title under which the purchaser claims. See Westland Oil Dev. Corp. v. Gulf Oil, 637 S.W.2d 903, 908 (Tex. 1982).
- **gold_idx**: `mbe_839` | **gold_retrieved (snap_hyde)**: False

**15. idx=mbe_1102** (correct='D', golden→'A', llm_only→'D', snap_hyde→'D')

- **Q**: The warden of State Prison prohibits the photographing of the face of any prisoner without the prisoner's consent. Photographer, a news photographer, wanted to photograph Mobster, a notorious organized crime figure incarcerated at State Prison. To circumvent the warden's prohibition, Photographer flew over the prison exercise yard and photographed Mobster. Prisoner, who was imprisoned for a tec...
- **Gold passage** (492 chars): To satisfy the first element of an IIED claim, a plaintiff must demonstrate that the defendant either “desired to inflict severe emotional distress, knew that such distress was certain or substantially certain to result from his conduct, or acted recklessly in deliberate disregard of a high degree of probability that emotional distress would follow.” Interphase Garment Solutions, LLC, 566 F.Sup...
- **gold_idx**: `mbe_840` | **gold_retrieved (snap_hyde)**: False

**16. idx=mbe_1112** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: Ven owned Goldacre, a tract of land, in fee simple. Ven and Pur entered into a written agreement under which Pur agreed to buy Goldacre for $100,000, its fair market value. The agreement contained all the essential terms of a real estate contract to sell and buy, including a date for closing. The required $50,000 down payment was made. The contract provided that in the event of Pur's breach, Ve...
- **Gold passage** (500 chars): To establish a claim for conversion, a plaintiff must prove that (1) the plaintiff owned or had possession of the property or entitlement to possession; (2) the defendant unlawfully and without authorization assumed and exercised control over the property to the exclusion of, or inconsistent with, the plaintiff's rights as an owner; (3) the plaintiff demanded return of the property; and (4) the...
- **gold_idx**: `mbe_848` | **gold_retrieved (snap_hyde)**: False

**17. idx=mbe_1148** (correct='A', golden→'D', llm_only→'A', snap_hyde→'A')

- **Q**: Thirty years ago Able, the then-record owner of Greenacre, a lot contiguous to Blueacre, in fee simple, executed and delivered to Baker an instrument in writing which was denominated "Deed of Conveyance." In pertinent part it read, "Able does grant to Baker and her heirs and assigns a right-of-way for egress and ingress to Blueacre." If the quoted provision was sufficient to create an interest ...
- **Gold passage** (500 chars): “To establish a prescriptive easement, the claimant must prove by clear and convincing evidence: ‘(1) the continued and uninterrupted use or enjoyment of the right for a period of [twenty] years; (2) the identity of the thing enjoyed; and (3) the use [was] adverse under claim of right.’ ” Simmons v. Berkeley Elec. Coop., Inc., Op. No. 27674 (S.C. Sup. Ct. filed Nov. 2, 2016) (Shearouse Adv. Sh....
- **gold_idx**: `mbe_872` | **gold_retrieved (snap_hyde)**: False

**18. idx=mbe_1158** (correct='B', golden→'D', llm_only→'D', snap_hyde→'B')

- **Q**: The governor of the state of Green proposes to place a Christmas nativity scene, the components of which would be permanently donated to the state by private citizens, in the Green Capitol Building rotunda where the Green Legislature meets annually. The governor further proposes to display this state- owned nativity scene annually from December 1 to December 31, next to permanent displays that ...
- **Gold passage** (500 chars): In this case, the focus of our inquiry must be on the crèche in the context of the Christmas season. See, e.g., Stone v. Graham, 449 U.S. 39, 101 S.Ct. 192, 66 L.Ed.2d 199 (1980) (per curiam ); Abington School District v. Schempp, supra. In Stone, for example, we invalidated a state statute requiring the posting of a copy of the Ten Commandments on public classroom walls. But the Court carefull...
- **gold_idx**: `mbe_881` | **gold_retrieved (snap_hyde)**: False

**19. idx=mbe_1161** (correct='A', golden→'C', llm_only→'A', snap_hyde→'A')

- **Q**: Three years ago Adam conveyed Blackacre to Betty for $50,000 by a deed that provided: "By accepting this deed, Betty covenants for herself, her heirs and assigns, that the premises herein conveyed shall be used solely for residential purposes and, if the premises are used for nonresidential purposes, Adam, his heirs and assigns, shall have the right to repurchase the premises for the sum of one...
- **Gold passage** (364 chars): The rule against perpetuities is generally stated with deceptive simplicity as follows: “No interest is good unless it must vest, if at all, not later than twenty-one years after some life in being at the creation of the interest.” Iglehart v. Phillips, 383 So.2d 610, 614 (Fla.1980) (quoting John Chipman Gray, The Rule Against Perpetuities, § 201 (4th ed.1942)).
- **gold_idx**: `mbe_884` | **gold_retrieved (snap_hyde)**: False

**20. idx=mbe_117** (correct='C', golden→'D', llm_only→'C', snap_hyde→'C')

- **Q**: Doctor, a licensed physician, resided in her own home. The street in front of the home had a gradual slope. Doctor's garage was on the street level, with a driveway entrance from the street. At two in the morning Doctor received an emergency call. She dressed and went to the garage to get her car and found a car parked in front of her driveway. That car was occupied by Parker, who, while intoxi...
- **Gold passage** (500 chars): Kansas courts discussed assumption of risk as early as 1898. See Greef Bros. v. Brown, 7 Kan.App. 394, 51 P. 926 (1898). But in those early cases it was viewed as a “species of contributory negligence.” Greef Bros., 7 Kan.App. at 398, 51 P. 926 (discussing relationship between assumption of risk and contributory negligence). Contributory negligence is defined as “conduct on the part of the plai...
- **gold_idx**: `mbe_95` | **gold_retrieved (snap_hyde)**: False
