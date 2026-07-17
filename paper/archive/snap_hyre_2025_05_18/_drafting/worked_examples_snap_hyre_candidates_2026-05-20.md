# Snap-HyRE Worked Example Candidates - 2026-05-20

Purpose: shortlist concrete passages/examples where Snap-HyRE fixes a miss from
raw RAG and HyDE. These are not yet manuscript prose; they are paper-drafting
candidates to vet and then convert into one compact qualitative figure/table.

## Recommended Main Example

### HousingQA Hawaii `idx=1066`

- Question: "Can a landlord evict a tenant for committing or failing to dispose
  of waste?"
- Model/dataset: `groq-llama8b` on HousingQA.
- Gold: `Yes`.
- Raw RAG: `No`; HyDE: `No`; Snap-HyRE: `Yes`.
- Snap-HyRE log:
  `logs/merged/eval_snap_hyre_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl:1616`.
- Raw RAG log:
  `logs/merged/eval_rag_simple_groq-llama8b_20260519_housing_nfull_k5_merged_detail.jsonl:1616`.
- HyDE log:
  `logs/eval_rag_hyde_groq-llama8b_20260519_204401_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl:1616`.

Why it is clean: raw RAG retrieves generic waste statutes from South Dakota,
Minnesota, and Nevada; HyDE moves toward Hawaii but lands on a tenant-remedy
provision. Snap-HyRE retrieves the exact Hawaii landlord-remedy statute as its
top passage:

- Snap top evidence: `1359768`, `HI Rev Stat § 521-69 (2021)`,
  "Landlord's remedies for tenant's waste, failure to maintain, or unlawful
  use."
- Raw top evidence: `310721`, `SD Codified L § 21-7-1 (2021)`, an out-of-state
  action-for-waste provision.
- HyDE top evidence: `1359761`, `HI Rev Stat § 521-62 (2021)`, a Hawaii
  tenant-remedy provision rather than the landlord waste remedy.

Suggested paper use: a three-column qualitative comparison showing generated
query, top retrieved passage, and final answer for Raw RAG, HyDE, and Snap-HyRE.
This is probably the best example because the jurisdiction, legal issue, and
retrieved statute all line up.

## Strong BarExamQA Candidates

### BarExamQA `idx=mbe_1159`

- Scenario: police chase, shouted stop command, discarded cocaine, suppression
  motion.
- Model/dataset: `groq-llama70b` on BarExamQA.
- Gold: `B`.
- Raw RAG: `D`; HyDE: `D`; Snap-HyRE: `B`.
- Snap-HyRE log:
  `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl:910`.

Why it works: raw RAG retrieves a nearby police-encounter fact pattern, and HyDE
over-focuses on abandonment. Snap-HyRE generates the governing seizure framing
and retrieves Fourth Amendment passages stating that a seizure occurs through
physical force or show of authority restricting liberty.

Useful contrast:

- Raw top evidence: `caselaw_12576207_1`, another officer-approach/drug case.
- HyDE top evidence: `caselaw_12531733_104`, abandoned-property doctrine.
- Snap top evidence: `caselaw_12531704_17`, Fourth Amendment seizure standard.

Suggested paper use: good narrative example for "Snap-HyRE fixes the legal
predicate before retrieval." It is more doctrinal than the HousingQA example,
but may need careful wording because the retrieved evidence is caselaw rather
than the original exam explanation.

### BarExamQA `idx=mbe_351`

- Scenario: vehicular manslaughter definition most favorable to defendant.
- Model/dataset: `groq-llama70b` on BarExamQA.
- Gold: `C`.
- Raw RAG: `D`; HyDE: `D`; Snap-HyRE: `C`.
- Snap-HyRE log:
  `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl:281`.

Why it works: raw RAG retrieves broad vehicular-manslaughter statute snippets,
and HyDE retrieves general criminal-negligence material. Snap-HyRE retrieves
MBE-style explanations focused on gross/wanton culpability and the higher
threshold most favorable to the defendant.

Suggested paper use: good shorter example if we want an exam-domain case that
looks like legal reasoning rather than jurisdiction matching.

### BarExamQA `idx=mbe_459`

- Scenario: recording statute, estoppel by deed, and after-acquired title.
- Model/dataset: `groq-llama70b` on BarExamQA.
- Gold: `D`.
- Raw RAG: `B`; HyDE: `B`; Snap-HyRE: `D`.
- Snap-HyRE log:
  `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl:368`.

Why it works: raw RAG retrieves case-specific title-chain noise, HyDE retrieves
estoppel-by-deed doctrine but still answers incorrectly, and Snap-HyRE retrieves
both recording-statute and after-acquired-title explanations. This is useful if
the paper needs an example where retrieval is necessary but not sufficient.

Caveat: the Snap-HyRE intermediate answer appears to lean toward the wrong
option before the final answer corrects it. Use only if we want to discuss
answer-stage correction; otherwise prefer `mbe_1159` or HousingQA `1066`.

## Additional HousingQA Candidates

### HousingQA Kansas `idx=3651`

- Question: "Is it specified what is required to be in the notice?"
- Model/dataset: `groq-llama8b` on HousingQA.
- Gold: `Yes`.
- Raw RAG: `No`; HyDE: `No`; Snap-HyRE: `Yes`.
- Snap-HyRE log:
  `logs/merged/eval_snap_hyre_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl:2314`.

Why it may work: Snap-HyRE retrieves `KS Stat § 58-2564` in the top three,
where raw and HyDE drift into generic notice provisions or unrelated Kansas
procedure. This may be useful as a state-specific retrieval example.

Caveat: the first Snap-HyRE passage is Maryland, not Kansas, so this needs
manual vetting before becoming a headline example.

### HousingQA New Mexico `idx=8187`

- Question: "Is the term 'Restitution of premises' used to refer to the order
  from the court to the authorities to remove a tenant?"
- Model/dataset: `groq-llama8b` on HousingQA.
- Gold: `No`.
- Raw RAG: `Yes`; HyDE: `Yes`; Snap-HyRE: `No`.
- Snap-HyRE log:
  `logs/merged/eval_snap_hyre_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl:4148`.

Why it may work: raw RAG and HyDE retrieve statutes from other jurisdictions or
New Mexico passages that use "restitution" broadly, while Snap-HyRE retrieves a
New Mexico writ-of-possession passage and answers that the exact term is not
the court-to-authorities phrase.

Caveat: the generated passage itself leans toward the wrong answer before the
final answer flips to `No`. This is interesting but less clean than Hawaii
`1066`.

## Suggested Qualitative Figure

Use one compact table with columns:

1. Example/question.
2. Method.
3. Generated/retrieval query summary.
4. Top retrieved passage.
5. Prediction vs. gold.

Start with HousingQA `1066` plus BarExamQA `mbe_1159`. Add `mbe_351` only if
there is enough space or if we want a three-example narrative: jurisdiction
matching, doctrinal framing, and culpability-threshold framing.
