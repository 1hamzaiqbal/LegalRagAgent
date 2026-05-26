# MedQA-USMLE Widening - 2026-05-26

## Phase 0 - Download And Normalization

Downloaded online Hugging Face sources and normalized them to local CSVs. Later phases should use offline HF mode unless a missing model artifact must be explicitly bootstrapped.

| Artifact | Source | Local path | Rows | Notes |
|---|---|---|---:|---|
| MedQA questions | `GBaker/MedQA-USMLE-4-options`, test split | `datasets/medqa_usmle/questions.csv` | 1273 | Four options, gold answer normalized to A-D. |
| Textbook corpus | `MedRAG/textbooks`, train split | `datasets/medqa_usmle/textbooks.csv` | 125847 | Pre-chunked textbook snippets, `idx` preserves MedRAG `id`; retrieval text uses `contents`. |

Question `meta_info` counts:

| meta_info | rows |
|---|---:|
| step1 | 679 |
| step2&3 | 594 |

Textbook title counts:

| title | chunks |
|---|---:|
| Anatomy_Gray | 3017 |
| Biochemistry_Lippinco | 1973 |
| Cell_Biology_Alberts | 7070 |
| First_Aid_Step1 | 850 |
| First_Aid_Step2 | 1369 |
| Gynecology_Novak | 7947 |
| Histology_Ross | 4411 |
| Immunology_Janeway | 4852 |
| InternalMed_Harrison | 32628 |
| Neurology_Adams | 12370 |
| Obstentrics_Williams | 9166 |
| Pathology_Robbins | 5297 |
| Pathoma_Husain | 505 |
| Pediatrics_Nelson | 4260 |
| Pharmacology_Katzung | 7356 |
| Physiology_Levy | 4370 |
| Psichiatry_DSM-5 | 4057 |
| Surgery_Schwartz | 14349 |

Question examples:

- `medqa_test_0000` answer `B` / Tell the attending that he cannot fail to disclose this mistake: A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired
- `medqa_test_0001` answer `D` / Cross-linking of DNA: A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. P

Passage examples:

- `Anatomy_Gray_0` (Anatomy_Gray): Anatomy_Gray. What is anatomy? Anatomy includes those structures that can be seen grossly (without the aid of magnification) and microscopically (with the aid of magnification). Typically, when used by itself, the term anatomy tends to mean
- `Anatomy_Gray_1` (Anatomy_Gray): Anatomy_Gray. Observation and visualization are the primary techniques a student should use to learn anatomy. Anatomy is much more than just memorization of lists of names. Although the language of anatomy is important, the network of infor

## Current Status

- Phase 0 complete.
- Phase 1 pending: embed `datasets/medqa_usmle/textbooks.csv` into Chroma collection `medqa_textbooks` with `Alibaba-NLP/gte-large-en-v1.5`.
- MedQA has no gold passage labels; downstream answer EM is the primary outcome and no Hit@k/MRR/Recall will be reported.

## Reproduction

```bash
uv run python scripts/download_medqa_usmle.py
```
