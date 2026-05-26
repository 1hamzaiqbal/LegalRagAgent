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

## Phase 3 - Perplexity Pre-Screen

Add-1 unigram LMs were built over the retrieval corpora and scored against intermediate-generation question text. MedQA uses `medqa_textbooks`; BarExamQA and HousingQA reuse the existing perplexity-axis LM cache.

| Dataset | Questions | LM scope | Median PPL | IQR PPL | Mean log PPL | Mean OOV rate | Median tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| BarExamQA | 1195 | corpus-wide | 1898.4 | 1403.4-2597.1 | 7.564 | 0.9% | 193 |
| HousingQA state-filtered | 6853 | per state | 1434.8 | 1027.5-2096.4 | 7.320 | 0.7% | 22 |
| MedQA-USMLE | 1273 | corpus-wide | 2650.6 | 2257.4-3093.5 | 7.886 | 0.2% | 152 |

Separation checks on log-perplexity:

| Comparison | AUC first > second | Cohen's d | Mean log-PPL gap |
|---|---:|---:|---:|
| MedQA > HousingQA | 0.841 | 0.71 | 0.565 |
| MedQA > BarExamQA | 0.738 | 0.90 | 0.322 |

Reading:

- MedQA is materially higher than HousingQA on the corpus-surprise pre-screen by the configured gate: mean log-PPL gap 0.565, AUC 0.841.
- MedQA median token count is 152, compared with HousingQA 22 and BarExamQA 193.
- Gate decision: continue to q200 downstream answer probe.

Reproduction:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run python scripts/analyze_medqa_prescreen.py
```

## Phase 4 - q200 Downstream Probe

The q200 Groq Llama 70B probe used cached retrieval/generation artifacts where applicable and strict final-answer checks. MedQA has no gold passage labels, so this table reports downstream exact-match answer accuracy only.

| Method | Detail log | N | Correct | EM | Delta vs raw | McNemar vs raw | Health |
|---|---|---:|---:|---:|---:|---:|---|
| LLM only | `logs/eval_llm_only_groq-llama70b_20260526_013008_medqa_local-snap-hyre-groq-llama70b-medqa-llm_only-n200-k5_detail.jsonl` | 200 | 156 | 78.0% | +1.5pp | -- | clean |
| Raw question RAG | `logs/eval_rag_simple_groq-llama70b_20260526_013500_medqa_local-snap-hyre-groq-llama70b-medqa-rag_simple-n200-k5_detail.jsonl` | 200 | 153 | 76.5% | -- | -- | clean |
| HyDE | `logs/eval_rag_hyde_groq-llama70b_20260526_013944_medqa_local-snap-hyre-groq-llama70b-medqa-rag_hyde-n200-k5_detail.jsonl` | 200 | 161 | 80.5% | +4.0pp | p=0.1338 | clean; 1 final-answer format retry |
| Snap-HyRE / SCOPE | `logs/eval_snap_hyre_groq-llama70b_20260526_014411_medqa_local-snap-hyre-groq-llama70b-medqa-snap_hyre-n200-k5_detail.jsonl` | 200 | 167 | 83.5% | +7.0pp | p=0.00661 | clean; 1 final-answer format retry |

Pairwise SCOPE checks:

| Comparison | N | Baseline EM | SCOPE EM | Delta | b | c | McNemar p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCOPE vs raw RAG | 200 | 76.5% | 83.5% | +7.0pp | 19 | 5 | 0.00661 | +2.5 to +12.0pp |
| SCOPE vs HyDE | 200 | 80.5% | 83.5% | +3.0pp | 14 | 8 | 0.2863 | -1.5 to +7.5pp |
| SCOPE vs LLM only | 200 | 78.0% | 83.5% | +5.5pp | 15 | 4 | 0.0192 | +1.5 to +9.5pp |

Generation and retrieval caches:

- `caches/retrieval/full/medqa_q200_seed42_raw_question_k10.jsonl`
- `caches/hyre/full/medqa_q200_seed42_groq-llama70b_rag_hyde.jsonl`
- `caches/retrieval/full/medqa_q200_seed42_groq-llama70b_rag_hyde_k10.jsonl`
- `caches/hyre/full/medqa_q200_seed42_groq-llama70b_snap_hyre.jsonl`
- `caches/retrieval/full/medqa_q200_seed42_groq-llama70b_snap_hyre_k10.jsonl`

Gate decision: q200 clears the widening gate. SCOPE beats raw RAG by +7.0pp with McNemar p=0.00661 and also beats LLM-only by +5.5pp. Per the 2026-05-26 plan change, q200 is the banked downstream answer result for now; full-N answer cells are deferred to the cheaper parallel OpenRouter runner and refreshed model set.

## Full-N Reusable Caches

After the q200 gate cleared, the full 1273-question MedQA cache inputs were built for the Llama 70B arm. These are reusable artifacts only; no full-N MedQA answer accuracy is reported from them here.

| Artifact | Rows | Purpose |
|---|---:|---|
| `caches/retrieval/full/medqa_qfull_seed42_raw_question_k10.jsonl` | 1273 | Raw-question top-10 retrieval cache for raw RAG. |
| `caches/hyre/full/medqa_qfull_seed42_groq-llama70b_rag_hyde.jsonl` | 1273 | Llama 70B HyDE generated passages. |
| `caches/retrieval/full/medqa_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | 1273 | Top-10 retrieval over the HyDE generated passages. |
| `caches/hyre/full/medqa_qfull_seed42_groq-llama70b_snap_hyre.jsonl` | 1273 | Llama 70B Snap-HyRE/SCOPE draft-plus-passage generations. |
| `caches/retrieval/full/medqa_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | 1273 | Top-10 retrieval over the Snap-HyRE/SCOPE passage block. |

Health notes:

- Full raw retrieval cache: 1273/1273 rows.
- Full HyDE generation cache: 1273/1273 rows, zero failures.
- Full HyDE retrieval cache: 1273/1273 rows.
- Full Snap-HyRE/SCOPE generation cache: 1273/1273 rows, zero failures after enabled same-model format retry.
- Full Snap-HyRE/SCOPE retrieval cache: 1273/1273 rows.
- Full-N answer cells are intentionally not banked in this report.

## Current Status

- Phase 0 complete.
- Phase 1 complete: embedded `datasets/medqa_usmle/textbooks.csv` into Chroma collection `medqa_textbooks` with `Alibaba-NLP/gte-large-en-v1.5`.
- Chroma verification: `medqa_textbooks` contains 125847 documents. Sample metadata preserves `idx`, `source=medrag_textbooks`, and `title`.
- Embedding run settings: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 EMBED_CHUNK=5000 EMBED_GPU_BATCH=32`; total runtime 48.5 minutes on the local RTX 3070 Laptop GPU.
- Phase 2 complete: registered MedQA in the eval loader, prompt formatter, Chroma collection map, generation/retrieval cache builders, and local answer-cell health checks.
- Formatter sanity check: `uv run python tests/test_formatter.py` passes 13/13, including MedQA four-option prompt and intermediate-generation checks.
- Phase 3 complete: perplexity pre-screen clears the q200 answer-budget gate.
- Phase 4 q200 complete: Groq Llama 70B SCOPE is 167/200, ahead of raw RAG 153/200 and HyDE 161/200.
- Full-N reusable caches complete for raw retrieval plus Groq Llama 70B HyDE and Snap-HyRE/SCOPE.
- Full-N downstream answer scoring stopped by plan change; answer scale-up is deferred.
- MedQA has no gold passage labels; downstream answer EM is the primary outcome and no Hit@k/MRR/Recall will be reported.

## Reproduction

```bash
uv run python scripts/download_medqa_usmle.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  EMBED_CHUNK=5000 EMBED_GPU_BATCH=32 \
  uv run python utils/fast_embed.py medqa
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run python tests/test_formatter.py
```
