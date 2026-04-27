# Log Quality Audit 2026-04-27 ~03:05 CDT

## Summary
- Audited 5 detail logs, sampling 38 records total: 10/30 friend_foe, 9/30 iter_hyde Llama, 8/100 multi_hyde_diverse Scout, 8/100 rag_simple Scout, and 3/3 iter_hyde Gemma smoke.
- Verdicts: friend_foe = minor concern; iter_hyde Llama = minor concern; multi_hyde_diverse Scout = minor concern; rag_simple Scout = minor concern; iter_hyde Gemma smoke = clean, limited by N=3.
- No sampled record showed an empty `predicted_answer`, placeholder prediction, visible `<think>` leakage, or an answer tail ending mid-token.
- The largest validity issue is not truncation. It is that raw `final_answer` fields often contain essays, markdown headings, or explicit external-knowledge reasoning despite MuSiQue's brief-span instruction. The parsed `predicted_answer` fields are brief and usable for answer metrics.
- Friend/foe attribution-bias check: sampled review passes did not collapse to byte-identical outputs. In the 10 sampled friend_foe records, all three review fields were populated, 0/10 review triplets were identical, and `mq_2hop__622308_61845` showed a material split: self changed the answer to `John Moncur`, while foe/control kept `Frank Lampard Sr.`.

## Per-log findings

### 1. logs/eval_friend_foe_attribution_or-gemma27b_20260427_0249_detail.jsonl
- Sampled 10/30 records: `mq_2hop__121145_561444`, `mq_3hop1__68732_39743_24526`, `mq_2hop__96062_159673`, `mq_3hop2__230_89048_66294`, `mq_2hop__142699_67465`, `mq_2hop__622308_61845`, `mq_4hop2__161602_474028_88460_18966`, `mq_4hop2__161602_474028_88460_20999`, `mq_2hop__418941_429437`, `mq_3hop1__654651_55349_651302`.
- Truncation observations: sampled answer tails ended cleanly with answer spans, not mid-token. Examples: `mq_2hop__121145_561444` ended `Answer: The Ari`; `mq_3hop1__68732_39743_24526` ended `Answer: 40-50°F`; `mq_2hop__622308_61845` ended `Answer: John Moncur`. No sampled `<think>` tags.
- Empty/garbage observations: no sampled `predicted_answer` was empty, whitespace, `None`, or `<placeholder>`. Examples include `The Ari`, `40-50°F`, `The Thing (1982)`, `Joe Jackson`, `1999`, `John Moncur`.
- Format observations: raw `final_answer`, `snap_answer`, and review fields are not brief spans. They are review-style explanations plus a final `Answer:` line. Example `mq_3hop1__68732_39743_24526`: `Okay, let's review my previous reasoning... Therefore, I stand by my previous answer. Answer: 40-50°F`. This is acceptable only if cited metrics use parsed `predicted_answer`, not raw `final_answer` schema compliance.
- Reasoning trace observations: `snap_answer`, `self_review_answer`, `foe_review_answer`, and `control_review_answer` were populated in all sampled records. Some traces are sensible as reviews, but can be factually wrong. Example `mq_2hop__121145_561444` says `Rabbi Moshe Chaim Luzzatto (Ramchal), the author of *Derech Mitzvosecha*`, while the gold passage identifies Rabbi Menachem Mendel Schneersohn.
- Routed_to observations: `routed_to` field is absent in this log schema; no fallback flag was available to inspect.
- Friend_foe-specific: review passes do vary. In sampled records, all equality checks between self/foe/control review text were false. Most differences are tonal or attributional rather than outcome-changing, for example `mq_2hop__121145_561444` self says `The earlier response correctly identifies...`, foe says `Grok-2 correctly identifies...`, and control says `The candidate response correctly identifies...`, all ending `Answer: The Ari`. One sampled record has substantive divergence: `mq_2hop__622308_61845` snap said `Answer: Frank Lampard Sr.`, self review said `My previous answer focused on Charlton Athletic, which was incorrect... Answer: John Moncur`, while foe review said `Grok-2 correctly identifies Frank Lampard Sr.` and control said `the candidate's reasoning and answer appear to be correct. Answer: Frank Lampard Sr.`
- VERDICT: minor concern. The attribution-bias mechanism is present and not collapsed in the sample, but many review differences are shallow label/tone differences. Cite friend_foe results as parsed-answer results; use a separate all-record kept-snap/review-delta analysis before making a strong claim about attribution bias magnitude.

### 2. logs/eval_iter_hyde_groq-llama70b_20260427_0244_detail.jsonl
- Sampled 9/30 records: `mq_2hop__121145_561444`, `mq_3hop1__68732_39743_24526`, `mq_2hop__96062_159673`, `mq_3hop2__230_89048_66294`, `mq_2hop__142699_67465`, `mq_2hop__622308_61845`, `mq_2hop__6870_16335`, `mq_3hop2__304722_591230_63959`, `mq_3hop1__654651_55349_651302`.
- Truncation observations: sampled `final_answer` tails ended cleanly at answer spans. Examples: `mq_2hop__121145_561444` ended `Answer: Rabbi Dovber Schneuri`; `mq_3hop2__304722_591230_63959` ended `Answer: Unknown`; `mq_3hop1__654651_55349_651302` ended `Answer: Conifer, Colorado`. No sampled `<think>` tags.
- Empty/garbage observations: no sampled `predicted_answer` was empty, whitespace, `None`, or `<placeholder>`. `mq_3hop2__304722_591230_63959` had `predicted_answer` = `Unknown`; that is answer-shaped but low-value.
- Format observations: raw `final_answer` fields are verbose rather than strict brief spans. Example `mq_2hop__622308_61845` ended `we must acknowledge the limitations of our findings. Answer: Mido`. The extracted `predicted_answer` remained brief.
- Reasoning trace observations: expected `snap_text` and `iter_findings` fields are absent. This schema uses `hyde_chain`, which was populated in all sampled records. Sampled `hyde_chain` lengths matched `rounds_completed`: 3 for most records, 1 for early exits such as `mq_2hop__142699_67465` and `mq_3hop1__654651_55349_651302`. Some traces are explicitly speculative or unsupported, for example `mq_3hop2__230_89048_66294` final says `not directly supported by the provided findings`, and `mq_3hop2__304722_591230_63959` says `considering the lack of direct evidence`.
- Routed_to observations: `routed_to` field is absent in this log schema; no fallback flag was available to inspect.
- VERDICT: minor concern. No parse/truncation failure found in the sample. Trace quality includes unsupported inferences, so cite as answer-quality eval rather than as evidence-grounded reasoning quality.

### 3. logs/eval_multi_hyde_diverse_groq-scout_20260427_0249_detail.jsonl
- Sampled 8/100 records: `mq_2hop__121145_561444`, `mq_4hop1__166471_49925_13759_736921`, `mq_3hop1__654651_55349_651302`, `mq_3hop2__88342_93066_47738`, `mq_2hop__77878_189357`, `mq_2hop__46550_85990`, `mq_3hop1__617062_127905_87812`, `mq_2hop__748182_78303`.
- Truncation observations: sampled `final_answer` tails ended cleanly at answer spans. Examples: `mq_3hop2__88342_93066_47738` ended `Answer: 2009`; `mq_2hop__46550_85990` ended `Answer: Nancy Pelosi`; `mq_2hop__748182_78303` ended `Answer: Chuck Norris`. No sampled `<think>` tags.
- Empty/garbage observations: no sampled `predicted_answer` was empty, whitespace, `None`, or `<placeholder>`.
- Format observations: Scout raw answers are often full mini-essays, not brief spans. `mq_3hop2__88342_93066_47738` includes markdown-style structure ending `## 5: Conclusion ... Answer: 2009`; `mq_4hop1__166471_49925_13759_736921` contains numbered reasoning ending `Therefore, connecting all the dots: Answer: Saxony-Anhalt`. Parsed `predicted_answer` fields are brief.
- Reasoning trace observations: `hyde_passages`, `hyde_passages_raw`, and `retrieval_queries` were populated in sampled records with `n_hyde_passages=3`. `snap_text` and `iter_findings` are absent, which is expected for this mode. `rerank_query` was present but empty in every sampled record inspected. Some HyDE passages are answer-bearing guesses rather than neutral retrieval expansions; `mq_3hop1__654651_55349_651302` hyde says `Trey Parker was born in Conifer, Colorado`, and the final says `based on general knowledge not provided in the passages`.
- Routed_to observations: `routed_to` field is absent in this log schema; no fallback flag was available to inspect.
- VERDICT: minor concern. No truncation/empty-output issue found in the sample. HyDE traces are populated, but Scout frequently uses answer-bearing/generated knowledge; cite answer metrics carefully and avoid representing these traces as purely retrieved evidence.

### 4. logs/eval_rag_simple_groq-scout_20260427_0246_detail.jsonl
- Sampled 8/100 records: `mq_2hop__121145_561444`, `mq_4hop1__166471_49925_13759_736921`, `mq_3hop1__654651_55349_651302`, `mq_3hop2__88342_93066_47738`, `mq_2hop__77878_189357`, `mq_2hop__46550_85990`, `mq_3hop1__617062_127905_87812`, `mq_2hop__748182_78303`.
- Truncation observations: sampled answer tails ended cleanly at answer spans. Examples: `mq_2hop__121145_561444` ended `Answer: Rabbi Dovber Schneuri`; `mq_3hop1__617062_127905_87812` ended `Answer: late March / early April`; `mq_2hop__748182_78303` ended `Answer: Chuck Norris`. No sampled `<think>` tags.
- Empty/garbage observations: no sampled `predicted_answer` was empty, whitespace, `None`, or `<placeholder>`.
- Format observations: raw `final_answer` fields are verbose and sometimes markdown-structured. `mq_4hop1__166471_49925_13759_736921` includes `## 4: Connect the information to answer the question`; `mq_3hop1__654651_55349_651302` includes `## 6: External Information for Stephen Full`. Parsed `predicted_answer` fields are brief.
- Reasoning trace observations: `snap_text`, `hyde_passages`, and `iter_findings` fields are absent; this is a simple RAG log. `retrieved_ids`, `gold_retrieved`, and `evidence_store` are present in sampled records. A groundedness concern appears in sampled final answers: `mq_3hop1__654651_55349_651302` had `gold_retrieved=false` and says `External Information for Stephen Full`; `mq_2hop__77878_189357` says `based on general knowledge and not directly from the passages provided` even though `gold_retrieved=true`.
- Routed_to observations: `routed_to` field is absent in this log schema; no fallback flag was available to inspect.
- VERDICT: minor concern. No log-corruption issue found in the sample. The concern is interpretive: simple RAG outputs sometimes rely on external/general knowledge, so cite these results as final-answer accuracy, not as proof of retrieved-context grounding.

### 5. logs/eval_iter_hyde_or-gemma27b_20260427_0300_detail.jsonl
- Sampled 3/3 records because this is an N=3 smoke log: `mq_4hop1__58323_375563_161848_67585`, `mq_3hop2__57233_11282_56883`, `mq_4hop1__726152_153080_33897_81096`.
- Truncation observations: all three sampled `final_answer` tails ended cleanly at answer spans: `Answer: 0`, `Answer: Vietnam War era`, and `Answer: Josef Newgarden`. No sampled `<think>` tags.
- Empty/garbage observations: no sampled `predicted_answer` was empty, whitespace, `None`, or `<placeholder>`.
- Format observations: raw `final_answer` fields remain explanatory, not strict brief spans. Parsed `predicted_answer` values are brief.
- Reasoning trace observations: expected `snap_text` and `iter_findings` fields are absent. This schema uses `hyde_chain`, populated for all 3 records. `rounds_completed` matched observed chain length: 3, 3, and 2. Example `mq_4hop1__58323_375563_161848_67585` has a 3-item chain and ends `Therefore, the answer is zero stars. Answer: 0`.
- Routed_to observations: `routed_to` is present and `null` for all 3 sampled records; no fallback path was triggered in the smoke sample.
- VERDICT: clean, limited by N=3. The smoke log shows no repeat of an empty/truncated/fallback issue in the inspected records.

## Cross-cutting concerns
- Raw `final_answer` does not reliably follow the MuSiQue brief-span instruction across all five logs. The parsed `predicted_answer` field is the reliable field for metrics.
- No sampled outputs showed visible reasoning-token leakage via `<think>` or abrupt mid-answer cutoff.
- `snap_text` and `iter_findings` are absent in the sampled schemas. Iterative modes use `hyde_chain`; multi-HyDE uses `hyde_passages`; friend_foe uses `snap_answer` and three review-answer fields.
- `routed_to` is absent in four logs and present-but-null in the N=3 Gemma smoke log. No sampled record showed an unexpected fallback path.
- Scout logs show repeated external/general-knowledge language in raw answers, for example `based on general knowledge not provided in the passages` and `External Information for Stephen Full`. This is a groundedness concern if the cited result is meant to demonstrate retrieval-grounded answer generation.

## Recommendations
- No immediate re-run is needed for truncation, empty prediction, placeholder output, `<think>` leakage, or unexpected fallback based on this sample.
- Cite leaderboard/table results from `predicted_answer`, not raw `final_answer`, unless the writeup explicitly notes the verbose answer format.
- For friend_foe: the sampled 3-pass review did vary and did not collapse, including one substantive self-vs-foe/control disagreement. Before citing a strong attribution-bias effect size, run an all-30 structured comparison of `self_kept_snap`, `foe_kept_snap`, `control_kept_snap`, and final answer deltas.
- For RAG/HyDE groundedness claims: either avoid claiming strict retrieved-context grounding for the Scout logs, or re-run with stricter prompts/validators that reject `based on general knowledge` and require answers to be supported by retrieved passages.
