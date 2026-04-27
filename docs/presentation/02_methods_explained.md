# Methods Explained

## rag_simple — original-question RAG baseline

`rag_simple` retrieves passages with the original MuSiQue question, places those passages in the answer prompt, and asks Llama 70b for one final answer. It is the control row for every paired comparison: same dataset slice, same model/provider, no snap answer, no HyDE passage, no planning table, and no subagent gap report. Its job is not to be clever; it is to set the stable retrieval baseline that other methods must beat under paired McNemar.

Result on Llama 70b N=200: 27.5% EM, baseline, p=n/a, sign-off APPROVED baseline. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__835710_7298`; source `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl:6`.

```json
{"label":"mq_2hop__835710_7298","idx":"2hop__835710_7298","question":"Along with the Closer performer, what notable pop artist started out his career on adult contemporary radio?","predicted_answer":"Michael Bublé","correct_answer":"Michael Bublé","is_correct":true}
```

## multi_hyde_diverse — pooled diverse HyDE

`multi_hyde_diverse` generates several hypothetical answer-bearing passages, then retrieves over those passages plus the raw question before one final synthesis call. The method targets a common multi-hop failure: the original question may underspecify the bridge entity, while answer-shaped hypotheses can name plausible bridges and pull better evidence. The Tier 2 audit found 199/200 rows with exactly three HyDE passages and no answer-artifact contamination, so the lift is a method signal rather than a logging artifact.

Result on Llama 70b N=200: 35.5% EM, +8.0pp, p=0.0195, sign-off APPROVED paper headline. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__96414_47902`; source `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl:11`.

```json
{"label":"mq_2hop__96414_47902","idx":"2hop__96414_47902","question":"Who is the actor who plays the title character of The Bourne Ultimatum?","predicted_answer":"Matt Damon","correct_answer":"Matt Damon","is_correct":true,"hyde_passages_count":3}
```

## iterative_planning_table — structured multi-hop planning

`iterative_planning_table` turns a question into explicit TODO rows, retrieves evidence for each row, records findings, and synthesizes from the filled table. It tests whether multi-hop QA improves when intermediate state is made concrete rather than left implicit in one answer prompt. On Llama 70b it has the best raw EM in the matrix, but the paired p-value sits just above 0.05, so it should be described as trending rather than fully significant.

Result on Llama 70b N=200: 36.0% EM, +8.5pp, p=0.0533, sign-off APPROVED TRENDING-SIG. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__142699_67465`; source `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl:13`.

```json
{"label":"mq_2hop__142699_67465","idx":"2hop__142699_67465","question":"When did the rapper on On and On and Beyond release Best day Ever?","predicted_answer":"March 11, 2011","correct_answer":"March 11, 2011","is_correct":true,"planning_rows":2}
```

## rag_multi_query — pooled query rewrites

`rag_multi_query` asks the model to rewrite the original question into multiple search queries, pools retrieval, and then answers once. It is the mechanism control for `multi_hyde_diverse`: if diversity alone caused the mhd lift, multi-query retrieval should have approached the mhd row. It did not. On the signed Llama 70b matrix, query diversity alone is small and non-significant, which makes HyDE-style answer-bearing passages the dominant mechanism.

Result on Llama 70b N=200: 29.0% EM, +1.5pp, p=0.728, sign-off APPROVED mechanism decomposition. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__86689_728109`; source `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl:2`.

```json
{"label":"mq_2hop__86689_728109","idx":"2hop__86689_728109","question":"What team drafted the winner of the NBA scoring title this year?","predicted_answer":"Houston Rockets","correct_answer":"Oklahoma City Thunder","is_correct":false,"rewrites_count":2}
```

## rag_snap_hyde — snap answer then HyDE

`rag_snap_hyde` first asks for a quick answer, converts that snap into a HyDE-style passage, retrieves from that passage plus the question, and then synthesizes. It is the signed BarExam winner, but it does not transfer to Llama 70b MuSiQue. The BarExam caveat is architectural: snap reasoning dominates because Gemma 4 has strong legal-MC priors; HyDE provides marginal lift and can hurt when it conflicts. That is mechanism understanding, not contamination.

Result on Llama 70b N=200: 24.0% EM, -3.5pp, p=0.36, sign-off APPROVED cross-domain negative. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__511296_2684`; source `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl:62`.

```json
{"label":"mq_2hop__511296_2684","idx":"2hop__511296_2684","question":"What was the name of the law passed by the actor from Terminator?","snap_answer":"Answer: AB 1881","predicted_answer":"None","correct_answer":"\"Donda West Law\"","is_correct":false}
```

## iter_hyde — iterative HyDE refinement

`iter_hyde` repeats the HyDE idea in rounds: generate a focused hypothetical passage, retrieve evidence, record a finding, and optionally continue before final synthesis. The design tries to repair missing bridge facts that one synthetic passage might miss. On Llama 70b N=200, the method is clean but does not improve the baseline. The example shows a typical near miss: the chain identifies Oregon State and the Beavers, but the final answer returns the team nickname instead of the mascot name.

Result on Llama 70b N=200: 24.5% EM, -3.0pp, p=0.47, sign-off APPROVED multi-round neutral. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__846844_47134`; source `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl:142`.

```json
{"label":"mq_2hop__846844_47134","idx":"2hop__846844_47134","question":"What is the mascot of the school that owns Goss Stadium at Coleman Field?","predicted_answer":"Beavers","correct_answer":"Benny Beaver","is_correct":false,"hyde_chain_rows":1}
```

## advisor_planning_table — cheap advisor plan

`advisor_planning_table` asks for an advisor-style decomposition before final answering, but it uses a cheaper planning form than the iterative planning table. It tests whether simply naming TODOs is enough. The signed Llama result says no: the data are clean, but the method is negative and non-significant. This makes it useful as a control: planning syntax alone is not the winning ingredient; the stronger table likely helps because it ties TODOs to retrieved findings.

Result on Llama 70b N=200: 23.0% EM, -4.5pp, p=0.222, sign-off APPROVED NS but informative negative. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `4hop1__567737_141375_458768_33633`; source `logs/eval_advisor_planning_table_groq-llama70b_20260427_1216_detail.jsonl:123`.

```json
{"label":"mq_4hop1__567737_141375_458768_33633","idx":"4hop1__567737_141375_458768_33633","question":"When did the explorer reach the headquarters location of the group Study in Brown's record label is part of?","predicted_answer":"No information available","correct_answer":"August 3, 1769","is_correct":false,"planning_rows":3}
```

## subagent_rag — gap-routing subagent RAG

`subagent_rag` identifies information gaps, runs targeted gap queries, collects reports, and asks the final answerer to use those reports. The idea fits multi-hop QA, but this implementation over-triggers: the audit found gap routing on 200/200 records and 29.5% "Unknown/Not found" finals. The -12pp result is real for this implementation, but it should be framed as systematic over-abstention from the current gap prompt, not proof that subagent methods cannot work.

Result on Llama 70b N=200: 15.5% EM, -12.0pp, p=0.0007, sign-off APPROVED sig negative with implementation caveat. Source: `docs/signoff_log.md:Section B.1`.

Example: question_id/idx `2hop__13592_49388`; source `logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl:89`.

```json
{"label":"mq_2hop__13592_49388","idx":"2hop__13592_49388","question":"How many games in a season of the league in which Barcelona won titles in 1948 and 1949?","predicted_answer":"26","correct_answer":"38","is_correct":false,"gaps":1,"gap_results":1}
```
