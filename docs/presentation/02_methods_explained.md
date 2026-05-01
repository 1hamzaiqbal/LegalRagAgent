# Methods Explained

## rag_simple — original-question RAG baseline

What it does: retrieves with the original MuSiQue question, then answers once. It is the paired control for all Llama 70b method deltas.
Result: Llama 70b N=200 baseline 27.5% EM; sign-off APPROVED baseline.
Example: `2hop__835710_7298` asks for the pop artist who began on adult contemporary radio; predicted `Michael Bublé`, gold `Michael Bublé`; source `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl:6`.

## multi_hyde_diverse — pooled diverse HyDE

What it does: generates answer-bearing hypothetical passages, retrieves over those passages plus the raw question, then answers once.
Result: Llama 70b N=200 35.5% EM, +8pp, p=0.0195; sign-off APPROVED as a strong positive pseudo-doc arm, now secondary to `snap_hyde_2call` and the bottleneck-taxonomy framing.
Example: `2hop__96414_47902` asks who played the title character in The Bourne Ultimatum; predicted `Matt Damon`, gold `Matt Damon`, `hyde_passages_count=3`; source `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl:11`.

## iterative_planning_table — structured multi-hop planning

What it does: writes TODO rows, retrieves evidence per row, records findings, then synthesizes from the filled table.
Result: Llama 70b N=200 36.0% EM, +8.5pp, p=0.0533 TRENDING-SIG; sign-off APPROVED.
Example: `2hop__142699_67465` asks when the On and On and Beyond rapper released Best Day Ever; predicted `March 11, 2011`, gold `March 11, 2011`, `planning_rows=2`; source `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl:13`.

## rag_multi_query — pooled query rewrites

What it does: rewrites the question into multiple search queries, pools retrieval, then answers once. It isolates query diversity from HyDE-style passages.
Result: Llama 70b N=200 29.0% EM, +1.5pp, p=0.728; sign-off APPROVED mechanism control.
Example: `2hop__86689_728109` asks which team drafted the NBA scoring-title winner; predicted `Houston Rockets`, gold `Oklahoma City Thunder`, `rewrites_count=2`; source `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl:2`.

## rag_snap_hyde — snap answer then HyDE

What it does: asks for a snap answer, turns that snap into a HyDE passage, retrieves with snap+question, then synthesizes. On BarExam this is BY DESIGN architecture and mechanism understanding.
Result: Llama 70b MuSiQue N=200 24.0% EM, -3.5pp, p=0.36; sign-off APPROVED cross-domain negative.
Example: `2hop__511296_2684` asks for the law passed by the actor from Terminator; snap `AB 1881`, predicted `None`, gold `"Donda West Law"`; source `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl:62`.

## iter_hyde — iterative HyDE refinement

What it does: repeats HyDE generation, retrieval, and finding capture before final synthesis.
Result: Llama 70b N=200 24.5% EM, -3.0pp, p=0.47; sign-off APPROVED multi-round neutral.
Example: `2hop__846844_47134` asks for the mascot of the school owning Goss Stadium; predicted `Beavers`, gold `Benny Beaver`, `hyde_chain_rows=1`; source `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl:142`.

## advisor_planning_table — cheap advisor plan

What it does: asks for a lightweight advisor decomposition before final answering; it tests planning syntax without the full iterative table.
Result: Llama 70b N=200 23.0% EM, -4.5pp, p=0.222; sign-off APPROVED NS negative.
Example: `4hop1__567737_141375_458768_33633` asks when an explorer reached a record-label headquarters location; predicted `No information available`, gold `August 3, 1769`, `planning_rows=3`; source `logs/eval_advisor_planning_table_groq-llama70b_20260427_1216_detail.jsonl:123`.

## subagent_rag — gap-routing subagent RAG

What it does: detects information gaps, runs gap queries, collects reports, then asks the final answerer to use them.
Result: Llama 70b N=200 15.5% EM, -12.0pp, p=0.0007; real finding with gap-routing over-abstention caveat.
Example: `2hop__13592_49388` asks how many games were in the league season Barcelona won titles in 1948 and 1949; predicted `26`, gold `38`, `gaps=1`, `gap_results=1`; source `logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl:89`.
