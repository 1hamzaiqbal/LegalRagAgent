# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | logical_calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| barexam | rag_simple | 10 | 10 | 0 | 0 | 0 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| barexam | snap_hyre | 10 | 10 | 0 | 0 | 0 | 1.00 | 1.00 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.1000 | 0.0500 | 0.1000 | 0.1000 | 0.0500 |
| barexam | snap_hyre_exemplar_parallel3 | 10 | 10 | 0 | 0 | 0 | 3.00 | 3.00 | 0.0000 | 0.0000 | 0.0000 | 0.2000 | 0.2000 | 0.0700 | 0.2000 | 0.2000 | 0.0700 |
| barexam | snap_hyre_exemplar_parallel3_llm_judge | 10 | 10 | 0 | 0 | 0 | 1.00 | 4.00 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 |
