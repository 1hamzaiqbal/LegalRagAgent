# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| barexam | rag_simple | 20 | 20 | 0 | 0 | 0 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| barexam | snap_hyre | 20 | 20 | 0 | 0 | 0 | 1.00 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.0500 | 0.0250 | 0.0500 | 0.0500 | 0.0250 |
| barexam | snap_hyre_exemplar_parallel3 | 20 | 20 | 0 | 0 | 0 | 3.00 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.0500 | 0.0250 | 0.1500 | 0.1500 | 0.0371 |
