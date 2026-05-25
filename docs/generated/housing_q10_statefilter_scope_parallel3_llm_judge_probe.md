# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | logical_calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| housing | rag_simple | 10 | 10 | 0 | 0 | 0 | 0.00 | 0.00 | 0.4000 | 0.3333 | 0.4000 | 0.6000 | 0.4833 | 0.4750 | 0.7000 | 0.5333 | 0.4861 |
| housing | snap_hyre | 10 | 10 | 0 | 0 | 0 | 1.00 | 1.00 | 0.2000 | 0.1333 | 0.2000 | 0.6000 | 0.4333 | 0.3667 | 0.6000 | 0.4833 | 0.3667 |
| housing | snap_hyre_exemplar_parallel3 | 10 | 10 | 0 | 0 | 0 | 3.00 | 3.00 | 0.3000 | 0.2500 | 0.3000 | 0.4000 | 0.3500 | 0.3200 | 0.4000 | 0.3500 | 0.3200 |
| housing | snap_hyre_exemplar_parallel3_llm_judge | 10 | 10 | 0 | 0 | 0 | 1.00 | 4.00 | 0.5000 | 0.4000 | 0.5000 | 0.5000 | 0.4000 | 0.5000 | 0.5000 | 0.4000 | 0.5000 |
