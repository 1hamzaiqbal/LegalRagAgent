# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| housing | rag_simple | 20 | 20 | 0 | 0 | 0 | 0.00 | 0.2500 | 0.1738 | 0.2500 | 0.4500 | 0.2760 | 0.3292 | 0.5500 | 0.3502 | 0.3419 |
| housing | snap_hyre | 20 | 20 | 0 | 0 | 0 | 1.00 | 0.2500 | 0.1367 | 0.2500 | 0.4000 | 0.2210 | 0.3017 | 0.4000 | 0.2281 | 0.3017 |
| housing | snap_hyre_exemplar_parallel3 | 20 | 20 | 0 | 0 | 0 | 3.00 | 0.3000 | 0.1521 | 0.3000 | 0.3500 | 0.2264 | 0.3167 | 0.3500 | 0.2436 | 0.3167 |
