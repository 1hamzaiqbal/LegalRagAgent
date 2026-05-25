# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legal_link_eu | rag_simple | 10 | 10 | 0 | 0 | 0 | 0.00 | 0.7000 | 0.1400 | 0.7000 | 0.9000 | 0.4400 | 0.7833 | 1.0000 | 0.6000 | 0.7933 |
| legal_link_eu | snap_hyre | 10 | 10 | 0 | 0 | 0 | 1.00 | 0.4000 | 0.0800 | 0.4000 | 0.4000 | 0.2400 | 0.4000 | 0.4000 | 0.2400 | 0.4000 |
| legal_link_eu | snap_hyre_exemplar_parallel3 | 10 | 10 | 0 | 0 | 0 | 3.00 | 0.5000 | 0.1000 | 0.5000 | 0.7000 | 0.3200 | 0.5700 | 0.7000 | 0.3400 | 0.5700 |
