# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legal_link_eu | rag_simple | 50 | 50 | 0 | 0 | 0 | 0.00 | 0.7400 | 0.1490 | 0.7400 | 0.9200 | 0.4740 | 0.8233 | 0.9600 | 0.6420 | 0.8287 |
| legal_link_eu | snap_hyre | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.3600 | 0.0720 | 0.3600 | 0.5800 | 0.2940 | 0.4483 | 0.7400 | 0.4310 | 0.4679 |
| legal_link_eu | snap_hyre_exemplar_parallel3 | 50 | 50 | 0 | 0 | 0 | 3.00 | 0.4600 | 0.0930 | 0.4600 | 0.6800 | 0.3300 | 0.5497 | 0.8000 | 0.4470 | 0.5642 |
