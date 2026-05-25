# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legal_link_eu | rag_simple | 100 | 100 | 0 | 0 | 0 | 0.00 | 0.7100 | 0.1425 | 0.7100 | 0.9100 | 0.4410 | 0.7995 | 0.9500 | 0.5750 | 0.8048 |
| legal_link_eu | snap_hyre | 100 | 100 | 0 | 0 | 0 | 1.00 | 0.4300 | 0.0865 | 0.4300 | 0.6200 | 0.2590 | 0.5003 | 0.7200 | 0.3515 | 0.5143 |
| legal_link_eu | snap_hyre_exemplar_parallel3 | 100 | 100 | 0 | 0 | 0 | 3.00 | 0.4100 | 0.0825 | 0.4100 | 0.6600 | 0.3030 | 0.5057 | 0.7800 | 0.3955 | 0.5213 |
