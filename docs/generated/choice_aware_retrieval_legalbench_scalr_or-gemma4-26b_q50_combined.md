# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legalbench_scalr | rag_simple | 50 | 50 | 0 | 0 | 0 | 0.00 | 0.2600 | 0.2600 | 0.2600 | 0.5600 | 0.5600 | 0.3713 | 0.6400 | 0.6400 | 0.3830 |
| legalbench_scalr | rag_hyde_blind | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.3800 | 0.3800 | 0.3800 | 0.6000 | 0.6000 | 0.4700 | 0.6600 | 0.6600 | 0.4776 |
| legalbench_scalr | rag_hyde_choice | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.6000 | 0.6000 | 0.6000 | 0.7400 | 0.7400 | 0.6557 | 0.7600 | 0.7600 | 0.6585 |
| legalbench_scalr | snap_hyre | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.6400 | 0.6400 | 0.6400 | 0.7600 | 0.7600 | 0.6857 | 0.8000 | 0.8000 | 0.6904 |
| legalbench_scalr | multi_hyde_diverse | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.6200 | 0.6200 | 0.6200 | 0.6600 | 0.6600 | 0.6340 | 0.7200 | 0.7200 | 0.6411 |
| legalbench_scalr | snap_choice_hyre | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.5600 | 0.5600 | 0.5600 | 0.6800 | 0.6800 | 0.6167 | 0.7400 | 0.7400 | 0.6250 |
