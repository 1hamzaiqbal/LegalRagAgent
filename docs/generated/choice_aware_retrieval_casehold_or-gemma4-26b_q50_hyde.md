# Choice-Aware Retrieval Probe

| dataset | mode | rows | scored | errors | parse_fail | artifacts | calls | Hit@1 | Recall@1 | MRR@1 | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casehold | rag_hyde_blind | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.1600 | 0.1600 | 0.1600 | 0.2800 | 0.2800 | 0.2023 | 0.3400 | 0.3400 | 0.2110 |
| casehold | rag_hyde_choice | 50 | 50 | 0 | 0 | 0 | 1.00 | 0.4800 | 0.4800 | 0.4800 | 0.6600 | 0.6600 | 0.5550 | 0.7200 | 0.7200 | 0.5623 |
| casehold | rag_simple | 50 | 50 | 0 | 0 | 0 | 0.00 | 0.1000 | 0.1000 | 0.1000 | 0.2400 | 0.2400 | 0.1420 | 0.2800 | 0.2800 | 0.1478 |
