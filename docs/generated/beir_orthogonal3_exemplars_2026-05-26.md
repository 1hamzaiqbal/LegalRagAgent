# BEIR Orthogonal Exemplar Selection

| Dataset | Exemplar source | Chroma docs | Gold ids excluded | Candidates | Eval rows excluded | Exemplar ids | Mutual cosine range |
|---|---|---:|---:|---:|---:|---|---:|
| beir_scifact | chroma | 5183 | 283 | 4900 | 0 | `4444861, 3052213, 581832` | 0.458..0.561 |
| beir_nfcorpus | chroma | 3633 | 3128 | 505 | 0 | `MED-1034, MED-2235, MED-5007` | 0.544..0.681 |
| beir_fiqa | chroma | 57638 | 1706 | 55932 | 0 | `51311, 178061, 583912` | 0.643..0.755 |
| beir_trec_covid | chroma | 171332 | 35480 | 135852 | 0 | `k596omcy, 2xsjxjml, okqsvg8q` | 0.568..0.638 |
| beir_scidocs | chroma_with_eval_row_exclusion | 25657 | 25657 | 25657 | 11 | `9e463eefadbcd336c69270a299666e4104d50159, 4017f984d1b4b8748a06da2739183782bbe9b46d, 1a090df137014acab572aa5dc23449b270db64b4` | 0.593..0.706 |
