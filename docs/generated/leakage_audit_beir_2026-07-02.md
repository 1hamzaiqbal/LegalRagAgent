# Yoon-style leakage audit on BEIR (Gemma 4 26B) - 2026-07-02

Question: outside legal, is expansion help leakage-concentrated?
Strata: matched vs unmatched (max sentence entailment vs gold, tau);
reported per dataset x method: Hit@5 deltas and help/hurt counts.


## tau = 0.8

| Dataset | Method | N | matched | raw Hit@5 | exp Hit@5 (matched) | delta | exp Hit@5 (unmatched) | delta | help_m/help_u | McNemar-unmatched p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| beir_nfcorpus | rag_hyde | 323 | 0 (0%) | 0.693 | nan | +nanpp | 0.334 | -35.9pp | 0/6 | b/c=6/122 p=3.3e-29 |
| beir_nfcorpus | snap_hyre | 323 | 3 (1%) | 0.693 | 1.000 | +0.0pp | 0.647 | -4.4pp | 0/20 | b/c=20/34 p=7.6e-02 |
| beir_scidocs | rag_hyde | 1000 | 0 (0%) | 0.490 | nan | +nanpp | 0.255 | -23.5pp | 0/58 | b/c=58/293 p=6.7e-39 |
| beir_scidocs | snap_hyre | 1000 | 10 (1%) | 0.490 | 0.900 | -10.0pp | 0.467 | -1.8pp | 0/87 | b/c=87/105 p=2.2e-01 |
| beir_scifact | rag_hyde | 300 | 1 (0%) | 0.820 | 1.000 | +0.0pp | 0.348 | -47.2pp | 0/12 | b/c=12/153 p=2.6e-32 |
| beir_scifact | snap_hyre | 300 | 21 (7%) | 0.820 | 0.857 | -14.3pp | 0.642 | -16.5pp | 0/12 | b/c=12/58 p=2.2e-08 |

## tau = 0.9

| Dataset | Method | N | matched | raw Hit@5 | exp Hit@5 (matched) | delta | exp Hit@5 (unmatched) | delta | help_m/help_u | McNemar-unmatched p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| beir_nfcorpus | rag_hyde | 323 | 0 (0%) | 0.693 | nan | +nanpp | 0.334 | -35.9pp | 0/6 | b/c=6/122 p=3.3e-29 |
| beir_nfcorpus | snap_hyre | 323 | 3 (1%) | 0.693 | 1.000 | +0.0pp | 0.647 | -4.4pp | 0/20 | b/c=20/34 p=7.6e-02 |
| beir_scidocs | rag_hyde | 1000 | 0 (0%) | 0.490 | nan | +nanpp | 0.255 | -23.5pp | 0/58 | b/c=58/293 p=6.7e-39 |
| beir_scidocs | snap_hyre | 1000 | 6 (1%) | 0.490 | 0.833 | -16.7pp | 0.469 | -1.8pp | 0/87 | b/c=87/105 p=2.2e-01 |
| beir_scifact | rag_hyde | 300 | 1 (0%) | 0.820 | 1.000 | +0.0pp | 0.348 | -47.2pp | 0/12 | b/c=12/153 p=2.6e-32 |
| beir_scifact | snap_hyre | 300 | 19 (6%) | 0.820 | 0.947 | -5.3pp | 0.637 | -17.1pp | 0/12 | b/c=12/60 p=8.1e-09 |