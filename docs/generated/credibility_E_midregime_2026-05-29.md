# Credibility E Mid-Regime Construction - 2026-05-29

No `paper/` files were edited. This report is read-only over existing retrieval caches.

## Verdict

- Constructed mid-regime points using the allowed lower-k evidence-budget axis: the same caches are evaluated at Hit@1 through Hit@5.
- In the strict 20-30% raw band, the raw+SCOPE pool improves raw on all available points: SciDocs Hit@1 and Housing state-filtered Hit@2/Hit@3.
- The threshold is not a clean raw-Hit-only boundary. Pooling starts to help raw in the low-20% regime, but it only carries the method when SCOPE contributes complementary correct evidence; it remains weak relative to SCOPE on BarExamQA and CaseHOLD.
- Honest claim: raw+SCOPE pooling is useful as a risk-control fusion in mid/high raw regimes, not as a universal replacement for canonical SCOPE on sparse legal corpora.

## Strict Mid-Regime Points

| Dataset | Slice | N | Raw | SCOPE | Raw+SCOPE pool | SCOPE-Raw | Pool-Raw | Pool-SCOPE | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SciDocs | Hit@1 | 989 | 22.2% | 21.1% | 23.2% | -1.1% | 0.9% | 2.0% | helps raw |
| HousingQA state-filtered | Hit@2 | 6832 | 23.9% | 25.5% | 25.9% | 1.7% | 2.0% | 0.4% | helps raw |
| HousingQA state-filtered | Hit@3 | 6832 | 29.3% | 31.0% | 32.3% | 1.7% | 3.0% | 1.3% | helps raw |

## Near-Mid Regime Context

| Dataset | Slice | N | Raw | SCOPE | Raw+SCOPE pool | SCOPE-Raw | Pool-Raw | Pool-SCOPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciDocs | Hit@1 | 989 | 22.2% | 21.1% | 23.2% | -1.1% | 0.9% | 2.0% |
| HousingQA state-filtered | Hit@2 | 6832 | 23.9% | 25.5% | 25.9% | 1.7% | 2.0% | 0.4% |
| HousingQA state-filtered | Hit@3 | 6832 | 29.3% | 31.0% | 32.3% | 1.7% | 3.0% | 1.3% |
| SciDocs | Hit@2 | 989 | 31.7% | 31.0% | 34.4% | -0.7% | 2.6% | 3.3% |
| HousingQA state-filtered | Hit@4 | 6832 | 33.5% | 34.8% | 37.0% | 1.3% | 3.5% | 2.2% |

## Full Hit@5 Anchors

| Dataset | N | Raw Hit@5 | SCOPE Hit@5 | Raw+SCOPE pool Hit@5 | SCOPE-Raw | Pool-Raw | Pool-SCOPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | 1192 | 1.4% | 12.0% | 3.9% | 10.6% | 2.5% | -8.1% |
| CaseHOLD | 3600 | 17.9% | 45.0% | 19.2% | 27.0% | 1.3% | -25.8% |
| HousingQA state-filtered | 6832 | 36.8% | 38.0% | 41.1% | 1.2% | 4.3% | 3.1% |
| SciDocs | 989 | 49.3% | 47.2% | 53.6% | -2.1% | 4.2% | 6.4% |
| FiQA | 648 | 66.2% | 35.2% | 71.9% | -31.0% | 5.7% | 36.7% |
| NFCorpus | 323 | 69.3% | 65.0% | 70.0% | -4.3% | 0.6% | 5.0% |
| SciFact | 300 | 82.0% | 65.7% | 83.3% | -16.3% | 1.3% | 17.7% |
| TREC-COVID | 50 | 98.0% | 96.0% | 100.0% | -2.0% | 2.0% | 4.0% |

## Interpretation

- CaseHOLD is the lower anchor: raw Hit@5 is 17.9%, SCOPE jumps to 45.0%, but the raw+SCOPE pool reaches only 19.2%. In this sparse regime, fusion mostly preserves raw rather than the SCOPE gains.
- SciDocs Hit@1 gives a strict mid-regime point: raw 22.2%, SCOPE 21.1%, pool 23.2%. The pool helps raw modestly, but this is a fusion gain rather than a SCOPE-alone gain.
- Housing state-filtered Hit@2 gives the strongest strict mid-regime point: raw 23.9%, SCOPE 25.5%, pool 25.9%. Pooling is helpful and slightly better than either component.
- The upper side of the void is consistent: SciDocs Hit@2 at 31.7% raw and Housing Hit@5 at 36.8% raw both show pool gains over raw.
- Therefore the practical threshold for positive raw+SCOPE pooling appears around the low-20% raw-retrieval regime, but the threshold for replacing canonical SCOPE is higher and corpus-dependent.

## Notes

- This phase does not run new retrieval. Lower-k evaluation reuses the deterministic ranking already present in each cache.
- The mid-regime construction is an evidence-budget proxy, not a new benchmark split. It is useful for regime-shape diagnosis, not for final leaderboard claims.
- BarExamQA: N=1192 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- CaseHOLD: N=3600 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- HousingQA state-filtered: N=6832 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- SciDocs: N=989 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- FiQA: N=648 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- NFCorpus: N=323 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- SciFact: N=300 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- TREC-COVID: N=50 from intersection of raw, SCOPE, and raw+SCOPE pool caches
- Row-level points: `docs/generated/credibility_E_midregime_2026-05-29_points.jsonl`

