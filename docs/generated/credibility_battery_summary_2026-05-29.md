# Credibility Battery Summary - 2026-05-29

Aggregate of Phases A, B, and C. No `paper/` files were edited.

## Phase Reads

### credibility_B_partial_correlation_2026-05-29.md

- Mechanism circularity check: **survives**. Pooled gold-affinity-delta partial-R2 after CE(raw,gold) and BM25 controls is `0.096`.
- Kill criterion: below 0.05 means the gold-affinity-delta mechanism is mostly mechanical after controlling for raw closeness and BM25-space affinity.

### credibility_C_bm25_replication_2026-05-29.md

- BM25 mechanism verdict: **travels**. Mean per-dataset SCOPE Spearman between BM25 gold-affinity delta and BM25 retrieval gain is `0.342`.
- Kill criterion: <=0.2 means the mechanism is likely gte/CE-specific; >=0.3 means it travels to BM25.
- Scope caveat: full BM25 replication completed for SciFact, NFCorpus, and FiQA. BarExamQA completed the corpus statistics pass but full BM25 retrieval was deferred after the exact pure-Python scorer proved too slow; HousingQA was deferred after that blocker.

### credibility_A_independent_judge_2026-05-29.md

- Factuality falsification headline: **survives**. Original Gemma factuality AUC `0.581`, independent factuality AUC `0.586`, geometry AUC `0.816`, independent+geometry AUC `0.816`, marginal lift `0.000`.
- Inter-rater reliability: Spearman `0.671`, Pearson `0.691`, Cohen kappa `0.507` on score >= 0.75.
- Status: **provisional/rate-limited**. OpenRouter `openai/gpt-4o` completed 3,848 judge records before the monthly key limit stopped the run: all five BEIR sets are complete for the q200/q50 sample, BarExamQA is partial at 224 joined feature rows, and HousingQA has no independent-judge rows in this cache.

## Honest Claim

The honest paper claim is that the geometry/mechanism story survives the strongest completed circularity controls, and the mechanism is not obviously tied to the dense+CE stack because the BM25 BEIR replication stays above the pre-set correlation threshold. The independent-factuality result also supports the "geometry dominates factuality" falsification, but only as provisional until the rate-limited BarExam remainder and Housing rows are judged. The open limitations are full-corpus BM25 retrieval on the large legal collections and completion of the independent-judge cache once provider budget resets.
