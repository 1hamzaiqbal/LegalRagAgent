# Diagnostic Controller Route Plan

Source diagnostics: `docs/legal_rag_diagnostic_table_2026-05-10.json`

| Dataset | Bottleneck | Route | Best acc | Calls | Secondary flags |
|---|---|---|---:|---:|---|
| barexam | `query_retrieval_gap` | `adaptive_snap_hyre_v2` | 86.0% | 2.00 | answer_conversion_gap |
| casehold | `answer_conversion_gap` | `adaptive_snap_hyre_diverse` | 73.5% | 2.00 | answer_conversion_gap, reject_or_escalate |
| housing | `statutory_entailment_gap` | `adaptive_snap_hyre_housing_verifier` | 74.5% | 1.00 | answer_conversion_gap |
| legalbench_scalr | `method_disagreement_gap` | `adaptive_snap_hyre_disagreement_replay` | 77.5% | 0.19 | - |

## Rationale

### barexam
- Snap/HyRE route increases gold exposure over plain retrieval.
- Best route still has 1 gold-retrieved-but-wrong rows.

### casehold
- Gold retrieval improves without meaningful accuracy lift, indicating answer-option conversion rather than retrieval alone.
- Best route still has 3 gold-retrieved-but-wrong rows.

### housing
- Housing-style yes/no statutory QA benefits from conservative entailment verification.
- Verifier accuracy 74.5% vs base 60.5%.
- Best route still has 25 gold-retrieved-but-wrong rows.

### legalbench_scalr
- Cached disagreement arbitration matches or beats the strongest source route at low marginal call cost.
