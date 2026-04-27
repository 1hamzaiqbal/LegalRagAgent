# Post-fixes log audit — 2026-04-26 (round 2)

Subagent audit covering the recent MuSiQue + planning_table runs and all 14
cluster post-fix N=1195 detail logs. Verdict: **all cluster landings clean;
new modes structurally sound; ONE major synthesizer-prompt issue found and
fixed in this commit.**

## Verdict

- 16 cluster N=1195 detail logs match `validation_log_2026-04-25.md` to ≤0.005 — no regressions, no NaN/null accuracies, no silent crashes
- planning_table family CLEAN at the schema level (TODOs fact-focused, dedup works, no exact duplicates)
- ⚠️ **17/30 planning_table records exhibit a contradiction failure**: findings say "passages do not contain X" but the final answer ignores them and asserts a confident parametric guess. This was the dominant composition failure mode

## Cluster landings sanity (16 logs)

- 26B llm_only @ 79.75%: 953/1195 verified; failures concentrated in EVIDENCE/CRIM. LAW/CONTRACTS, CONST. LAW strongest at 9.5% fail
- 26B rag_snap_hyde @ 81.17%: 970/1195 verified; full schema 1195/1195
- E4B subagent_hyde @ 60.17%: 719/1195 verified; same subject pattern as 26B with ~20pp uniform fail-rate gap (model-size effect, not subject regression)
- All 16: 0 errors, 0 null final_answer, 0 empty pred, 0 is_correct=None

## Composition failure analysis (planning_table)

- 23/30 records have `gold_retrieved=True` AND `em=False` — 77% composition-failure rate when retrieval was correct
- Same 30 questions, 25/30 (83%) gold_retrieved for both rag_simple and ptable_no_snap, yet rag_simple gets 8/30 EM vs ptable_no_snap 4/30 EM
- 5 wins for rag_simple where ptable_no_snap fails despite identical retrieval; only 1 win in the other direction
- **Smoking gun**: 17/30 records show contradiction — findings say "passages do not contain X" but the final answer asserts a parametric guess anyway:
  - "Derech Mitzvosecha" Q: finding says "no info on whom he followed", final answer asserts "Rabbi Schneur Zalman of Liadi" (wrong) instead of gold "Dovber Schneuri"
  - NBA scoring title: BOTH findings say "no info", final answer asserts "Dallas Mavericks" anyway
- Diagnosis: synthesizer prompt is too weak about trusting the findings vs parametric memory

## Fix applied this commit

Tightened the planning_table final synthesizer prompt with explicit instructions:
1. Treat the planning table findings as PRIMARY evidence; do NOT contradict a finding
2. If a finding says "passages did not contain", do NOT assert that fact from parametric memory
3. Walk through multi-finding chains explicitly before concluding
4. If findings insufficient, say so and pick best-supported option (don't guess)

Applies to both `run_planning_table` and `run_planning_table_no_snap`.

## Anomalies flagged

- planning_table_no_snap mode does NOT have `final_prompt_preview` populated — would help future audits verify the table reaches the final prompt
- Cluster runs aren't in `experiments.jsonl` (existing pattern, not a regression)
- One file naming confusion: doc claimed planning_table N=29 partial but actual file is N=5 (different run; verify before citing)

## Next steps

1. Re-run planning_table N=30 on MuSiQue with the tightened synthesizer prompt — does the EM lift toward rag_simple's 26.7%?
2. If yes → the failure was synthesizer prompt, not decomposition cost
3. If no → the decomposition cost story still holds
