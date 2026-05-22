# Paper Iteration Signal - 2026-05-20

Purpose: handoff note for eval/investigation agents working alongside the
paper draft. Keep these details out of manuscript prose unless they are later
converted into signed, paper-facing claims.

Observed around 2026-05-20 03:28 CDT.

## High-Priority Signals

1. HousingQA `groq-llama8b` `snap_hyre` is not citable yet.
   - `current_status.md` reports the row as partial/stale at 5335/6853 with
     one fallback-key issue in the original partial detail log.
   - A repair-tail tmux session was still present:
     `housing_8b_snap_hyre_repair_tail_20260520_0314`.
   - The repair-tail detail log had 939 rows at the local check:
     `logs/eval_snap_hyre_groq-llama8b_20260520_031420_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-repairtail-nfull-k5_detail.jsonl`.
   - Do not sign or cite this row until the original partial and repair tail
     are merged, audited, and added to `docs/signoff_log.md`.

2. `current_status.md` may understate live activity.
   - The dashboard said `0/78 active, 1/78 partial stale`.
   - Active tmux sessions were still visible for the HousingQA repair tail and
     the Gemma q500 exemplar retrieval probe.
   - Before using dashboard active/partial language, refresh status or inspect
     tmux/log tails directly.

3. `logs/experiments.jsonl` contains partial/tail rows that look like results.
   - Examples include merged rows whose latest JSONL summary is only a tail:
     HousingQA 70B `rag_hyde`, HousingQA 8B `rag_simple`, and Legal-Link-EU
     Gemma `snap_hyre`.
   - For paper claims, prefer `docs/signoff_log.md` and the paper-generated
     `current_audited_rows.csv` over raw JSONL tail entries.

4. The q500 exemplar probe is provisional.
   - A tmux session was still present:
     `exemplar_gemma_q500_retrieval_probe_resume1_20260520_0309`.
   - The q20 real-passage exemplar probe is useful as a pilot, but not a main
     matrix result. It improved over canonical Snap-HyRE on q20 retrieval slices
     but did not beat raw retrieval on Legal-Link-EU or MASLegalBench.
   - Do not promote `snap_hyre_exemplar` beyond probe language until larger
     retrieval and downstream answer checks pass the same inclusion standard as
     canonical `snap_hyre`.

5. Usage/cost assets should not support cost-efficiency claims yet.
   - The current usage table covers rows with answer-pass token fields only.
   - First-stage query-generation usage is not included in that table, so the
     conceptual call-cost comparison in the paper should remain separate from
     measured answer-pass token accounting.

## Paper-Side Guardrails

- Keep CaseHOLD and LegalBench-SCALR out of the current main benchmark matrix.
- Keep the paper claim narrow: Snap-HyRE helps relative to raw RAG on
  BarExamQA and some cells, but it is not a universal improvement over raw RAG
  or HyDE.
- Treat MASLegalBench retrieval as a same-source-document proxy, not
  passage-level gold-evidence retrieval.
