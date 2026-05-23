# Overleaf Fast-Iteration Workflow

This repo can act as the source-of-truth machine while Overleaf remains the
editing surface. The goal is quick replacement packs: upload/overwrite files in
Overleaf, avoid manual `main.tex` edits when possible, and keep data/claim
sources traceable.

## Current Paths

- Repo on this machine: `/home/techguy227/grad/LegalRagAgent`
- Current Overleaf source snapshot: `paper/scope_edit_v0.zip`
- Current overwrite-only replacement pack:
  `paper/scope_edit_overleaf_replacements.zip`
- Full handoff bundle:
  `paper/paper_fast_edit_complete_handoff.zip`
- Corrected final/audit source of truth:
  `paper/after_report/`

An agent SSHing into this machine should start at:

```bash
cd /home/techguy227/grad/LegalRagAgent
sed -n '1,220p' paper/PAPER_FAST_EDIT_HANDOFF.md
sed -n '1,220p' paper/OVERLEAF_FAST_ITERATION_WORKFLOW.md
```

## Data Source Pathway

For quick claim edits, use these in order:

1. `paper/after_report/number_lineage.md`
   Exact paper numbers mapped to JSONL/cache source files.
2. `paper/after_report/repro_bundle/answer_log_summaries.csv`
   Compact answer accuracy, token, latency, and health summaries from raw logs.
3. `paper/after_report/repro_bundle/retrieval_cache_summaries.csv`
   Compact Hit@k/MRR@k summaries from retrieval caches.
4. `paper/after_report/repro_bundle/source_file_manifest.csv`
   Raw source paths, sizes, line counts, and SHA-256 hashes.
5. `paper/after_report/tables/`
   Final paper-facing `.tex` tables and CSVs.

The raw JSONL logs/caches are intentionally not duplicated under `paper/`
because they total about 2.35 GiB. If a future agent needs to verify a raw
source, it should use the manifest path and SHA:

```bash
sha256sum <path-from-source_file_manifest.csv>
wc -l <path-from-source_file_manifest.csv>
```

## Plot Regeneration

Figure 3 is generated from summarized CSV data, not by rerunning experiments:

```bash
python3 paper/after_report/scripts/regenerate_figure3_from_final_csv.py
```

Input:

- `paper/after_report/tables/topk_retrieval_metrics.csv`

Output:

- `paper/after_report/plots/figure3_topk_retrieval_curves_regenerated.png`

To refresh compact summaries from the local raw logs/caches:

```bash
python3 paper/after_report/scripts/build_repro_bundle.py
```

## Bringing In a New Overleaf Source Zip

When a newer Overleaf source zip is downloaded:

1. Save it under `paper/`, preferably with a versioned name:

```bash
cp ~/Downloads/<overleaf>.zip paper/scope_edit_v1.zip
```

2. Unpack it into a throwaway audit directory:

```bash
rm -rf paper/_scope_edit_v1_unzipped
mkdir -p paper/_scope_edit_v1_unzipped
unzip -q paper/scope_edit_v1.zip -d paper/_scope_edit_v1_unzipped
```

3. Find the source root:

```bash
find paper/_scope_edit_v1_unzipped -maxdepth 2 -name main.tex -print
```

4. Compile before editing:

```bash
cd paper/_scope_edit_v1_unzipped/final_icml_submission
tectonic main.tex
```

5. Scan for known stale claim strings:

```bash
rg -n "parity|matches or exceeds|every benchmark|31 signed|2001|268\\.3|ZZZ|Legal-Link-EU|without answer loss|without costing" \
  paper/_scope_edit_v1_unzipped/final_icml_submission
```

## Making Drop-In Replacement Packs

Prefer overwrite-only packs that preserve existing Overleaf paths. This avoids
manual edits in Overleaf.

Good pattern:

- Replace `sections/2Preliminary.tex` instead of adding a new `main.tex` input.
- If a new section must appear before conclusion, prepend it to
  `sections/6Conclusion.tex` and leave `main.tex` unchanged.
- Put replacement files under a folder that mirrors Overleaf paths, e.g.
  `paper/scope_edit_overleaf_replacements/sections/6Conclusion.tex`.
- Zip from inside that replacement folder:

```bash
cd paper/scope_edit_overleaf_replacements
zip -qr ../scope_edit_overleaf_replacements.zip sections README_UPLOAD.md
```

Test the replacement pack against the source zip before using it:

```bash
rm -rf /tmp/scope_upload_check
cp -a paper/_scope_edit_v0_unzipped/final_icml_submission /tmp/scope_upload_check
cp paper/scope_edit_overleaf_replacements/sections/*.tex /tmp/scope_upload_check/sections/
cd /tmp/scope_upload_check
tectonic main.tex
```

## Damage-Control Sequence

Use this order for fast paper repair:

1. Structure/page budget: background and related work drop-ins.
2. Main claim text: abstract, introduction, results, conclusion.
3. Tables/figures/captions: Table 4, Table 13 bolding, Figure 3 caption.
4. Bibliography placeholders.
5. Final compile and page check.

For the concrete claim queue, use:

- `paper/scope_edit_dropins/DAMAGE_CONTROL_EDIT_QUEUE.md`
- `paper/scope_edit_dropins/DEFERRED_DATA_CLAIM_FIXES.md`
- `paper/after_report/damage_report.md`

## What Not To Do

- Do not rerun experiments for a prose/table repair unless a source file is
  missing or contradicted.
- Do not copy the 2.35 GiB raw logs into `paper/`; use manifests and compact
  summaries.
- Do not rerun the older `paper/snap_hyre_2025_05_18/build_current_paper_assets.py`
  as if it exactly regenerates the current final package. It is useful
  provenance, but it is not the final-package generator.
- Do not silently import Legal-Link-EU or MASLegalBench narrative into the
  two-benchmark main story unless it is intentionally framed as boundary
  evidence.
