# Log Viewer — quick reference

A lightweight local viewer for `logs/eval_*_detail.jsonl`. Stdlib-only (no Flask, no extra installs).

## Run it

```bash
python scripts/log_viewer.py
# Open http://localhost:8765
```

Stop with Ctrl-C.

## Use it

**Home page:**
- Drag-drop a `.jsonl` file into the dashed box
- Or type a path (relative to repo root) and click Open
- Recent local detail logs are listed for one-click access

**View page:**
- Header shows: record N of total, pred (orange), gold (green), PASS/FAIL badge
- Long fields (`snap_text`, `hyde_passages`, `final_answer`, `evidence_store` etc.) are collapsed by default — click to expand
- Summary bar shows: total / accuracy + clickable filter chips (All / ✓ PASS / ✗ FAIL / empty pred)

**Filtering by status:**
- Click `✓ PASS` (count) to view only correct records
- Click `✗ FAIL` (count) to view only incorrect records
- Click `empty pred` (count) to find parse failures / runaway loops
- Click `All` to clear the filter
- When filtered, position bar shows both: `record 5 of 71 (filtered) · orig #9 of 200`
- Navigation/jump/find-id all stay within the active filter

**Navigation:**
- ← / → arrow keys (or `p` / `n`) — prev/next record
- "jump to: <N>" — go to record number N
- "find id: <substring>" — first record whose `idx` contains the substring (e.g., `2hop__835710` finds `mq_2hop__835710_7298`)

## Tips for the meeting

- Drag a log onto the home page — no command-line gymnastics
- `idx`, `predicted_answer`, `correct_answer`, `is_correct` are surfaced at the top of every record
- Use the find-id box to jump to specific examples cited in `02_methods_explained.md`
- Cross-reference with `05_logs_index.md` to know which file to load for which result

## Custom port

```bash
LOG_VIEWER_PORT=9000 python scripts/log_viewer.py
```
