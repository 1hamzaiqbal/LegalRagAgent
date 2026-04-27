#!/usr/bin/env python3
"""
Lightweight detail-log viewer for LegalRagAgent.

Run:
    python scripts/log_viewer.py
    # then open http://localhost:8765

Features:
- Drag-drop a .jsonl file OR enter a path (relative to repo root or absolute)
- Browse records prev/next, jump to record number, jump to record_id substring
- Pretty-prints structured fields (snap_text, hyde_passages, retrieved_ids, etc.)
- Color-codes PASS/FAIL, highlights gold vs predicted
- Long fields are collapsible
- No third-party deps — pure stdlib

Stop with Ctrl-C.
"""
import http.server
import socketserver
import json
import os
import urllib.parse
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("LOG_VIEWER_PORT", 8765))

# Field rendering hints
LONG_FIELDS = {
    "snap_answer", "snap_text", "final_answer", "final_prompt_preview",
    "hyde_passages", "hyde_passages_raw", "hyde_chain", "iter_findings",
    "self_review_answer", "foe_review_answer", "control_review_answer",
    "evidence_store", "call_trace", "gap_results", "gaps",
    "retrieval_queries", "retrieved_ids", "aliases_used", "rerank_query",
    "table_entries", "snap_only_in_final_text",
}
KEY_FIELDS_ORDER = [
    "idx", "label", "subject", "mode", "provider", "is_correct",
    "question", "formatted_question", "correct_answer", "predicted_answer",
    "gold_idx", "choices",
    "snap_letter", "snap_answer",
    "final_answer", "final_prompt_preview",
    "hyde_passages", "hyde_chain", "iter_findings",
    "retrieved_ids", "gold_retrieved", "evidence_store",
    "elapsed_sec", "llm_calls", "input_tokens", "output_tokens",
    "rounds_completed", "early_exit", "routed_to",
    "f1", "em",
    "self_review_answer", "foe_review_answer", "control_review_answer",
    "gaps", "gap_results", "aliases_used",
    "rerank_query", "retrieval_queries",
    "table_entries", "intermediate_question",
    "error",
]

# In-memory cache of loaded files
LOADED = {}


def load_jsonl(path: str):
    """Load a .jsonl file into a list of dicts. Path can be absolute or relative to repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / path
    p = p.resolve()
    # Safety: only allow reading inside repo
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        # Allow /tmp/ paths and absolute paths under user's home for dragged files
        if not str(p).startswith(("/tmp/", str(Path.home()))):
            raise PermissionError(f"Path {p} is outside the repo.")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    rows = []
    with p.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                rows.append({"_parse_error": ln[:200]})
    return str(p), rows


def render_value(v, depth=0):
    """Render a single value to HTML with collapsibility for long content."""
    if v is None:
        return '<span class="null">null</span>'
    if isinstance(v, bool):
        return f'<span class="bool">{str(v).lower()}</span>'
    if isinstance(v, (int, float)):
        return f'<span class="num">{v}</span>'
    if isinstance(v, str):
        s = html.escape(v)
        if len(v) > 300:
            return f'<details><summary class="long-summary">[{len(v)} chars] click to expand</summary><pre class="str-long">{s}</pre></details>'
        if "\n" in v:
            return f'<pre class="str">{s}</pre>'
        return f'<span class="str-short">{s}</span>'
    if isinstance(v, list):
        if not v:
            return '<span class="null">[]</span>'
        if len(v) > 5:
            inner = "".join(f'<li>[{i}] {render_value(item, depth+1)}</li>' for i, item in enumerate(v[:5]))
            return f'<details><summary>list of {len(v)} items (showing first 5)</summary><ul>{inner}</ul></details>'
        return "<ul>" + "".join(f'<li>[{i}] {render_value(item, depth+1)}</li>' for i, item in enumerate(v)) + "</ul>"
    if isinstance(v, dict):
        if not v:
            return '<span class="null">{}</span>'
        inner = "".join(f'<dt>{html.escape(k)}</dt><dd>{render_value(val, depth+1)}</dd>' for k, val in v.items())
        if depth == 0 or len(v) <= 4:
            return f'<dl class="dict">{inner}</dl>'
        return f'<details><summary>dict ({len(v)} keys)</summary><dl class="dict">{inner}</dl></details>'
    return html.escape(repr(v))


def render_record(rec, idx_in_file, total):
    is_correct = rec.get("is_correct")
    badge = ""
    if is_correct is True:
        badge = '<span class="badge pass">PASS</span>'
    elif is_correct is False:
        badge = '<span class="badge fail">FAIL</span>'

    rec_id = rec.get("idx") or rec.get("record_id") or f"#{idx_in_file}"
    pred = rec.get("predicted_answer") or rec.get("final_answer") or ""
    gold = rec.get("correct_answer") or rec.get("gold_answer") or ""

    # Reorder fields
    keys_in_record = list(rec.keys())
    ordered = [k for k in KEY_FIELDS_ORDER if k in rec] + [k for k in keys_in_record if k not in KEY_FIELDS_ORDER]

    rows_html = []
    for k in ordered:
        v = rec[k]
        is_long = k in LONG_FIELDS
        rendered = render_value(v)
        cls = "field long" if is_long else "field"
        rows_html.append(f'<div class="{cls}"><div class="key">{html.escape(k)}</div><div class="val">{rendered}</div></div>')

    return f"""
    <div class="record">
        <div class="rec-header">
            <span class="rec-pos">[{idx_in_file+1} / {total}]</span>
            <span class="rec-id">{html.escape(str(rec_id))}</span>
            {badge}
            <span class="rec-pred">pred: <strong>{html.escape(str(pred)[:200])}</strong></span>
            <span class="rec-gold">gold: <strong>{html.escape(str(gold)[:200])}</strong></span>
        </div>
        <div class="rec-fields">
            {"".join(rows_html)}
        </div>
    </div>
    """


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #fafafa; color: #222; }
.controls { background: #2c3e50; color: #ecf0f1; padding: 12px 20px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
.controls input[type=text] { padding: 6px 10px; font-size: 14px; border: 1px solid #555; border-radius: 4px; width: 600px; max-width: 60%; }
.controls input[type=number] { padding: 6px; font-size: 14px; width: 80px; border: 1px solid #555; border-radius: 4px; }
.controls button { padding: 6px 14px; font-size: 14px; background: #3498db; color: white; border: 0; border-radius: 4px; cursor: pointer; margin-left: 4px; }
.controls button:hover { background: #2980b9; }
.controls button:disabled { background: #777; cursor: not-allowed; }
.controls .meta { font-size: 12px; color: #bdc3c7; margin-top: 6px; }
.drop-zone { border: 2px dashed #3498db; border-radius: 6px; padding: 20px; margin: 12px 0; text-align: center; background: rgba(255,255,255,0.05); cursor: pointer; }
.drop-zone.dragover { background: rgba(52,152,219,0.3); }
.record { background: white; margin: 12px 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.rec-header { background: #34495e; color: white; padding: 10px 16px; border-radius: 6px 6px 0 0; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; font-size: 13px; }
.rec-pos { font-weight: bold; color: #ecf0f1; }
.rec-id { font-family: monospace; color: #bdc3c7; }
.rec-pred, .rec-gold { font-size: 12px; }
.rec-pred strong { color: #f39c12; }
.rec-gold strong { color: #2ecc71; }
.badge { padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; }
.badge.pass { background: #27ae60; color: white; }
.badge.fail { background: #e74c3c; color: white; }
.rec-fields { padding: 8px 16px; }
.field { display: flex; gap: 12px; padding: 4px 0; border-bottom: 1px solid #eee; }
.field:last-child { border-bottom: 0; }
.field .key { min-width: 180px; max-width: 180px; font-family: monospace; font-size: 12px; color: #555; flex-shrink: 0; }
.field .val { flex: 1; font-size: 13px; word-break: break-word; }
.field.long .val { background: #f4f4f4; padding: 6px 10px; border-radius: 3px; }
.str { white-space: pre-wrap; margin: 0; font-family: 'Menlo', monospace; font-size: 12px; max-height: 400px; overflow-y: auto; }
.str-long { white-space: pre-wrap; margin: 0; font-family: 'Menlo', monospace; font-size: 12px; max-height: 600px; overflow-y: auto; background: white; padding: 8px; border: 1px solid #ddd; }
.str-short { font-family: monospace; font-size: 12px; }
.long-summary { cursor: pointer; color: #2980b9; font-size: 12px; }
ul, dl { margin: 4px 0; padding-left: 20px; }
ul li { font-size: 12px; }
dl dt { font-family: monospace; font-size: 11px; color: #555; margin-top: 4px; }
dl dd { margin-left: 16px; font-size: 12px; }
.num { color: #16a085; font-family: monospace; }
.bool { color: #c0392b; font-family: monospace; }
.null { color: #95a5a6; font-style: italic; font-family: monospace; }
.dict { background: #fafafa; border-left: 2px solid #bdc3c7; padding-left: 8px; }
details summary { cursor: pointer; color: #2980b9; font-size: 12px; }
.empty { padding: 40px; text-align: center; color: #777; }
.summary-bar { background: #ecf0f1; padding: 10px 20px; font-size: 13px; color: #555; border-bottom: 1px solid #ddd; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.summary-bar strong { color: #222; }
.filter-link { padding: 3px 10px; border-radius: 3px; text-decoration: none; color: #2980b9; border: 1px solid #bdc3c7; background: white; font-size: 12px; }
.filter-link:hover { background: #d6eaf8; }
.filter-link.filter-active { background: #2980b9; color: white; border-color: #2980b9; }
.filter-link.filter-active .filter-count { color: #ecf0f1; }
.filter-count { color: #95a5a6; font-size: 11px; }
.error { background: #e74c3c; color: white; padding: 10px 16px; margin: 12px; border-radius: 4px; }
"""

JS = """
// Drag-drop
const dz = document.getElementById('drop-zone');
if (dz) {
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('dragover');
        const f = e.dataTransfer.files[0];
        if (!f) return;
        // Get file path if available (only works in Electron-style; in browser we just send name)
        const fd = new FormData();
        fd.append('file', f);
        fetch('/upload', {method: 'POST', body: fd}).then(r => r.text()).then(p => {
            window.location = '/view?path=' + encodeURIComponent(p) + '&i=0';
        });
    });
}
// Keyboard nav
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft' || e.key === 'p') {
        const prev = document.getElementById('btn-prev');
        if (prev && !prev.disabled) prev.click();
    } else if (e.key === 'ArrowRight' || e.key === 'n') {
        const next = document.getElementById('btn-next');
        if (next && !next.disabled) next.click();
    }
});
"""


def page(body, title="Log Viewer"):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head>
<body>{body}<script>{JS}</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(url.query)
        if url.path == "/":
            self._home()
        elif url.path == "/view":
            self._view(params)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/upload":
            self.send_error(404)
            return
        clen = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self.send_error(400)
            return
        # Parse boundary and read body
        boundary = ctype.split("boundary=")[1].encode()
        body = self.rfile.read(clen)
        # Find file content section (very lightweight parser)
        parts = body.split(b"--" + boundary)
        for part in parts:
            if b"Content-Disposition: form-data; name=\"file\"" not in part:
                continue
            # filename=
            fname = b"upload.jsonl"
            for line in part.split(b"\r\n")[:5]:
                if b"filename=\"" in line:
                    fname = line.split(b"filename=\"")[1].split(b"\"")[0]
            # body after \r\n\r\n
            content = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n--")
            tmp_path = Path("/tmp") / f"log_viewer_{fname.decode(errors='replace')}"
            tmp_path.write_bytes(content)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(str(tmp_path).encode())
            return
        self.send_error(400)

    def _home(self):
        # List recent local logs
        recent = sorted(
            (REPO_ROOT / "logs").glob("eval_*detail.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:30]
        recent_html = "".join(f'<li><a href="/view?path={urllib.parse.quote(str(p.relative_to(REPO_ROOT)))}&i=0">{p.name}</a> <span style="color:#888">({p.stat().st_size//1024} KB)</span></li>' for p in recent)
        body = f"""
        <div class="controls">
            <h2 style="margin:0">📋 Detail Log Viewer</h2>
            <div class="meta">stdlib-only Flask-free viewer · drag-drop a .jsonl OR enter a path</div>
            <form action="/view" method="GET" style="margin-top:8px">
                <input type="text" name="path" placeholder="logs/eval_..._detail.jsonl  (relative to repo root)" required>
                <input type="hidden" name="i" value="0">
                <button type="submit">Open</button>
            </form>
            <div id="drop-zone" class="drop-zone">📂 Drag &amp; drop a .jsonl file here</div>
        </div>
        <div style="padding: 20px">
            <h3>Recent local logs</h3>
            <ul>{recent_html}</ul>
            <h3>Tips</h3>
            <ul>
                <li>Use ← → arrow keys (or p / n) to navigate questions</li>
                <li>Long fields collapse — click "expand" to read</li>
                <li>Color codes: PASS = green, FAIL = red, pred = orange, gold = green</li>
            </ul>
        </div>
        """
        self._send_html(page(body))

    def _view(self, params):
        path = params.get("path", [""])[0]
        i = int(params.get("i", ["0"])[0])
        find_id = params.get("find", [""])[0]
        flt = params.get("filter", ["all"])[0]
        if not path:
            return self._home()
        try:
            resolved, all_rows = LOADED.get(path) or load_jsonl(path)
            LOADED[path] = (resolved, all_rows)
        except Exception as e:
            self._send_html(page(f'<div class="error">Error loading {html.escape(path)}: {html.escape(str(e))}</div><p><a href="/">← back</a></p>'))
            return

        # Compute summary stats from ALL rows (not filtered)
        all_total = len(all_rows)
        pass_count = sum(1 for r in all_rows if r.get("is_correct") is True)
        fail_count = sum(1 for r in all_rows if r.get("is_correct") is False)
        empty_pred = sum(1 for r in all_rows if not (r.get("predicted_answer") or "").strip())

        # Apply filter — keep original index in the row tuple so user can see "[#827 of 1195]"
        def matches(row, f):
            if f == "all": return True
            if f == "pass": return row.get("is_correct") is True
            if f == "fail": return row.get("is_correct") is False
            if f == "empty": return not (row.get("predicted_answer") or "").strip()
            return True

        # filtered list of (orig_index, row)
        filtered = [(idx, r) for idx, r in enumerate(all_rows) if matches(r, flt)]
        f_total = len(filtered)

        if find_id and filtered:
            for j, (_, r) in enumerate(filtered):
                rid = str(r.get("idx") or r.get("record_id") or "")
                if find_id in rid:
                    i = j
                    break

        if f_total == 0:
            no_match = f'<div class="empty">No records match filter "{html.escape(flt)}". <a href="/view?path={urllib.parse.quote(path)}&i=0&filter=all">show all</a></div>'
            body = f"""
            <div class="controls">
                <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
                    <a href="/" style="color:#3498db; text-decoration:none">← home</a>
                    <span style="color:#bdc3c7; font-size:12px">{html.escape(resolved)}</span>
                </div>
            </div>
            {self._filter_bar(path, flt, all_total, pass_count, fail_count, empty_pred)}
            {no_match}
            """
            self._send_html(page(body, title=f"{Path(resolved).name} [filter:{flt}]"))
            return

        i = max(0, min(i, f_total - 1))
        orig_idx, rec = filtered[i]

        prev_link = f'/view?path={urllib.parse.quote(path)}&i={max(0,i-1)}&filter={flt}'
        next_link = f'/view?path={urllib.parse.quote(path)}&i={min(f_total-1,i+1)}&filter={flt}'

        body = f"""
        <div class="controls">
            <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
                <a href="/" style="color:#3498db; text-decoration:none">← home</a>
                <span style="color:#bdc3c7; font-size:12px">{html.escape(resolved)}</span>
                <a id="btn-prev" href="{prev_link}"><button {'disabled' if i==0 else ''}>← prev (←/p)</button></a>
                <span>record {i+1} of {f_total}{' (filtered)' if flt != 'all' else ''} · orig #{orig_idx+1} of {all_total}</span>
                <a id="btn-next" href="{next_link}"><button {'disabled' if i>=f_total-1 else ''}>next (→/n) →</button></a>
                <form action="/view" method="GET" style="display:inline">
                    <input type="hidden" name="path" value="{html.escape(path)}">
                    <input type="hidden" name="filter" value="{flt}">
                    jump to: <input type="number" name="i" value="{i+1}" min="1" max="{f_total}"> <button type="submit">go</button>
                </form>
                <form action="/view" method="GET" style="display:inline">
                    <input type="hidden" name="path" value="{html.escape(path)}">
                    <input type="hidden" name="filter" value="{flt}">
                    find id: <input type="text" name="find" placeholder="substring of record id"> <button type="submit">find</button>
                </form>
            </div>
        </div>
        {self._filter_bar(path, flt, all_total, pass_count, fail_count, empty_pred)}
        {render_record(rec, orig_idx, all_total)}
        """
        self._send_html(page(body, title=f"{Path(resolved).name} [{i+1}/{f_total} {flt}]"))

    def _filter_bar(self, path, current, all_total, pass_count, fail_count, empty_pred):
        def link(name, label, count):
            active = "filter-active" if current == name else ""
            return f'<a href="/view?path={urllib.parse.quote(path)}&i=0&filter={name}" class="filter-link {active}">{label} <span class="filter-count">({count})</span></a>'
        acc = (pass_count*100/all_total if all_total else 0)
        return f"""
        <div class="summary-bar">
            <strong>{all_total} records</strong> · accuracy <strong>{acc:.1f}%</strong> · filter:
            {link('all', 'All', all_total)}
            {link('pass', '✓ PASS', pass_count)}
            {link('fail', '✗ FAIL', fail_count)}
            {link('empty', 'empty pred', empty_pred)}
        </div>
        """

    def _send_html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())


if __name__ == "__main__":
    print(f"📋 Log viewer starting on http://localhost:{PORT}")
    print(f"   Repo root: {REPO_ROOT}")
    print(f"   Stop with Ctrl-C")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  bye")
