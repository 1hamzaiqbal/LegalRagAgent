# Active-surface archive manifest — 2026-07-17

## Outcome

The active `codex/three_dial` and `codex/opd_distillation` surfaces no longer
carry the predecessor course project, pre-pivot class report, SCOPE paper tree,
or stale root research/navigation files. The historical `codex/scope_old`
branch was not changed.

## Verified archives

Archive directory: `/Users/hamzaiqbal/grad/LegalRagAgent_archive/`

| ZIP | SHA-256 | Source files | Contents |
|---|---|---:|---|
| `01_early-agentic-course-project_2026-02_to_2026-03.zip` | `d42841cb11461fccca0bbcd96762c58227e5385428b0e86b297bc550e2706007` | 11 | Proposal, private notes, graph, Gemma/DeepSeek screenshots, midterm source/bundle, slides |
| `02_scope-class-report-and-pre-pivot-surface_2026-04_to_2026-06.zip` | `fe9c031998a75e14e0b782606a7af12809450148c36d9f622f426ee57088678d` | 1,023 | Class report, paper tree, upload ZIPs, root diaries/ideas, old local literature, stale status/navigation/editor files |
| `LegalRagAgent-scope-old-20260717.zip` | `4690f746b259398f7408cb421aaab35774761807bd319b8ee779542af05a4073` | full worktree | Git-backed SCOPE historical branch snapshot |

For the first two ZIPs, `unzip -t` passed and a fresh extraction was hashed
file by file; the result exactly matched the adjacent `*.contents.sha256`
manifest before source removal. `.DS_Store` and `__pycache__` were excluded as
disposable machine artifacts.

## Removed from active branches

### Early agentic course project

- `agentic_ideas/`
- `initial_project.tex`
- `agentic_langgraph_graph.png`
- `gemma_agentic_eval.png`
- `deepseek_agentic_eval.png`
- `neurips_2024.sty`
- `midterm_report.tex`
- `mid-term-progress-agentic-rag.zip`
- `current_slides.pdf`
- `Agentic agent.pdf`

### Pre-pivot SCOPE/class-report surface

- `paper/`
- `reports/`
- `analysis/`
- `literature/`
- `ideas/` and `ideas.md`
- `RESEARCH.md` and `EXPERIMENTS.md`
- `final_class_report_overleaf_2026-05-01.zip`
- `scope_edit_v0.zip`
- root `22_query_contrast_diagram.png`
- generated `current_status.md`
- stale paper-only `.vscode/` tasks

The prior versions of `README.md`, `CLAUDE.md`, `AGENTS.md`, and
`docs/README.md` are included in the second ZIP; the active copies were replaced
with concise July track navigation.

### Disposable build/cache residue

Root and nested `.DS_Store`, `.pytest_cache`, and `__pycache__` directories were
removed without archival. `legal_rag/` and `playtests/` contained only compiled
`.pyc` residue—no source files—and were removed with that residue.

## Intentionally retained

- `main.py`, `rag_utils.py`, `llm_config.py`, and `web_scraper.py`: still
  imported by the evaluation harness.
- `skills/`: runtime prompts used by the harness and agent demo.
- `eval/`, `scripts/`, `tests/`, `utils/`, and `configs/`: reusable experiment
  and OPD code.
- `datasets/`, `chroma_db/`, `caches/`, `logs/`, and `references/`: active
  local substrate for reconstructing the paired table and running next-step
  experiments.
- `docs/`, `evidence/`, and `wiki/`: citation gates, reconciled July evidence,
  literature synthesis, and current track state.
- `.env`, `.venv`, and `uv.lock`: local environment state. Secrets remain
  ignored and were neither archived nor printed.

## Literature migration

Before removing the old local `literature/` duplicate, its three PDFs were
copied to the persistent EIT vault and verified:

| Source | EIT file | SHA-256 |
|---|---|---|
| LRAGE | `papers/arxiv_2504.01840v1.pdf` | `c7ce2b8871ce4486ea70170d630e74a4e178ede8e16ff2f92c294fe46882a370` |
| Zheng et al. legal retrieval benchmark | `papers/arxiv_2505.03970.pdf` | `6f2663e2a1df3755cfad0dcf2fc127ce77f6d762a74a6bbc52db6ac5563e393b` |
| L-MARS | `papers/arxiv_2509.00761v3.pdf` | `705a8222fe8192755926215c08b6a414c0e666b125023325c40e65da7c753b3a` |

Tracked vault manifests now cover 27 PDFs, one primary web snapshot, and 11
pinned repositories. LRAGE and L-MARS also have new wiki source pages and
repository checkouts; both pages remain marked triaged until their full
code/results reproduction passes are completed.

## Restore boundary

Restore archives only into a new empty directory. For historical development,
prefer the untouched `codex/scope_old` worktree. A broader pre-cleanup recovery
package also remains at
`/Users/hamzaiqbal/grad/LegalRagAgent_recovery_20260717`.
