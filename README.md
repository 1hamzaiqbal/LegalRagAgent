# LegalRagAgent

Research code and evidence for studying when retrieval helps or harms a
particular reader, and how much search effort is worth paying for.

## Active tracks

- **Primary:** [`codex/three_dial`](wiki/tracks/three-dial.md) — paired,
  reader-conditioned marginal evidence-set utility under cost.
- **Gated engineering:** [`codex/opd_distillation`](wiki/tracks/opd-distillation.md)
  — task reward plus gap-gated on-policy distillation, contingent on a teacher
  skill-gap result.
- **Historical:** `codex/scope_old` — SCOPE/Snap-HyRE, reviews, old papers, and
  class-project provenance.

Start with the durable
[`research-state-2026-07-17`](wiki/snapshots/research-state-2026-07-17.md)
snapshot, then [`ACTIVE_TRACK.md`](ACTIVE_TRACK.md),
[`docs/OPERATIONS.md`](docs/OPERATIONS.md),
[`docs/README.md`](docs/README.md), and the
[`literature vault index`](wiki/literature/index.md).

## Research question

Given a reader, question, current evidence set, and remaining budget, can a
policy estimate whether another retrieval action will improve task success,
leave it unchanged, or cause harm—and stop, abstain, or arbitrate when the set
is sufficient or conflicting?

The three measured dials are:

1. evidence exposure and set quality;
2. reader ability to convert evidence into a correct answer;
3. retrieval/search effort, latency, and marginal cost.

## Active repository map

```text
ACTIVE_TRACK.md       branch entrypoint
wiki/                 Obsidian-linked research, result, and literature pages
evidence/july_2026/   compact reconciled evidence package
docs/                 citation gates, completion audits, and machine map
eval/                 controlled evaluation harness
scripts/              analysis, audit, bandit, OPD, and HPC tooling
tests/                focused regressions
main.py               shared LangGraph/runtime functions imported by the harness
rag_utils.py          retrieval, reranking, and vector-store helpers
llm_config.py         provider/model configuration
datasets/             local benchmark assets
caches/               reusable generation/retrieval/document caches
logs/                 source experiment logs and machine ledger
references/           local working copies; EIT is the persistent source vault
```

The old `paper/`, class report, proposal, slides, root research diaries, and
duplicate literature folder were removed from the active branches after
byte-verified archiving. The payloads live in persistent EIT storage and the
exact Git states have named archive branches; the Mac keeps only small restore
manifests. See `docs/archive_manifest_2026-07-17.md`.

## Setup

```bash
uv sync
cp .env.example .env
uv run python llm_config.py
```

Python 3.11–3.13 is supported. If `uv` is not on `PATH`, use
`~/.local/bin/uv`.

## Core checks

```bash
uv run python scripts/check_workspace.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -q
uv run python scripts/opd/test_opd_loss.py
git diff --check
```

Use focused tests for the code being changed; broad model/data runs require the
appropriate local or EIT cache and should not be launched merely as a smoke
test.

## Result and literature discipline

For numbers, use `docs/signoff_log.md` and the July completion audit—not old
narrative files. The persistent full-paper/repository vault is on EIT at
`/engrfs/project/jacobsn/hiqbal/literature/legalrag/`; tracked PDF hashes and
repository commits live under `wiki/literature/manifests/`.
