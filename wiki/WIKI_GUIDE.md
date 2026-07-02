# LegalRagAgent Wiki — Schema & Maintenance Guide

This is the **schema** for the LegalRagAgent knowledge wiki (the LLM-Wiki pattern,
sibling of the BoundEO wiki at `~/grad/boundeo/wiki`). The wiki is a persistent,
compounding, Obsidian-linked synthesis layer over our sources, ideas,
implementations, and results. **The LLM writes and maintains the wiki; the human
curates sources and asks questions.** Open `wiki/` as an Obsidian vault to browse
the graph.

## Three layers
1. **Raw sources** (`references/`, gitignored, immutable) — downloaded papers/PDFs,
   cloned repos, data dumps. Read from, never edit. Heavy copies archived on the
   WUSTL EIT cluster at `/engrfs/tmp/jacobsn/hiqbal_legalrag/references/`.
2. **The wiki** (`wiki/`, tracked markdown) — LLM-generated, interlinked. This layer.
3. **The schema** (this file) — conventions + workflows. Co-evolved over time.

The wiki does **not** duplicate the repo's own docs — it *links* them. Eval harness
lives in `eval/`; run ledger in `logs/experiments.jsonl`; citation gate in
[docs/signoff_log.md](../docs/signoff_log.md); generated analyses in `docs/generated/`. Wiki pages
cross-reference all of these (relative links work from `wiki/`, e.g.
`[signoff](../docs/signoff_log.md)`).

## Directory layout
```
wiki/
  WIKI_GUIDE.md   ← this schema
  START_HERE.md   ← orienting read for new agents/humans
  index.md        ← content catalog (every page, by category, one-line summary)
  log.md          ← chronological append-only op log
  sources/        ← one page per ingested paper/repo/post
  concepts/       ← ideas/regimes/mechanisms (vocabulary-gap, query-drift, qpp, …)
  methods/        ← things implemented in OUR code (scope, eval-harness, …)
  results/        ← dated result-family records (synthesis + links out to signed rows)
  reviews/        ← submission postmortems: criticism inventories + responses
references/       ← (gitignored) raw papers/repos the source pages summarize
```

## Page conventions
- **Filename**: kebab-case, descriptive (`koblex-parser.md`, `query-drift.md`).
- **Links**: Obsidian double-bracket wikilinks between wiki pages (by filename
  without `.md`), standard markdown relative links to repo files. Link
  **liberally** — a wikilink to a page that doesn't exist yet is a valid TODO
  marker, not an error.
- **Frontmatter** (YAML, for Obsidian Dataview):
```yaml
---
title: Human Readable Title
type: source | concept | method | result | review | hub
tags: [legal-rag, qpp, ...]
created: 2026-07-02
updated: 2026-07-02
status: stub | draft | maintained
# source pages: url, local (path under references/), authors, year, venue
# method pages: code (repo path), impl_status: validated | prototype | unbuilt
# result pages: date, verdict: win | negative | inconclusive, evidence (paths)
---
```
- **Body shape by type**:
  - **source**: TL;DR · key claims (each with *our-relevance*) · how it bears on
    our reviews/criticisms · differentiation (how our work differs, honestly) ·
    concept/method links · raw-source path.
  - **concept**: definition · why it matters to us · status in our work ·
    related links · open questions.
  - **method**: what it does · where (code link + impl_status) · backing concepts ·
    results it produced · validation gate.
  - **result**: setup · numbers (link detail logs + signoff entries) · verdict ·
    what it changed · source/method links.
  - **review**: verbatim-faithful criticism inventory · per-criticism assessment
    (valid? addressed-since? open?) · what it demands of the next submission.
  - **hub**: an overview that orients (index, roadmap, a track).

## Skeptic's discipline (carries over from the repo)
- **Verify before trusting** — especially negatives and "we already addressed X"
  claims. Check numbers against detail logs / [docs/signoff_log.md](../docs/signoff_log.md); flag
  inference vs stated fact.
- **Traceability** — every result page links its evidence paths.
- **No silent staleness** — when a new source contradicts a page, note the
  contradiction on the page (don't overwrite); the disagreement is knowledge.
- **Reviewer-grade honesty** — the ICML AI4Law rejection happened partly because
  internal framing drifted from what the tables showed. Wiki pages state the
  weakest-baseline caveat wherever a delta is quoted.

## Workflows
- **Ingest** (new source): download into `references/` → read it → write/update a
  `sources/` page → update every `concepts/` page it touches → add to `index.md` →
  append to `log.md`. One source can touch 10–15 pages.
- **Query** (a question): read `index.md` → drill into pages → synthesize with
  citations. File good answers back as new pages so explorations compound.
- **Lint** (health check): contradictions, stale claims, orphan pages, missing
  cross-references. Record findings in `log.md`.

## Indexing & logging
- `index.md` is **content-oriented**: read first when answering. Update every ingest.
- `log.md` is **chronological**: append-only, entries `## [YYYY-MM-DD] <op> | <title>`.
