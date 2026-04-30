# Agentic Legal RAG Angles - 2026-04-30

Purpose: decide whether there is a research story beyond the current N=200
method chase. The user-facing goal is still agentic RAG for the legal domain,
with possible workshop fit first and an EMNLP-style angle if the novelty is
strong enough.

## Read Of The Space

Agentic legal RAG is now an active target, not a blank area.

- L-MARS frames legal QA as structured multi-agent retrieval over web, local
  RAG, and case-law tools. It decomposes questions, retrieves through targeted
  searches, filters through a Judge Agent, and synthesizes cited answers. It
  explicitly reports the same benchmark-dependence we see locally: large gains
  on current-law questions, negligible gains on Bar Exam QA.
- LRAS frames the core legal-agent problem as an introspection deficit: models
  do not know when they need external verification. It trains active inquiry
  behavior with imitation learning and RL. Its limitations section points to
  longer interaction horizons and robustness to incomplete/irrelevant retrieved
  statutes.
- General agentic RAG surveys now emphasize the same future-work pressure
  points: reasoning efficiency, retrieval efficiency, adaptive retrieval
  control, dynamic tool selection, retrieval planning, and adaptive
  orchestration.
- Multi-agent RAG papers such as MA-RAG and RAGentA already claim modular
  planner/extractor/verifier style workflows. MBA-RAG already treats retrieval
  strategies as bandit arms with a reward balancing accuracy and cost.
- Recent memory/governance papers argue that multi-agent systems need shared,
  auditable state instead of isolated agents repeatedly reconstructing context.

Implication: "we built a legal multi-agent RAG system" is not enough. A publishable
angle needs to answer a sharper question.

## Best Research Problem

**Evidence-budgeted orchestration for legal RAG.**

Legal research agents must decide under a fixed budget:

1. whether the answer is likely in parametric knowledge or requires evidence,
2. which evidence source/method to spend budget on,
3. which specialized agent should consume that evidence,
4. what intermediate evidence state should be shared across agents,
5. when further retrieval or debate is no longer worth the cost.

This is narrower and more novel than generic agentic RAG. It also fits our
local evidence: heavy subagent pipelines often lose because they spend calls
without improving the shared evidence state, while simpler methods win when
the task's bottleneck is clear.

Possible thesis:

> Legal agentic RAG fails less because agents are missing, and more because
> evidence budgets are misallocated. We propose a bottleneck-aware controller
> that treats retrieval methods and specialized agents as costed arms, exposes
> a shared evidence ledger, and escalates only when cheap task and retrieval
> probes predict that additional evidence can change the answer.

## Why This Is Motivated

Legal is a good domain for this because retrieval is not uniformly helpful:

- current-law and statute-threshold questions need up-to-date external sources;
- bar-style doctrinal MC questions can already be solved from facts and legal
  priors, so more context may add noise;
- case/holding identification depends on fine-grained disambiguation rather
  than broad context volume;
- long-form legal RAG needs grounded citations and attribution, so evidence
  state and provenance matter.

The motivating failure is therefore not hallucination alone. It is **misplaced
legal research effort**: searching when reasoning is enough, reasoning when
current law is missing, summarizing evidence away before the final answerer can
audit it, or giving each agent private context with no shared provenance.

## Candidate Angles

### 1. Evidence-Budgeted Active Legal RAG

Core idea: a controller routes each question through a small set of method arms
using cheap static features plus retrieval probes.

Arms:

- direct / snap-only,
- simple RAG k=1,
- simple RAG k=5,
- multi-query,
- pseudo-doc / HyDE,
- iterative or multi-agent evidence synthesis.

Novelty claim:

- prior adaptive RAG routes mostly by query complexity or uncertainty;
- this routes by **legal bottleneck type** and **marginal expected value of
  evidence**;
- evaluation is cost-aware and uses legal-specific error decomposition.

Why feasible:

- `scripts/evaluate_routing_oracle.py` already shows oracle headroom:
  MuSiQue static best 37.0% vs oracle 57.0%; BarExam static best 85.5% vs
  oracle 93.5%.
- next step is offline learnability, not online agent deployment.

Workshop fit: strong.

EMNLP fit: plausible if the learned router beats static best under
leave-one-dataset-out evaluation and includes a clean analysis of when it fails.

### 2. Shared Evidence Ledger For Multi-Agent Legal RAG

Core idea: replace private subagent summaries with a shared evidence ledger:
every agent writes claims, citations, source type, jurisdiction/date metadata,
contradictions, and support status into a common schema. The final agent sees
both the summary and the source-backed ledger.

Novelty claim:

- many multi-agent RAG systems decompose roles, but they often pass natural
  language summaries;
- legal reasoning needs auditable provenance, contradiction tracking, and
  jurisdiction/time validity;
- the contribution is a governed evidence-state interface for legal agents,
  not just more agents.

Why feasible:

- our old `subagent_rag` failures are useful: the report-only path can
  over-abstain or lose raw evidence. A ledger is the obvious fix to test.
- `main.py` already has `evidence_store`, `audit_log`, and step status fields;
  `eval_harness.py` already has subagent variants that can be reworked into a
  ledger ablation.

Workshop fit: very strong, especially for legal AI / AI4Law.

EMNLP fit: weaker unless paired with clear quantitative gains and a general
multi-agent lesson.

### 3. Evidence-Constrained Multi-Agent Debate

Core idea: legal agents do not free-debate. They receive distinct roles
(proponent, skeptic, authority checker, jurisdiction/time checker), but every
argument must attach to a retrieved source or explicit parametric/no-source tag.
Disagreement triggers targeted retrieval only for the disputed claim.

Novelty claim:

- multi-agent debate is common, but legal debate without evidence constraints
  risks amplifying plausible hallucinations;
- targeted retrieval on disagreement points is a resource-sharing/decoding
  strategy: spend tokens where agents disagree, not across every gap.

Why feasible:

- existing `friend_foe_attribution`, `subagent_*`, and planning-table modes are
  close enough to prototype a debate/authority-checker variant.
- local audits already show that exposing snap letters or lossy reports can
  hurt, so role separation plus evidence constraints is a grounded fix.

Workshop fit: strong.

EMNLP fit: possible if the debate mechanism generalizes beyond legal MC and
shows cost-controlled gains on multi-hop/legal long-form tasks.

### 4. Legal RAG Error-Decomposition Benchmarking

Core idea: a paper about methodology rather than a new agent. Use factorial
evaluations to separate retrieval failure, evidence-utilization failure,
answer-option disambiguation, and stale-knowledge failure across legal tasks.

Novelty claim:

- LegalBench-RAG measures retrieval precision;
- Reasoning-Focused Legal Retrieval Benchmark shows legal RAG remains hard;
- Legal RAG Bench decomposes retrieval vs generator effects;
- our contribution would add method-level interventions and task-bottleneck
  labels across several legal benchmarks.

Why feasible:

- this is closest to current repo state and least risky.
- add Legal RAG Bench, MLEB-SCALR retrieval-only, and maybe LegalSearchQA-style
  current-law questions.

Workshop fit: strong.

EMNLP fit: moderate unless the benchmark artifact is clean and the taxonomy is
predictive, not just descriptive.

## Recommendation

The best path is a two-layer story:

1. **Workshop paper**: "Evidence-Budgeted Orchestration for Legal RAG." Lead
   with the legal-agent motivation and show a controller/ledger prototype plus
   diagnostic results.
2. **EMNLP attempt**: make it general enough by evaluating on legal QA plus
   MuSiQue/FRAMES, and by showing that bottleneck-aware routing beats fixed
   pipelines under a cost budget.

The multi-agent resource-sharing angle should be implemented as a **shared
evidence ledger**, not as more natural-language subagents. This is the legal
domain hook: legal work product needs traceable support, dates, jurisdictions,
and contradiction handling. It also explains why the current subagent variants
underperform: they summarize evidence, but they do not preserve enough
auditable state for the final decision.

## Immediate Next Work

1. Build `scripts/build_router_training_set.py` from existing logs:
   row-per-question features, arm correctness/cost, and oracle label.
2. Add a cheap static + retrieval-probe router baseline:
   static best vs learned router vs oracle, leave-one-dataset-out.
3. Design the evidence ledger schema:
   `claim`, `source_id`, `quote_span`, `jurisdiction`, `date`, `support`,
   `contradicts`, `agent_role`, `confidence`.
4. Prototype one ledger mode:
   `ledger_subagent_rag`: retrieve -> role agents write ledger entries ->
   final answerer receives ledger + compact evidence, not free-form reports.
5. Add Legal RAG Bench if possible:
   it is the best external benchmark for this story because it has long-form
   legal questions, supporting passages, and a published retrieval/reasoning
   decomposition.

## Source Notes

- L-MARS: legal multi-agent workflow with query decomposition, agentic search,
  judge verification, and benchmark-dependent retrieval gains. It leaves
  multi-turn mode evaluation to future work for open-ended legal research.
- LRAS: legal agentic search as an introspection problem; future work highlights
  longer interaction horizons and robustness to tool-induced errors.
- Towards Agentic RAG with Deep Reasoning: future work emphasizes reasoning
  efficiency, retrieval efficiency, adaptive retrieval control, dynamic tool
  selection, retrieval planning, and orchestration.
- MBA-RAG: bandit framing for retrieval-strategy selection under accuracy/cost
  reward.
- RAGentA / MA-RAG: multi-agent RAG already exists; role decomposition alone is
  not enough novelty.
- Collaborative/Governed Memory: shared, auditable memory/state is becoming a
  recognized multi-agent systems problem; legal RAG gives a concrete high-stakes
  version of it.
