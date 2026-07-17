---
title: Harness Decision
type: plan
tags: [harness, verifiers, inspect-ai, agent-lightning, verl]
created: 2026-07-17
updated: 2026-07-17
status: proposed
---

# Harness decision

## Recommendation

Use a deliberately staged stack:

1. **Canonical environment:** a small Verifiers environment wrapping Reasoning
   Gym plus one persistent Python action.
2. **Canonical record:** framework-neutral JSONL trajectories owned by this
   project, not by any one harness.
3. **Independent evaluation:** add an Inspect AI adapter after the first local
   environment smoke, if it can consume the same task and trace contract
   without forking semantics.
4. **Training backend:** direct SFT first; Prime RL/VERL only after Phase 0
   proves nontrivial teacher switching.
5. **Agent Lightning:** use when wrapping an existing complex agent or when its
   OpenTelemetry-style span store materially reduces instrumentation work.

The harness is infrastructure, not the contribution.

## Evidence-based comparison

| Option | Strength | Cost/risk | Decision |
|---|---|---|---|
| Verifiers | Direct Reasoning Gym integration; environments serve evaluation and RL; Python/sandbox abstractions | Current API is evolving; Python sandbox may require hosted/container setup | Primary prototype |
| Inspect AI | Mature model-agnostic evaluation, tools, sandboxes, multi-turn logs/viewer | Training is not its core job | Independent evaluator |
| Agent Lightning | Captures prompts, tool calls, rewards as structured spans; decouples arbitrary agents from trainers | Central store/trainer is excessive for a one-tool Phase 0 | Add only for richer agents or VERL bridge |
| NeMo Gym | Complete dataset+harness+verifier+state abstraction; Reasoning Gym and Verifiers integrations; multi-reward support | Larger operational surface; early and fast-moving | Strong fallback if Verifiers composition is awkward |
| VerlTool | Native multi-turn tool RL and environment state | Heavy, tied to VERL, unnecessary before behavior gate | Phase 2 backend |
| Custom-only | Maximum control and fastest first diagnostic | Easy to create brittle logging/training debt | Permit only as a thin reference adapter, not a new framework |

## Canonical trajectory schema

Every framework adapter must emit the same fields:

```json
{
  "episode_id": "stable paired identifier",
  "task_id": "generator-stable task id",
  "task_family": "prime_factorization",
  "difficulty": "medium",
  "split": "test",
  "model_id": "...",
  "teacher_or_student": "teacher",
  "skill_condition": "tool_verify_v1",
  "price_condition": {"tool": 2.0, "token": 0.0},
  "hard_limits": {"max_tokens": 2048, "max_tool_calls": 2},
  "messages": [],
  "actions": [],
  "task_reward_native": 1.0,
  "task_success_exact": 1,
  "usage": {"input_tokens": 0, "output_tokens": 0, "tool_calls": 1},
  "cost_components": {"tool": 2.0, "tokens": 0.0},
  "termination": "final_answer",
  "wall_time_s": 0.0,
  "code_version": "git sha",
  "environment_version": "manifest sha"
}
```

Never store only `reward = success - lambda * cost`. Store task reward, exact
success, every usage component, and the displayed price separately. This lets
the entire lambda grid be rescored and audited later.

## Required invariants

- Same task IDs and prompts at every price.
- Price is part of the observation, never baked into the verifier.
- Tool implementation is identical across teacher, student, and baselines.
- Task success is calculated before cost scalarization.
- Invalid tool calls, timeouts, and truncation have distinct termination codes.
- Raw trajectories are append-only; derived metrics are reproducible from them.
- A harness adapter must pass a golden trace test before any model comparison.

## Why not start with a large harness

Phase 0 needs fewer than five operations: sample task, expose price, run model,
execute Python if called, and verify. If the behavior gate fails, distributed
RL orchestration has no value. Reuse libraries at their natural boundary, but
do not let infrastructure decide the scientific object.
