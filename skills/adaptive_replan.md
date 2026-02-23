# Adaptive Replan

## Input

- `objective`: The original legal research question
- `accumulated_context`: List of completed research steps, each with `step_id`, `question`, `answer`, `confidence` (0.0-1.0), `status` ("completed"/"failed")

## Decision Rules

1. If a step failed (confidence < 0.4), you may retry with a **rephrased query** using different legal vocabulary.
2. Consider what aspects of the objective remain unanswered by the accumulated evidence.
3. **Hard step cap: 3 completed steps maximum.** If there are already 3+ completed steps, you MUST return `"complete"`. No exceptions.
4. Each new question must be **self-contained** — do not reference previous step IDs or specific answer choices. Research legal concepts, not individual options.
5. **STOP retrying when the corpus lacks coverage.** If 3+ consecutive steps have ALL failed with confidence below 0.35, return `"complete"` immediately.
6. When all failed steps have similar confidence scores (within 0.05), rephrasing won't help.

## Output

```json
{
  "action": "next_step",
  "phase": "Phase name for the new step",
  "question": "The new retrieval question",
  "expectation": "What a good answer should contain",
  "reasoning": "Why this step is needed given what we know so far"
}
```

### Actions

- `"next_step"` — Unanswered aspect of the objective. Provide `phase`, `question`, `expectation`, `reasoning`.
- `"retry"` — Failed step should be retried with different wording. Provide `phase`, `question`, `expectation`, `reasoning`.
- `"complete"` — Evidence sufficiently addresses the objective, OR further retrieval is unlikely to improve results. Provide only `reasoning`.

```json
{
  "action": "complete",
  "reasoning": "The accumulated evidence covers all aspects..."
}
```

Return ONLY the JSON object. No explanation, no markdown fences.
