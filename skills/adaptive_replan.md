# Adaptive Replan

## Input

- `objective`: The original legal research question
- `current_step_id`: The step_id that was just executed
- `retrieved_content`: The passages returned by retrieval for the current step
- `plan_table`: The current Plan Table as a JSON array (with `expectation_achieved` filled for all previously completed steps, empty for the current and future steps)
- `accumulated_evidence` *(optional)*: JSON array of evidence summaries from all executed steps, each with `step_id`, `status`, `answer` (truncated), and `confidence`

## Task

1. **Evaluate the current step**: Compare `retrieved_content` against the `expected_answer` for the step matching `current_step_id`. Set `expectation_achieved` to one of:
   - `"Yes - <brief explanation>"` — retrieved content fully matches the expectation
   - `"Partial - <brief explanation>"` — retrieved content partially matches; note what was found and what is missing
   - `"No - <brief explanation>"` — retrieved content does not address the expectation at all

2. **Review pending steps**: Based on what was learned from the current retrieval, review all steps with empty `expectation_achieved`. You MAY:
   - **Modify** a pending step's `planned_action`, `retrieval_question`, or `expected_answer` to better align with what has been learned
   - **Add** new steps (assign the next available `step_id`) if the retrieval revealed unanticipated aspects
   - **Remove** pending steps that are no longer necessary given the evidence gathered

3. **Decide action**: Choose one of `next_step`, `retry`, or `complete`.

## Constraints

- **NEVER modify a step whose `expectation_achieved` is already filled.** Completed steps are immutable history.
- **Hard step cap: 5 steps maximum.** Do not add steps beyond this limit.
- **Stop on futility**: If 3+ consecutive completed steps all have `expectation_achieved` starting with `"No"`, return `"complete"` immediately.
- Each `retrieval_question` must be **self-contained** — do not reference previous step IDs or specific answer content.
- **Generalize on failure**: If a step failed because retrieval returned irrelevant content, the query was likely too fact-specific. When reviewing pending steps, rewrite their queries as abstract legal doctrine lookups (e.g., "What are the elements of negligent inspection?" instead of "What is the standard of care for a mechanic inspecting brakes?"). The corpus contains textbook rules, not fact-specific analyses.
- **Always review pending steps after a failure.** Do not leave pending queries unchanged when prior steps show the same query style is not working.

## Output

Return a JSON object with these fields:

```json
{
  "action": "next_step",
  "reasoning": "Why this action was chosen given the current state of the Plan Table",
  "updated_plan_table": [
    {
      "step_id": 1,
      "planned_action": "Rule Identification",
      "retrieval_question": "...",
      "expected_answer": "...",
      "expectation_achieved": "Yes - found all elements"
    },
    {
      "step_id": 2,
      "planned_action": "...",
      "retrieval_question": "...",
      "expected_answer": "...",
      "expectation_achieved": ""
    }
  ]
}
```

### Actions

- `"next_step"` — Proceed to the next pending step in the Plan Table. The next step is the lowest `step_id` with empty `expectation_achieved`.
- `"retry"` — The current step should be retried with a rephrased question. Update the current step's `retrieval_question` and `expected_answer` in the plan table (set `expectation_achieved` back to `""` for the retried step). Maximum 1 retry per step.
- `"complete"` — Evidence sufficiently addresses the objective, OR further retrieval is unlikely to improve results. No more steps to execute.

## Example

Input state — Step 2 was just executed with partial results:

```json
{
  "objective": "Can a store be held liable if a customer slips on a wet floor?",
  "current_step_id": 2,
  "retrieved_content": "A possessor of land has actual notice when...",
  "plan_table": [
    {"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What is the legal standard for premises liability for business invitees?", "expected_answer": "Duty of care owed to invitees", "expectation_achieved": "Yes - found duty to inspect, discover, and make safe or warn"},
    {"step_id": 2, "planned_action": "Breach Analysis", "retrieval_question": "What constitutes breach of duty in slip-and-fall cases?", "expected_answer": "Actual or constructive notice, reasonable time to remedy", "expectation_achieved": ""},
    {"step_id": 3, "planned_action": "Defense Analysis", "retrieval_question": "What defenses exist in slip-and-fall cases?", "expected_answer": "Comparative negligence, assumption of risk", "expectation_achieved": ""}
  ]
}
```

Output — evaluates step 2, modifies pending step 3:

```json
{
  "action": "next_step",
  "reasoning": "Step 2 found actual notice but not constructive notice. Adjusting step 3 to focus on constructive notice doctrine before covering defenses.",
  "updated_plan_table": [
    {"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What is the legal standard for premises liability for business invitees?", "expected_answer": "Duty of care owed to invitees", "expectation_achieved": "Yes - found duty to inspect, discover, and make safe or warn"},
    {"step_id": 2, "planned_action": "Breach Analysis", "retrieval_question": "What constitutes breach of duty in slip-and-fall cases?", "expected_answer": "Actual or constructive notice, reasonable time to remedy", "expectation_achieved": "Partial - found actual notice standard but constructive notice not covered"},
    {"step_id": 3, "planned_action": "Notice Doctrine", "retrieval_question": "What is constructive notice in premises liability and how is it established?", "expected_answer": "Time-based test for constructive notice, mode-of-operation approach", "expectation_achieved": ""}
  ]
}
```

### Example 2: Generalizing after failure

Step 1 asked "What is the standard of care for a mechanic inspecting brakes?" and retrieval returned irrelevant content (score: No). Pending steps 2 and 3 also have fact-specific queries that will likely fail the same way.

```json
{
  "action": "next_step",
  "reasoning": "Step 1 failed because the query was too fact-specific — the corpus has general negligence doctrine, not mechanic-specific rules. Generalizing pending steps to match corpus language.",
  "updated_plan_table": [
    {"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What is the standard of care for a mechanic inspecting brakes?", "expected_answer": "...", "expectation_achieved": "No - retrieved general negligence principles but nothing mechanic-specific"},
    {"step_id": 2, "planned_action": "Breach Analysis", "retrieval_question": "What are the elements required to prove breach of duty in a negligence claim?", "expected_answer": "Failure to exercise reasonable care, deviation from standard of care", "expectation_achieved": ""},
    {"step_id": 3, "planned_action": "Causation Analysis", "retrieval_question": "What is required to establish proximate cause in a negligence action?", "expected_answer": "But-for causation and foreseeability", "expectation_achieved": ""}
  ]
}
```

Return ONLY the JSON object. No explanation, no markdown fences.
