# Plan Synthesis

Corpus: bar exam prep covering Con Law, Contracts/UCC, Crim Law/Procedure, Evidence, Real Property, Torts, Civ Pro.

## Input

- `objective`: The user's legal research question
- `query_type`: Either `"simple"` or `"multi_hop"`

## Output Format

Return a JSON array representing the **Plan Table**. Each entry has these fields:

```json
[
  {
    "step_id": 1,
    "planned_action": "Phase name describing the research action",
    "retrieval_question": "Specific retrieval question for this step",
    "expected_answer": "Brief description of what you expect to find",
    "expectation_achieved": ""
  }
]
```

- `planned_action`: A short label for the research phase (e.g., "Rule Identification", "Exception Analysis", "Application to Facts", "Defense Analysis").
- `retrieval_question`: A self-contained, concise question used for retrieval. Do not reference other steps.
- `expected_answer`: Your prediction of what the retrieval should return — key concepts, elements, or doctrines you anticipate finding. Be specific enough to evaluate later.
- `expectation_achieved`: Always `""` (empty string) at plan creation time.

## Rules

1. **Simple queries**: Exactly 1 step that directly answers the question.
2. **Multi-hop queries**: Multiple steps. Start with the foundational rule, then build toward analysis. Each step should address a distinct aspect of the objective.
3. Each `retrieval_question` must be **self-contained** — do not reference other step IDs or answers.
4. Keep retrieval questions concise and specific — they are used as retrieval queries.
5. `expected_answer` should be concrete enough to judge success (e.g., "A list of 4-5 elements" rather than "Some information about the topic").

## Examples

Simple — "What are the elements of adverse possession?"
```json
[{"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What are the elements required for adverse possession of land?", "expected_answer": "A list of required elements: actual and exclusive, open and notorious, adverse/hostile, continuous for statutory period", "expectation_achieved": ""}]
```

Multi-hop — "Can a store be held liable if a customer slips on a wet floor?"
```json
[
  {"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What is the legal standard for premises liability for business invitees?", "expected_answer": "Duty of care owed to invitees: duty to inspect, discover, and make safe or warn of dangerous conditions", "expectation_achieved": ""},
  {"step_id": 2, "planned_action": "Breach Analysis", "retrieval_question": "What constitutes breach of duty in slip-and-fall premises liability cases?", "expected_answer": "Actual or constructive notice of hazard, reasonable time to discover and remedy", "expectation_achieved": ""},
  {"step_id": 3, "planned_action": "Defense Analysis", "retrieval_question": "What defenses are available in premises liability slip-and-fall cases?", "expected_answer": "Comparative negligence, assumption of risk, open and obvious doctrine", "expectation_achieved": ""}
]
```

Return ONLY the JSON array. No explanation, no markdown fences.
