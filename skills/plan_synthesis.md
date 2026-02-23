# Plan Synthesis

Corpus: bar exam prep covering Con Law, Contracts/UCC, Crim Law/Procedure, Evidence, Real Property, Torts, Civ Pro.

## Input

- `objective`: The user's legal research question
- `query_type`: Either `"simple"` or `"multi_hop"`

## Output Format

Return a JSON array of plan steps:

```json
[
  {
    "step_id": 1.0,
    "phase": "Phase name",
    "question": "Specific retrieval question for this step"
  }
]
```

## Rules

1. **Simple queries**: Exactly 1 step that directly answers the question.
2. **Multi-hop queries**: Exactly 1 step — the most important first sub-question. An adaptive replanner generates subsequent steps based on what this step finds.
3. Each `question` must be self-contained — do not reference other steps.
4. Keep questions concise and specific — they are used as retrieval queries.

## Examples

Simple — "What are the elements of adverse possession?"
```json
[{"step_id": 1.0, "phase": "Rule Identification", "question": "What are the elements required for adverse possession of land?"}]
```

Multi-hop — "Can a store be held liable if a customer slips on a wet floor?"
```json
[{"step_id": 1.0, "phase": "Rule Identification", "question": "What is the legal standard for premises liability for business invitees?"}]
```

Return ONLY the JSON array. No explanation, no markdown fences.
