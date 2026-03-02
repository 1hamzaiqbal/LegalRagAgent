# Classify and Plan

Corpus: bar exam prep covering Con Law, Contracts/UCC, Crim Law/Procedure, Evidence, Real Property, Torts, Civ Pro.

## Input

- `objective`: The user's legal research question

## Task

1. **Classify** the objective as `"simple"` or `"multi_hop"`.
2. **Generate** a Plan Table matching the classification.

### Classification Criteria

- **Simple**: Asks about a single legal rule, definition, or standard. One retrieval step suffices.
- **Multi-hop**: Requires combining multiple legal concepts, applying law to facts, or analyzing rules with exceptions/defenses. Multiple retrieval steps needed.
- Multiple-choice questions: single-concept MC → simple; multi-concept MC → multi_hop. When in doubt, `multi_hop`.

## Output Format

Return a JSON object with exactly these fields:

```json
{
  "query_type": "simple",
  "plan_table": [
    {
      "step_id": 1,
      "planned_action": "Phase name describing the research action",
      "retrieval_question": "Specific retrieval question for this step",
      "expected_answer": "Brief description of what you expect to find",
      "expectation_achieved": ""
    }
  ]
}
```

- `query_type`: Either `"simple"` or `"multi_hop"`.
- `planned_action`: Short label for the research phase (e.g., "Rule Identification", "Exception Analysis", "Defense Analysis").
- `retrieval_question`: Self-contained, concise question used for retrieval. Do not reference other steps.
- `expected_answer`: Concrete prediction of what retrieval should return (e.g., "A list of 4-5 elements" not "Some information").
- `expectation_achieved`: Always `""` at plan creation time.

## Rules

1. **Simple**: Exactly 1 step that directly answers the question.
2. **Multi-hop**: Multiple steps. Start with the foundational rule, then build toward analysis. Each step should address a distinct aspect.
3. Each `retrieval_question` must be **self-contained** — do not reference other step IDs or answers.
4. Keep retrieval questions concise and specific — they are used as retrieval queries.

## Examples

Simple — "What are the elements of adverse possession?"
```json
{"query_type": "simple", "plan_table": [{"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What are the elements required for adverse possession of land?", "expected_answer": "A list of required elements: actual and exclusive, open and notorious, adverse/hostile, continuous for statutory period", "expectation_achieved": ""}]}
```

Multi-hop — "Can a store be held liable if a customer slips on a wet floor?"
```json
{"query_type": "multi_hop", "plan_table": [{"step_id": 1, "planned_action": "Rule Identification", "retrieval_question": "What is the legal standard for premises liability for business invitees?", "expected_answer": "Duty of care owed to invitees: duty to inspect, discover, and make safe or warn of dangerous conditions", "expectation_achieved": ""}, {"step_id": 2, "planned_action": "Breach Analysis", "retrieval_question": "What constitutes breach of duty in slip-and-fall premises liability cases?", "expected_answer": "Actual or constructive notice of hazard, reasonable time to discover and remedy", "expectation_achieved": ""}, {"step_id": 3, "planned_action": "Defense Analysis", "retrieval_question": "What defenses are available in premises liability slip-and-fall cases?", "expected_answer": "Comparative negligence, assumption of risk, open and obvious doctrine", "expectation_achieved": ""}]}
```

Return ONLY the JSON object. No explanation, no markdown fences.
