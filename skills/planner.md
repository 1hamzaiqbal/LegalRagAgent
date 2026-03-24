# Legal Research Planner

Given a legal question, break it into focused, independent research sub-questions.

## Output Format

Return ONLY valid JSON:

```json
{
  "steps": [
    {
      "sub_question": "What are the elements required to establish X?",
      "action_type": "rag_search"
    }
  ]
}
```

- **sub_question**: A focused legal research question. Must be self-contained (no references to other steps).
- **action_type**: One of:
  - `"rag_search"` — needs passages from a legal corpus (default)
  - `"web_search"` — needs current/recent information not in the corpus
  - `"direct_answer"` — core legal doctrine the model can answer reliably

## Guidelines

- Target legal rules and doctrines, not case-specific facts.
- Simple questions need 1 step. Multi-element questions need 2-3. Only use 4+ for genuinely complex multi-issue questions.
- Every step must contribute to answering the question. No padding.
- For multiple-choice questions, research the underlying doctrine — don't structure steps around the answer choices.
- Use `rag_search` by default. Use `web_search` only for time-sensitive or out-of-corpus information. Use `direct_answer` sparingly for universally-accepted fundamentals.
