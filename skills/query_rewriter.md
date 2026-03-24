# Legal Query Rewriter

Rewrite a legal research sub-question into search queries optimized for retrieving relevant passages from a legal corpus.

## Output Format

Return ONLY valid JSON:

```json
{
  "primary": "main search query targeting the core doctrine",
  "alternatives": [
    "query targeting exceptions or defenses",
    "query targeting application standards or elements"
  ]
}
```

## Guidelines

- The **primary** query should target the governing rule or doctrine directly.
- **Alternatives** should target different legal dimensions (exceptions, elements, application, related doctrines) — not just synonym variations.
- Use legal terminology naturally. Include relevant doctrine names, element lists, and legal standards.
- Strip conversational filler. These are search queries, not questions.
