# Query Rewrite

Rewrite a legal research question into an optimized primary retrieval query PLUS two alternative queries that use different legal terminology to bridge terminological gaps.

## Rules

1. **Precise legal terms, no filler**: Expand legal terms with synonyms (e.g., "fired from job" -> "wrongful termination employment at-will discharge"). Strip conversational filler ("Can you tell me about", "I was wondering").
2. **MBE vocabulary with subject context**: Use formal bar exam terminology. Add subject area keywords (e.g., "negligence", "tort", "liability" for torts; "UCC", "contract", "breach" for contracts).
3. **Keep each query concise**: 10-30 words. Dense with legal keywords.
4. **Diversify alternatives**: The two alternatives MUST use substantially different terminology from the primary to catch passages using different vocabulary for the same concept.
5. **Be specific**: Replace vague terms with precise legal concepts (e.g., "get in trouble" -> "criminal liability elements").

## Output

```json
{
  "primary": "the main optimized retrieval query",
  "alternatives": [
    "first alternative query using different legal terminology",
    "second alternative query using different legal terminology"
  ]
}
```

Return ONLY the JSON object. No explanation, no markdown fences.

## Examples

Input: "What happens if someone breaks a contract?"
```json
{"primary": "breach of contract remedies damages expectation interest specific performance UCC common law", "alternatives": ["anticipatory repudiation material breach substantial performance rescission restitution", "contractual liability compensatory damages consequential damages mitigation duty"]}
```

Input: "Can the police search my car without a warrant?"
```json
{"primary": "Fourth Amendment warrantless automobile search exception probable cause vehicle search incident to arrest", "alternatives": ["Terry stop reasonable suspicion plain view doctrine motor vehicle exception exclusionary rule", "search and seizure automobile exception Carroll doctrine inventory search consent search"]}
```

Input: "Is a landlord responsible for fixing things in the apartment?"
```json
{"primary": "landlord tenant implied warranty habitability duty to repair residential lease covenant quiet enjoyment", "alternatives": ["premises liability lessor obligation constructive eviction uninhabitable conditions housing code", "rental property maintenance obligation tenant remedies rent withholding repair and deduct"]}
```
