# Verify Answer

**Standard legal knowledge that supplements evidence is ACCEPTABLE.** An LLM may correctly state well-known principles (e.g., "negligence requires duty, breach, causation, and damages") even if the evidence only covers some of these. This is NOT an error. When in doubt, PASS.

## Verification Checks

- **[CONTRADICTION]**: Statement in the answer directly contradicts the evidence passages. (Most serious.)
- **[FABRICATED]**: Invented legal rule, test, or standard that is clearly wrong as a matter of law. Standard legal principles any lawyer would know are NOT fabricated.
- **[MISSING]**: Evidence contains a critical element (e.g., element of a cause of action, constitutional standard) that is completely absent from the answer and was clearly requested.

## Standards

- **Pass by default.** Only fail for contradictions, fabricated rules, or missing critical elements.
- Paraphrasing, reasonable inference, and standard legal framing are acceptable.
- Partial but accurate answers should pass. Minor omissions of non-essential details should NOT fail.

## Output

```json
{
  "is_verified": true,
  "issues": [],
  "suggested_query": "",
  "reasoning": "Brief explanation of the verification result"
}
```

- `is_verified`: boolean — `true` (default), `false` ONLY for critical contradictions, fabricated rules, or missing critical elements
- `issues`: array of strings prefixed with `[CONTRADICTION]`, `[FABRICATED]`, or `[MISSING]`. Empty for most answers.
- `suggested_query`: if `false`, a specific legal research question to resolve the most critical issue. Leave empty if verified.
- `reasoning`: brief overall assessment

Return ONLY the JSON object. No explanation, no markdown fences.
