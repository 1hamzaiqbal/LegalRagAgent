# Classify and Route

## Classification Criteria

### Simple (`"simple"`)
- Asks about a single legal rule, definition, or standard
- Can be answered by retrieving one focused passage
- Examples: "What is consideration in contract law?", "Define hearsay", "What are the elements of battery?"

### Multi-hop (`"multi_hop"`)
- Requires combining multiple legal concepts or rules
- Involves analysis across different areas of law
- Asks about application of law to facts (hypotheticals)
- Requires comparing rules, exceptions, and defenses
- Examples: "Can a store be held liable for a slip and fall if the customer was texting?", "What rights does a criminal defendant have during a traffic stop that leads to a drug search?"

### Multiple-Choice Questions

- `simple`: Tests ONE legal concept applied to scenarios (e.g., "Which is consideration?")
- `multi_hop`: Requires analyzing MULTIPLE interacting concepts (e.g., "Will Plaintiff prevail on battery?" — battery + self-defense + transferred intent)
- When in doubt, classify as `multi_hop`

## Output

Return a JSON object with exactly these fields:

```json
{
  "query_type": "simple",
  "reasoning": "Brief one-sentence explanation of why this classification was chosen"
}
```

`query_type` must be either `"simple"` or `"multi_hop"`.

Return ONLY the JSON object. No explanation, no markdown fences.
