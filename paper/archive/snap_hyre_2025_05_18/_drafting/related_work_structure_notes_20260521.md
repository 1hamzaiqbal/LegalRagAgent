# Related RAG Paper Structure Notes - 2026-05-21

Papers checked locally:

- HyDE, `hyde_2212.10496.pdf`
- CRAG, `crag_2401.15884.pdf`
- Self-RAG, `selfrag_2310.11511.pdf`

Useful structure pattern:

1. Start with the failure mode, not the implementation details.
   HyDE motivates a retrieval-query mismatch before defining hypothetical
   document embeddings. CRAG motivates bad retrieval before presenting its
   corrective evaluator. Self-RAG motivates indiscriminate retrieval before
   explaining reflection tokens.

2. Let Method explain the design decision.
   Method should answer why the new mechanism exists and what information it
   changes. Hyperparameters, providers, seeds, and inclusion rules belong in
   Evaluation.

3. Keep the main figure concept-first.
   HyDE/CRAG-style method figures are useful when they show the single routing
   or transformation decision. The Snap-HyRE Figure 1 should therefore show
   only the private snap answer, search passage, retrieval, and final answer
   path, not every baseline row.

4. Put answer accuracy and retrieval quality next to each other.
   Generated retrieval papers usually need both retrieval-side and task-side
   metrics. Here, answer accuracy alone hides whether Snap-HyRE retrieved the
   right legal evidence; Hit@5/MRR@5 make the claim inspectable.

5. Report cost as part of the experimental design.
   Since Snap-HyRE and HyDE are two-logical-call methods, the paper should
   distinguish logical calls from answer-stage tokens. The usage table should
   not be sold as full end-to-end cost unless first-stage generation tokens are
   included.

Edits applied from these notes:

- Figure 1 became a Snap-HyRE-only design figure.
- Method now starts with design rationale.
- Evaluation now owns models, baselines, retrieval stack, cache replay,
  metrics, and efficiency definitions.
- Results now include a short cost/context subsection.
