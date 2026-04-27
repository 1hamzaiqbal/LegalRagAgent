# ARCHIVED 2026-04-27 — superseded by docs/rigour_signoff.md
# Method Implementation Audit 2026-04-27

## run_multi_hyde_diverse

Verdict: retrieval fields are written correctly:
`1123: "retrieved_ids": retrieval["retrieved_ids"],`
`1124: "gold_retrieved": retrieval["gold_retrieved"],`
Fallback routing is tagged:
`1097: routed_to = "single_hyde_fallback_empty_gen"` and `1127: out["routed_to"] = routed_to`.
Bug: empty generation is not raised:
`1094: if not hyde_passages:`
`1095: raw_hyde_passages = [raw or question_intermediate]`
`1096: hyde_passages = [_sanitize_intermediate_text(raw or question_intermediate, fallback=question_intermediate)]`.
Final answer also has no blank-response check after `1111: answer = _llm_call(...)`.

## run_iter_hyde

Verdict: retrieval fields are written correctly:
`4403: "retrieved_ids": list(dict.fromkeys(all_retrieved_ids)),`
`4404: "gold_retrieved": gold_retrieved,`
Bug: empty LLM outputs are silently converted or treated as early exit:
`4288: current_hyde = _sanitize_intermediate_text(initial_raw, fallback=question_intermediate).strip()`
`4289: if not current_hyde:`
`4290: current_hyde = question_intermediate`
`4370: if not next_hyde:`
`4371: early_exit = True`
`4372: break`.
There is no `routed_to` field for these fallback paths, and final answer line 4391 has no empty-response check.

## run_friend_foe_attribution

Verdict: correct for a no-retrieval mode:
`3638: "evidence_store": [],`
`3639: "retrieved_ids": [],`
`3640: "gold_retrieved": False,`
Empty responses are raised, not silently used:
`3582: if not (snap_text or "").strip():`
`3583: raise RuntimeError("friend_foe_attribution snap pass returned empty response")`
and similarly `3611-3618` for self, foe, and control reviews. No fallback path exists, so `routed_to` is not applicable.
