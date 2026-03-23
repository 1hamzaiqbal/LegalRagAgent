"""Curated playtest cases for manual pipeline refinement."""

PLAYTEST_CASES = [
    {
        "name": "bar_multihop_success",
        "profile": "full_parallel",
        "question": (
            "A police officer stopped a vehicle without reasonable suspicion, found drugs "
            "during a warrantless search, and the defendant seeks suppression. "
            "What are the defendant's strongest Fourth Amendment arguments and how would "
            "a court analyze application of the exclusionary rule?"
        ),
        "goal": "Doctrinal multi-hop success case with multiple completed steps.",
    },
    {
        "name": "partial_evidence_case",
        "profile": "full_parallel",
        "question": (
            "What did the Supreme Court hold in Loper Bright Enterprises v. Raimondo "
            "regarding Chevron deference, and how does this affect existing administrative law precedents?"
        ),
        "goal": "Current-facts plus doctrinal mix where at least one step may be partial.",
    },
    {
        "name": "implied_warranty_remodel",
        "profile": "full_parallel_aspect",
        "question": (
            "A homeowner hired a contractor to remodel their kitchen. The contractor used "
            "substandard materials and the work was defective. The homeowner wants to sue. "
            "What legal theories are available and what must be proven for each?"
        ),
        "goal": "Known weak case used to refine retrieval, escalation, and synthesis.",
    },
    {
        "name": "web_smoke_test",
        "profile": "full_parallel",
        "question": "As of June 28, 2024, what did the Supreme Court hold in Loper Bright Enterprises v. Raimondo?",
        "goal": "Smoke test for explicit current-facts / web-search behavior.",
    },
]

PLAYTEST_CASES_BY_NAME = {case["name"]: case for case in PLAYTEST_CASES}


def list_playtest_case_names() -> list[str]:
    """Return the stable ordered set of curated playtest names."""
    return [case["name"] for case in PLAYTEST_CASES]


def resolve_playtest_cases(case_name: str | None = None) -> list[dict]:
    """Return either one named case or the full curated suite."""
    if not case_name:
        return list(PLAYTEST_CASES)
    if case_name not in PLAYTEST_CASES_BY_NAME:
        known = ", ".join(list_playtest_case_names())
        raise KeyError(f"Unknown playtest case '{case_name}'. Known cases: {known}")
    return [PLAYTEST_CASES_BY_NAME[case_name]]
