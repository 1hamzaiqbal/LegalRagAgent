"""Inline prompt definitions and prompt-version helpers.

Read this module when you need to change the runtime's inline prompt surface.
Skill-backed prompts still live under `skills/`.
"""

from __future__ import annotations

from typing import Dict

from .core import get_prompt_version
from .models import ExperimentProfile, RAG_STRATEGY_ASPECT

COLLECTIONS_REGISTRY = {
    "legal_passages": (
        "Bar exam study materials, case law, and legal doctrine "
        "(torts, contracts, property, constitutional law, criminal law, evidence, etc.)"
    ),
    "housing_statutes": (
        "US housing statutes across all 50 states — landlord-tenant law, eviction, "
        "security deposits, habitability, lease termination, rent control"
    ),
}

ROUTER_PROMPT = f"""You are a legal research router. Given a legal question, decide which document collection(s) to search.

Available collections:
{chr(10).join(f'- "{name}": {desc}' for name, desc in COLLECTIONS_REGISTRY.items())}

Return ONLY valid JSON — no prose, no markdown fences:
{{"collections": ["collection_name_1"]}}

Rules:
- Choose the collection(s) most likely to contain relevant passages for this question.
- Most questions need only ONE collection. Use multiple only if the question clearly spans both domains.
- If uncertain, default to "legal_passages" (it has the broadest coverage).
"""

MC_ANSWER_PROMPT = """You are selecting the single best answer to a legal multiple-choice question.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "answer": "A",
  "reasoning": "One short sentence explaining why this option is best supported."
}

Rules:
- Choose exactly one of the listed answer letters.
- Base the choice on the research findings and synthesized analysis, not on guesswork.
- If the synthesized answer is partially uncertain, still choose the best-supported option from the listed choices.
- Never answer with a free-form string like "none of the above" unless that is literally one of the answer choices.
- Compare the actual text of the answer choices against the facts. An option can state a generally true doctrine and still be the wrong answer if another option fits the facts more precisely.
- Explicitly weigh the strongest competing option before choosing the final letter.
- Prefer the option that best resolves the exam question as written, not the option that merely sounds most legally sophisticated.
"""

COMPLETENESS_CHECK_PROMPT = """You are evaluating whether the accumulated research evidence is sufficient to fully answer the original legal question.

Given the original question, the research findings, and the synthesized answer, assess completeness.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "complete": true,
  "reasoning": "One sentence explaining why the evidence is or is not sufficient.",
  "missing_topics": []
}

Fields:
- `complete`: true if the synthesized answer adequately addresses the original question; false if there are significant gaps that additional research could fill.
- `reasoning`: Brief explanation of your assessment.
- `missing_topics`: If complete is false, list 1-3 focused follow-up questions suitable for new research steps.

Guidelines:
- Only mark incomplete if there is a clearly identifiable gap that would materially change the answer.
- If the question is simple, one well-supported step is usually enough.
- Be cautious about marking complete when a core issue is supported only by partial research or when a step explicitly names missing doctrine that would sharpen the element analysis.
- A partial step can still be enough if the missing detail is peripheral, but not if the synthesized answer relies on that step for a decisive legal theory.
- If the question includes answer choices, mark complete only when the research is strong enough to support one listed option and rule out the strongest competing options.
- If a multiple-choice answer depends on a step marked `partial` or `false`, prefer another research round focused on that doctrinal gap rather than marking the run complete.
- If a multiple-choice answer depends mainly on a step marked `support=support_only` or `origin=fallback_direct_answer`, prefer another research round rather than treating the issue as fully resolved.
- Maximum 3 rounds of research are allowed, so be conservative about requesting more.
"""

LLM_SNAP_PROMPT = """Answer the following legal question. Reason through it step by step, then give your final answer as **Answer: (X)**

Be concise but thorough. Identify the key legal issue, apply the relevant rule, and choose the best answer."""

ARBITRATION_PROMPT = """You are an expert legal arbitrator resolving a disagreement between two analyses of the same question.

ANALYSIS A was produced by a legal expert reasoning from doctrinal knowledge alone (no external sources).
ANALYSIS B was produced by a research pipeline that retrieved passages from a legal corpus and synthesized them.

The two analyses reached DIFFERENT conclusions. Your job is to determine which answer is correct.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "chosen": "A",
  "reasoning": "One to two sentences explaining which analysis better fits the specific facts."
}

Rules:
- Focus on the SPECIFIC FACTS of the question, not which analysis sounds more sophisticated.
- If Analysis B's retrieved evidence directly contradicts Analysis A on a factual or doctrinal point, prefer B.
- If Analysis B's evidence is tangential, addresses a different legal theory, or discusses doctrine that doesn't match the question's facts, prefer A.
- When in doubt, prefer the simpler, more direct legal theory over an elaborate one.
- An answer that correctly identifies the decisive legal issue (even without citations) is better than an answer that thoroughly researches the wrong issue.
"""


def inline_prompt_versions(profile: ExperimentProfile) -> Dict[str, str]:
    """Return the prompt versions captured in run artifacts."""
    versions = {
        "router.inline": get_prompt_version("router.inline", ROUTER_PROMPT),
        "completeness.inline": get_prompt_version("completeness.inline", COMPLETENESS_CHECK_PROMPT),
        "mc_answer.inline": get_prompt_version("mc_answer.inline", MC_ANSWER_PROMPT),
    }
    skill_names = ["planner", "synthesize_and_cite", "synthesizer"]
    if profile.use_query_rewrite:
        skill_names.append("query_rewriter")
    if profile.rag_strategy == RAG_STRATEGY_ASPECT:
        skill_names.append("aspect_query_rewriter")
    if profile.use_judge:
        skill_names.extend(["judge", "verifier"])
    for name in skill_names:
        versions[f"skill:{name}"] = get_prompt_version(name)
    return versions
