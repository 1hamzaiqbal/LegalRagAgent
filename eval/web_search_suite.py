"""Fixed multiple-choice suite for testing web-search tool planning.

These questions are intentionally outside the BarExam QA corpus and are framed
to require current, externally grounded information. The goal is to test
whether the pipeline chooses `web_search` when the question explicitly depends
on current facts.
"""

from typing import List, Dict


WEB_SEARCH_QUESTIONS: List[Dict] = [
    {
        "label": "web_legal_attorney_general_20260316",
        "subject": "CURRENT_LEGAL",
        "question": (
            "As of March 16, 2026, who is the Attorney General of the United States?"
        ),
        "correct_answer": "B",
        "choices": {
            "A": "Merrick Garland",
            "B": "Pamela Bondi",
            "C": "D. John Sauer",
            "D": "Lee Zeldin",
        },
    },
    {
        "label": "web_legal_solicitor_general_20260316",
        "subject": "CURRENT_LEGAL",
        "question": (
            "As of March 16, 2026, who is the Solicitor General of the United States?"
        ),
        "correct_answer": "C",
        "choices": {
            "A": "Elizabeth Prelogar",
            "B": "Pamela Bondi",
            "C": "D. John Sauer",
            "D": "Noel Francisco",
        },
    },
    {
        "label": "web_legal_sec_chair_20260316",
        "subject": "CURRENT_LEGAL",
        "question": (
            "As of March 16, 2026, who is serving as Chair of the U.S. Securities and Exchange Commission?"
        ),
        "correct_answer": "A",
        "choices": {
            "A": "Paul S. Atkins",
            "B": "Gary Gensler",
            "C": "Mark T. Uyeda",
            "D": "Hester M. Peirce",
        },
    },
    {
        "label": "web_legal_ftc_chair_20260316",
        "subject": "CURRENT_LEGAL",
        "question": (
            "As of March 16, 2026, who is serving as Chair of the Federal Trade Commission?"
        ),
        "correct_answer": "D",
        "choices": {
            "A": "Lina M. Khan",
            "B": "Melissa Holyoak",
            "C": "Rebecca Kelly Slaughter",
            "D": "Andrew N. Ferguson",
        },
    },
    {
        "label": "web_legal_fda_commissioner_20260316",
        "subject": "CURRENT_LEGAL",
        "question": (
            "As of March 16, 2026, who is the Commissioner of Food and Drugs at the FDA?"
        ),
        "correct_answer": "B",
        "choices": {
            "A": "Robert M. Califf",
            "B": "Martin A. Makary",
            "C": "Janet Woodcock",
            "D": "Mandy Cohen",
        },
    },
    {
        "label": "web_general_president_20260316",
        "subject": "CURRENT_GENERAL",
        "question": (
            "As of March 16, 2026, who is the President of the United States?"
        ),
        "correct_answer": "A",
        "choices": {
            "A": "Donald J. Trump",
            "B": "Joseph R. Biden Jr.",
            "C": "Kamala D. Harris",
            "D": "J.D. Vance",
        },
    },
    {
        "label": "web_general_super_bowl_lx_20260316",
        "subject": "CURRENT_GENERAL",
        "question": (
            "As of March 16, 2026, which team won Super Bowl LX on February 8, 2026?"
        ),
        "correct_answer": "C",
        "choices": {
            "A": "Kansas City Chiefs",
            "B": "Philadelphia Eagles",
            "C": "Seattle Seahawks",
            "D": "New England Patriots",
        },
    },
    {
        "label": "web_general_oscars_best_picture_20260316",
        "subject": "CURRENT_GENERAL",
        "question": (
            "As of March 16, 2026, which film won Best Picture at the 98th Academy Awards on March 15, 2026?"
        ),
        "correct_answer": "D",
        "choices": {
            "A": "Sinners",
            "B": "Hamnet",
            "C": "Train Dreams",
            "D": "One Battle after Another",
        },
    },
    {
        "label": "web_general_nobel_peace_20260316",
        "subject": "CURRENT_GENERAL",
        "question": (
            "As of March 16, 2026, who won the 2025 Nobel Peace Prize?"
        ),
        "correct_answer": "B",
        "choices": {
            "A": "Nihon Hidankyo",
            "B": "Maria Corina Machado",
            "C": "Narges Mohammadi",
            "D": "Volodymyr Zelenskyy",
        },
    },
    {
        "label": "web_general_world_series_20260316",
        "subject": "CURRENT_GENERAL",
        "question": (
            "As of March 16, 2026, which team won the 2025 World Series?"
        ),
        "correct_answer": "A",
        "choices": {
            "A": "Los Angeles Dodgers",
            "B": "New York Yankees",
            "C": "Toronto Blue Jays",
            "D": "Seattle Mariners",
        },
    },
]


def select_web_search_queries(n: int = 10) -> List[Dict]:
    """Return the fixed web-search benchmark suite."""
    return WEB_SEARCH_QUESTIONS[: min(n, len(WEB_SEARCH_QUESTIONS))]
