"""Regression tests for the shared evidence-ledger contract."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from evidence_ledger import EvidenceLedgerEntry, entries_from_evidence_store, ledger_to_prompt


def test_ledger_entry_validates_required_fields():
    entry = EvidenceLedgerEntry(
        claim="The statute requires notice before termination.",
        source_id="housing_123",
        quote="A landlord shall provide notice...",
        support="supports",
        confidence=0.75,
    )
    assert entry.to_dict()["support"] == "supports"


def test_ledger_rejects_invalid_support_status():
    entry = EvidenceLedgerEntry(claim="x", source_id="s", support="maybe")
    try:
        entry.validate()
    except ValueError as exc:
        assert "support must be one of" in str(exc)
    else:
        raise AssertionError("invalid support status should fail validation")


def test_prompt_block_preserves_provenance():
    prompt = ledger_to_prompt(
        [
            {
                "claim": "The case applies the exception.",
                "source_id": "case_7",
                "quote": "The exception applies when...",
                "support": "supports",
                "agent_role": "authority_checker",
                "jurisdiction": "CA",
                "date": "2024-01-01",
                "source": "case_law",
                "confidence": 0.8,
            }
        ]
    )
    assert "Claim: The case applies the exception." in prompt
    assert "Source: case_7" in prompt
    assert "jurisdiction=CA" in prompt
    assert "agent=authority_checker" in prompt


def test_entries_from_evidence_store_wraps_retrieved_passages():
    entries = entries_from_evidence_store(
        [
            {
                "idx": "doc_1",
                "text": "This is the controlling passage.",
                "source": "statute",
                "cross_encoder_score": 3.2,
            }
        ],
        claim="This source may answer the issue.",
        agent_role="retriever",
    )
    assert entries[0].source_id == "doc_1"
    assert entries[0].metadata["cross_encoder_score"] == 3.2
    assert "controlling passage" in entries[0].quote


if __name__ == "__main__":
    test_ledger_entry_validates_required_fields()
    test_ledger_rejects_invalid_support_status()
    test_prompt_block_preserves_provenance()
    test_entries_from_evidence_store_wraps_retrieved_passages()
    print("All evidence-ledger tests passed.")
