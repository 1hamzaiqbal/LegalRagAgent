"""Shared evidence-ledger primitives for agentic legal RAG experiments.

The online harness can remain unchanged while we iterate on the ledger contract.
This module defines the minimal structured state that subagents should pass
forward instead of free-form reports alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORT_STATUSES = {"supports", "contradicts", "mixed", "irrelevant", "unknown"}


@dataclass
class EvidenceLedgerEntry:
    """A source-backed claim produced by a worker/retriever/checker agent."""

    claim: str
    source_id: str
    quote: str = ""
    support: str = "unknown"
    agent_role: str = "unknown"
    jurisdiction: str = ""
    effective_date: str = ""
    source_type: str = ""
    confidence: float | None = None
    contradicts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.claim.strip():
            raise ValueError("EvidenceLedgerEntry.claim is required")
        if not self.source_id.strip():
            raise ValueError("EvidenceLedgerEntry.source_id is required")
        if self.support not in SUPPORT_STATUSES:
            allowed = ", ".join(sorted(SUPPORT_STATUSES))
            raise ValueError(f"support must be one of: {allowed}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "claim": self.claim.strip(),
            "source_id": self.source_id.strip(),
            "quote": self.quote.strip(),
            "support": self.support,
            "agent_role": self.agent_role.strip() or "unknown",
            "jurisdiction": self.jurisdiction.strip(),
            "effective_date": self.effective_date.strip(),
            "source_type": self.source_type.strip(),
            "confidence": self.confidence,
            "contradicts": list(self.contradicts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvidenceLedgerEntry":
        entry = cls(
            claim=str(data.get("claim", "")),
            source_id=str(data.get("source_id", data.get("idx", ""))),
            quote=str(data.get("quote", "")),
            support=str(data.get("support", "unknown")).lower(),
            agent_role=str(data.get("agent_role", "unknown")),
            jurisdiction=str(data.get("jurisdiction", "")),
            effective_date=str(data.get("effective_date", data.get("date", ""))),
            source_type=str(data.get("source_type", data.get("source", ""))),
            confidence=_coerce_confidence(data.get("confidence")),
            contradicts=[str(item) for item in data.get("contradicts", [])],
            metadata=dict(data.get("metadata", {})),
        )
        entry.validate()
        return entry

    def to_prompt_block(self, index: int) -> str:
        data = self.to_dict()
        parts = [
            f"[Ledger {index}] {data['support'].upper()}",
            f"Claim: {data['claim']}",
            f"Source: {data['source_id']}",
        ]
        if data["quote"]:
            parts.append(f"Quote: {data['quote']}")
        provenance = []
        if data["source_type"]:
            provenance.append(f"type={data['source_type']}")
        if data["jurisdiction"]:
            provenance.append(f"jurisdiction={data['jurisdiction']}")
        if data["effective_date"]:
            provenance.append(f"date={data['effective_date']}")
        if data["agent_role"]:
            provenance.append(f"agent={data['agent_role']}")
        if data["confidence"] is not None:
            provenance.append(f"confidence={data['confidence']:.2f}")
        if provenance:
            parts.append("Provenance: " + "; ".join(provenance))
        if data["contradicts"]:
            parts.append("Contradicts: " + ", ".join(data["contradicts"]))
        return "\n".join(parts)


def _coerce_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc


def normalize_ledger(entries: list[EvidenceLedgerEntry | dict[str, Any]]) -> list[EvidenceLedgerEntry]:
    normalized = [
        entry if isinstance(entry, EvidenceLedgerEntry) else EvidenceLedgerEntry.from_mapping(entry)
        for entry in entries
    ]
    for entry in normalized:
        entry.validate()
    return normalized


def ledger_to_prompt(entries: list[EvidenceLedgerEntry | dict[str, Any]]) -> str:
    normalized = normalize_ledger(entries)
    if not normalized:
        return "No structured evidence ledger entries were produced."
    return "\n\n".join(entry.to_prompt_block(i) for i, entry in enumerate(normalized, start=1))


def entries_from_evidence_store(
    evidence_store: list[dict[str, Any]],
    *,
    claim: str,
    agent_role: str,
    support: str = "supports",
) -> list[EvidenceLedgerEntry]:
    """Wrap retrieved passages as provisional source-backed ledger entries."""
    entries: list[EvidenceLedgerEntry] = []
    for item in evidence_store:
        text = str(item.get("text", ""))
        quote = text[:500].strip()
        entries.append(
            EvidenceLedgerEntry(
                claim=claim,
                source_id=str(item.get("idx", "")),
                quote=quote,
                support=support,
                agent_role=agent_role,
                source_type=str(item.get("source", "")),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"idx", "text", "source", "cross_encoder_score"}
                } | {"cross_encoder_score": item.get("cross_encoder_score")},
            )
        )
    return entries
