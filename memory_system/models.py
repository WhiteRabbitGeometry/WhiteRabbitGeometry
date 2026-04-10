from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class RawMemoryRecord:
    """Canonical, immutable memory text."""

    memory_id: int
    raw_text: str
    created_at: datetime
    source_type: str
    source_reference: Optional[str] = None


@dataclass(frozen=True)
class MetadataSidecar:
    """Derived interpretation attached to raw memory.

    Sidecars are provisional and revisable. They should not mutate raw memory.
    """

    sidecar_id: int
    memory_id: int
    sidecar_type: str
    axis: Optional[str]
    value: str
    confidence: float
    created_at: datetime


@dataclass(frozen=True)
class ContextCandidate:
    axis: str
    value: str
    score: float
    evidence: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrustProfile:
    memory_id: int
    source_type: str
    base_trust: float
    corroborated_count: int
    uncorroborated_count: int
    inferred_items: int
    summary_items: int
    computed_trust: float
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalItem:
    memory_id: int
    raw_text: str
    context_score: float
    text_score: float
    trust_score: float
    resonance_score: float
    final_score: float
    agreeing_axes: List[str] = field(default_factory=list)
    matched_contexts: List[ContextCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    primary_contexts: List[ContextCandidate]
    alternate_contexts: List[ContextCandidate]
    items: List[RetrievalItem]
    reflection_needed: bool


@dataclass(frozen=True)
class ReflectionResult:
    triggered: bool
    reason_codes: List[str]
    confidence: float
    inference_depth: int
    suggested_clarifying_questions: List[str]
    alternative_context_sets: List[List[ContextCandidate]]
