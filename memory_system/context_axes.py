"""Context axes and v2 context inference engine.

v2 upgrade: contexts are inferred from multiple signals, not simply matched tags.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .gematria import partition_bucket, simple_english_gematria
from .models import ContextCandidate
from .storage import SQLiteMemoryStore


DEFAULT_AXES = ["domain", "intent", "temporal", "source", "person", "artifact", "gematria_bucket"]


@dataclass(frozen=True)
class QueryHint:
    token: str
    axis: str
    value: str
    weight: float


@dataclass(frozen=True)
class ContextScoreBreakdown:
    lexical: float
    frequency: float
    resonance: float


def build_default_hints() -> List[QueryHint]:
    return [
        QueryHint("recipe", "domain", "cooking", 0.8),
        QueryHint("cook", "intent", "instruction_lookup", 0.7),
        QueryHint("mom", "person", "mom", 0.95),
        QueryHint("lover", "person", "lover", 0.9),
        QueryHint("alex", "person", "alex", 0.9),
        QueryHint("imported", "source", "imported", 0.75),
        QueryHint("quote", "source", "quoted", 0.7),
        QueryHint("yesterday", "temporal", "recent", 0.5),
        QueryHint("last", "temporal", "historical", 0.45),
    ]


class ContextInferenceEngine:
    """Infer and rank contexts from weak/medium/strong signals.

    Signal weighting:
    - lexical cues from query: weak
    - frequency in memory store: medium
    - multi-axis resonance with memories: strong
    """

    def __init__(self, store: SQLiteMemoryStore, hints: Iterable[QueryHint] | None = None) -> None:
        self.store = store
        self.hints = list(hints or build_default_hints())

    def infer(self, query: str, top_n: int = 6) -> List[ContextCandidate]:
        tokens = [t.strip().lower() for t in query.split() if t.strip()]
        score_map: Dict[tuple[str, str], ContextScoreBreakdown] = {}
        evidence_map: Dict[tuple[str, str], List[str]] = defaultdict(list)

        lexical_map = self._lexical_signal(tokens)
        freq_map = self._frequency_signal()

        all_candidates = set(lexical_map.keys()) | set(freq_map.keys())
        for candidate in list(all_candidates):
            resonance = self._resonance_signal(query, candidate)
            lexical = lexical_map.get(candidate, 0.0)
            frequency = freq_map.get(candidate, 0.0)
            score_map[candidate] = ContextScoreBreakdown(lexical=lexical, frequency=frequency, resonance=resonance)

            if lexical > 0:
                evidence_map[candidate].append(f"lexical:{lexical:.2f}")
            if frequency > 0:
                evidence_map[candidate].append(f"frequency:{frequency:.2f}")
            if resonance > 0:
                evidence_map[candidate].append(f"resonance:{resonance:.2f}")

        if not score_map:
            return [
                ContextCandidate(axis="intent", value="general_recall", score=0.25, evidence=["fallback:no_candidates"])
            ]

        ranked = []
        for (axis, value), breakdown in score_map.items():
            # Strongly prioritize resonance, then frequency, then lexical.
            total = min(1.0, 0.2 * breakdown.lexical + 0.3 * breakdown.frequency + 0.5 * breakdown.resonance)
            ranked.append(ContextCandidate(axis=axis, value=value, score=total, evidence=evidence_map[(axis, value)]))

        ranked.sort(key=lambda c: c.score, reverse=True)

        # Keep only likely contexts.
        filtered = [c for c in ranked if c.score >= 0.15]
        return (filtered or ranked)[:top_n]

    def _lexical_signal(self, tokens: List[str]) -> Dict[tuple[str, str], float]:
        scores: Dict[tuple[str, str], float] = defaultdict(float)
        for token in tokens:
            for hint in self.hints:
                if hint.token in token:
                    scores[(hint.axis, hint.value)] += hint.weight
        # normalize to [0,1]
        return {k: min(v, 1.0) for k, v in scores.items()}

    def _frequency_signal(self) -> Dict[tuple[str, str], float]:
        freq = self.store.get_context_frequencies()
        if not freq:
            return {}
        max_f = max(freq.values())
        if max_f <= 0:
            return {}
        return {k: v / max_f for k, v in freq.items()}

    def _resonance_signal(self, query: str, candidate: tuple[str, str]) -> float:
        axis, value = candidate
        memory_ids = self.store.get_memory_ids_for_context(axis, value)
        if not memory_ids:
            return 0.0

        rows = self.store.get_memories(memory_ids)
        query_tokens = set(t.lower() for t in query.split() if t)
        query_bucket = partition_bucket(simple_english_gematria(query))

        # Strong signal: best per-memory multi-axis agreement.
        best = 0.0
        for row in rows:
            axes = 0
            text = row["raw_text"].lower()
            if any(t in text for t in query_tokens):
                axes += 1

            sidecars = self.store.get_sidecars_for_memory(int(row["memory_id"]))
            if any((s["value"] or "").lower() in query.lower() for s in sidecars if s["value"]):
                axes += 1

            entry_gem = self.store.get_gematria_for_memory(int(row["memory_id"]), layer="entry")
            if entry_gem and int(entry_gem[0]["bucket"]) == query_bucket:
                axes += 1

            if row["source_type"].replace("_", " ") in query.lower():
                axes += 1

            score = 1 - (1 / (axes + 1))
            best = max(best, score)

        return best


def split_primary_and_alternate(candidates: List[ContextCandidate], primary_n: int = 2) -> tuple[List[ContextCandidate], List[ContextCandidate]]:
    primary = candidates[:primary_n]
    alternate = candidates[primary_n:]
    return primary, alternate
