"""Trust model for memory sources and sidecars."""

from __future__ import annotations

from .models import TrustProfile

BASE_TRUST_BY_SOURCE = {
    "direct_user": 0.92,
    "quoted": 0.7,
    "inferred": 0.5,
    "summary": 0.45,
    "imported_external": 0.4,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_trust(
    memory_id: int,
    source_type: str,
    corroborated_count: int,
    uncorroborated_count: int,
    inferred_items: int,
    summary_items: int,
) -> TrustProfile:
    base = BASE_TRUST_BY_SOURCE.get(source_type, 0.5)
    notes: list[str] = [f"base:{base:.2f} ({source_type})"]

    score = base
    score += 0.08 * corroborated_count
    if corroborated_count:
        notes.append(f"+corroboration:{corroborated_count}")

    score -= 0.06 * uncorroborated_count
    if uncorroborated_count:
        notes.append(f"-uncorroborated:{uncorroborated_count}")

    # Penalize heavy dependence on inferred and summary layers.
    score -= 0.03 * inferred_items
    score -= 0.02 * summary_items
    if inferred_items:
        notes.append(f"-inferred_items:{inferred_items}")
    if summary_items:
        notes.append(f"-summary_items:{summary_items}")

    computed = _clamp(score)

    return TrustProfile(
        memory_id=memory_id,
        source_type=source_type,
        base_trust=base,
        corroborated_count=corroborated_count,
        uncorroborated_count=uncorroborated_count,
        inferred_items=inferred_items,
        summary_items=summary_items,
        computed_trust=computed,
        notes=notes,
    )
