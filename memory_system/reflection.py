from __future__ import annotations

from typing import List

from .models import ContextCandidate, ReflectionResult, RetrievalItem


class ReflectionEngine:
    def evaluate(
        self,
        primary_contexts: List[ContextCandidate],
        alternate_contexts: List[ContextCandidate],
        retrieval_items: List[RetrievalItem],
        inference_depth: int = 0,
        emergent_pairs: list[tuple[int, int]] | None = None,
    ) -> ReflectionResult:
        reasons: list[str] = []
        questions: list[str] = []
        emergent_pairs = emergent_pairs or []

        if not retrieval_items:
            reasons.append("no_supporting_memories")
            questions.append("I found no matching memory. Should I widen by timeframe, source type, or person?")

        if len(primary_contexts) >= 2:
            gap = primary_contexts[0].score - primary_contexts[1].score
            if gap < 0.15:
                reasons.append("context_ambiguity")
                questions.append(
                    f"Context conflict: should I prioritize {primary_contexts[0].axis}='{primary_contexts[0].value}' "
                    f"or {primary_contexts[1].axis}='{primary_contexts[1].value}'?"
                )

        if retrieval_items:
            top_trust = retrieval_items[0].trust_score
            if top_trust < 0.55:
                reasons.append("low_trust_top_result")
                snippet = retrieval_items[0].raw_text[:80].strip()
                questions.append(
                    f"Top memory is low-trust: "
                    f"'{snippet}...'. Should I limit to direct_user or corroborated memories only?"
                )

        if alternate_contexts and primary_contexts and alternate_contexts[0].score > (primary_contexts[0].score - 0.2):
            reasons.append("strong_alternate_context")
            if retrieval_items:
                top = retrieval_items[0].raw_text[:50].strip()
                questions.append(
                    f"'{top}...' also supports alternate context '{alternate_contexts[0].value}'. "
                    "Switch primary context?"
                )

        if inference_depth > 0:
            reasons.append("high_inference_depth")
            questions.append("This result depends on multiple inferred hops. Should I restrict to direct links?")

        if emergent_pairs:
            reasons.append("emergent_resonance")
            pair_labels = ", ".join([f"{a}<->{b}" for a, b in emergent_pairs[:2]])
            questions.append(
                f"I detected strong multi-axis resonance between unlinked memories ({pair_labels}). "
                "Should I create a provisional association sidecar?"
            )

        # Targeted conflict question from top two memories.
        if len(retrieval_items) >= 2 and abs(retrieval_items[0].final_score - retrieval_items[1].final_score) < 0.08:
            reasons.append("competing_memories")
            a = retrieval_items[0].raw_text[:55].strip()
            b = retrieval_items[1].raw_text[:55].strip()
            questions.append(f"Which memory is intended: '{a}...' or '{b}...'?" )

        triggered = len(reasons) > 0
        confidence = 1.0 - min(0.85, 0.15 * len(reasons))

        alt_sets = []
        if alternate_contexts:
            alt_sets.append(alternate_contexts[:2])

        return ReflectionResult(
            triggered=triggered,
            reason_codes=reasons,
            confidence=confidence,
            inference_depth=inference_depth,
            suggested_clarifying_questions=questions,
            alternative_context_sets=alt_sets,
        )
