from __future__ import annotations

from typing import List

from .context_axes import ContextInferenceEngine, split_primary_and_alternate
from .gematria import partition_bucket, simple_english_gematria
from .models import ContextCandidate, RetrievalItem, RetrievalResult
from .reflection import ReflectionEngine
from .storage import SQLiteMemoryStore
from .trust import compute_trust


class MemoryRetriever:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        reflection_engine: ReflectionEngine | None = None,
        context_engine: ContextInferenceEngine | None = None,
        include_resonance: bool = True,
    ) -> None:
        self.store = store
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.context_engine = context_engine or ContextInferenceEngine(store)
        self.include_resonance = include_resonance

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievalResult, object]:
        # v2: infer contexts from weighted signals; caller supplies only query text.
        contexts = self.context_engine.infer(query)
        primary_contexts, alternate_contexts = split_primary_and_alternate(contexts)

        candidate_ids = self.store.candidate_memory_ids_by_contexts([(c.axis, c.value) for c in primary_contexts])
        if not candidate_ids:
            # fallback: gematria entry bucket narrowing before broad search.
            query_bucket = partition_bucket(simple_english_gematria(query))
            candidate_ids = self.store.get_memory_ids_by_entry_bucket(query_bucket)

        candidate_rows = self.store.search_text_in_memories(candidate_ids, query)

        items: List[RetrievalItem] = []
        for row in candidate_rows:
            memory_id = int(row["memory_id"])
            raw_text = row["raw_text"]
            sidecars = self.store.get_sidecars_for_memory(memory_id)
            matched_contexts = self._contexts_matching_memory(primary_contexts, sidecars)

            context_score = self._compute_context_score(matched_contexts)
            text_score = self._compute_text_score(raw_text, query)
            trust_profile = self._compute_trust(memory_id, row["source_type"], sidecars)
            resonance_score, agreeing_axes = self._compute_resonance(memory_id, raw_text, query, matched_contexts, sidecars)

            if self.include_resonance:
                final = (
                    0.35 * context_score
                    + 0.25 * text_score
                    + 0.2 * trust_profile.computed_trust
                    + 0.2 * resonance_score
                )
            else:
                final = 0.45 * context_score + 0.3 * text_score + 0.25 * trust_profile.computed_trust

            items.append(
                RetrievalItem(
                    memory_id=memory_id,
                    raw_text=raw_text,
                    context_score=context_score,
                    text_score=text_score,
                    trust_score=trust_profile.computed_trust,
                    resonance_score=resonance_score,
                    final_score=final,
                    agreeing_axes=agreeing_axes,
                    matched_contexts=matched_contexts,
                )
            )

        items.sort(key=lambda x: x.final_score, reverse=True)
        items = items[:top_k]

        emergent_pairs = self._detect_emergent_resonance(items)
        reflection = self.reflection_engine.evaluate(
            primary_contexts,
            alternate_contexts,
            items,
            inference_depth=0,
            emergent_pairs=emergent_pairs,
        )

        retrieval_result = RetrievalResult(
            query=query,
            primary_contexts=primary_contexts,
            alternate_contexts=alternate_contexts,
            items=items,
            reflection_needed=reflection.triggered,
        )
        return retrieval_result, reflection

    def _contexts_matching_memory(self, contexts: List[ContextCandidate], sidecars) -> List[ContextCandidate]:
        matched = []
        for context in contexts:
            for s in sidecars:
                if s["axis"] == context.axis and s["value"] == context.value:
                    matched.append(context)
                    break
        return matched

    def _compute_context_score(self, matched_contexts: List[ContextCandidate]) -> float:
        if not matched_contexts:
            return 0.0
        return min(1.0, sum(c.score for c in matched_contexts) / len(matched_contexts))

    def _compute_text_score(self, raw_text: str, query: str) -> float:
        q_tokens = [t for t in query.lower().split() if t]
        if not q_tokens:
            return 0.0
        overlap = sum(1 for t in q_tokens if t in raw_text.lower())
        return min(1.0, overlap / max(1, len(q_tokens)))

    def _compute_resonance(self, memory_id: int, raw_text: str, query: str, matched_contexts, sidecars) -> tuple[float, list[str]]:
        """Count independent axis agreements and map via monotonic function.

        score = 1 - 1/(n+1), where n is number of agreeing axes.
        """

        agreeing_axes: list[str] = []
        query_tokens = [t for t in query.lower().split() if t]

        if matched_contexts:
            agreeing_axes.append("context")
        if any(t in raw_text.lower() for t in query_tokens):
            agreeing_axes.append("text_overlap")
        if any((s["value"] or "").lower() in query.lower() for s in sidecars if s["value"]):
            agreeing_axes.append("sidecar_value_overlap")

        query_bucket = partition_bucket(simple_english_gematria(query))
        entry_g = self.store.get_gematria_for_memory(memory_id, layer="entry")
        if entry_g and int(entry_g[0]["bucket"]) == query_bucket:
            agreeing_axes.append("gematria_entry_bucket")

        n = len(agreeing_axes)
        resonance = 1 - (1 / (n + 1))
        return resonance, agreeing_axes

    def _detect_emergent_resonance(self, items: List[RetrievalItem]) -> list[tuple[int, int]]:
        """Detect highly resonant memory pairs that are not explicitly linked.

        v2 heuristic: if two top items both have resonance >= 0.75 and do not share
        an explicit inferred association sidecar, treat as emergent resonance.
        """

        pairs: list[tuple[int, int]] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                if min(a.resonance_score, b.resonance_score) < 0.75:
                    continue

                a_sidecars = self.store.get_sidecars_for_memory(a.memory_id)
                linked = any(
                    s["sidecar_type"] == "inferred" and str(b.memory_id) in (s["value"] or "")
                    for s in a_sidecars
                )
                if not linked:
                    pairs.append((a.memory_id, b.memory_id))
        return pairs

    def _compute_trust(self, memory_id: int, source_type: str, sidecars) -> object:
        cor, uncor = self.store.get_corroboration_counts(memory_id)
        inferred_items = sum(1 for s in sidecars if s["sidecar_type"] == "inferred")
        summary_items = sum(1 for s in sidecars if s["sidecar_type"] == "summary")
        return compute_trust(
            memory_id=memory_id,
            source_type=source_type,
            corroborated_count=cor,
            uncorroborated_count=uncor,
            inferred_items=inferred_items,
            summary_items=summary_items,
        )
