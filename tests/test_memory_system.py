from memory_system.context_axes import ContextInferenceEngine
from memory_system.examples import seed_sample_data
from memory_system.ingest import MemoryIngestor
from memory_system.retrieval import MemoryRetriever
from memory_system.storage import SQLiteMemoryStore


def build_store():
    store = SQLiteMemoryStore(":memory:")
    store.init_schema()
    return store


def test_raw_memory_is_immutable():
    store = build_store()
    ing = MemoryIngestor(store)
    mid = ing.ingest_memory("Original text", "direct_user")

    try:
        store.update_raw_memory_forbidden(mid, "Mutated text")
        assert False, "Expected immutable protection"
    except RuntimeError:
        pass


def test_gematria_layers_are_stored_in_sidecar_table():
    store = build_store()
    ing = MemoryIngestor(store)
    mid = ing.ingest_memory("Alpha beta. Gamma", "direct_user")

    all_layers = store.get_gematria_for_memory(mid)
    layers = {row["layer"] for row in all_layers}
    assert "word" in layers
    assert "sentence" in layers
    assert "entry" in layers


def test_context_inference_is_not_just_direct_tag_lookup():
    store = build_store()
    seed_sample_data(store)
    engine = ContextInferenceEngine(store)

    candidates = engine.infer("checklist for alex")
    assert candidates
    assert any(c.axis == "person" and c.value == "alex" for c in candidates)


def test_context_first_retrieval_prefers_context_match():
    store = build_store()
    seed_sample_data(store)
    retriever = MemoryRetriever(store)

    result, _ = retriever.retrieve("recipe mom lemon soup")

    assert result.items, "Should find at least one memory"
    top = result.items[0]
    assert "mom" in top.raw_text.lower()


def test_reflection_triggers_for_low_trust_imported_memory():
    store = build_store()
    seed_sample_data(store)
    retriever = MemoryRetriever(store)

    _, reflection = retriever.retrieve("imported rent freeze agreement")
    assert reflection.triggered is True
    assert any(
        code in reflection.reason_codes
        for code in [
            "low_trust_top_result",
            "strong_alternate_context",
            "context_ambiguity",
            "no_supporting_memories",
        ]
    )


def test_resonance_score_is_available_and_nonzero_for_matching_query():
    store = build_store()
    seed_sample_data(store)
    retriever = MemoryRetriever(store, include_resonance=True)

    result, _ = retriever.retrieve("alex song checklist")
    assert result.items
    assert any(item.resonance_score > 0 for item in result.items)
