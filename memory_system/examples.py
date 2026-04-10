from __future__ import annotations

from .ingest import MemoryIngestor, SidecarInput
from .retrieval import MemoryRetriever
from .storage import SQLiteMemoryStore


def seed_sample_data(store: SQLiteMemoryStore) -> None:
    ing = MemoryIngestor(store)

    # Scenario 1: recipe with competing relational context
    ing.ingest_memory(
        raw_text="Mom's lemon soup recipe: simmer garlic, rice, broth, then add lemon and egg.",
        source_type="direct_user",
        sidecars=[
            SidecarInput("context_tag", "domain", "cooking", 0.9),
            SidecarInput("context_tag", "person", "mom", 0.95),
            SidecarInput("summary", None, "Family lemon soup instructions", 0.7),
        ],
        corroborated=2,
    )
    ing.ingest_memory(
        raw_text="Later I cooked the same lemon soup for my lover on our anniversary.",
        source_type="direct_user",
        sidecars=[
            SidecarInput("context_tag", "domain", "cooking", 0.85),
            SidecarInput("context_tag", "person", "lover", 0.9),
            SidecarInput("inferred", "association", "same_recipe_as_mom_soup", 0.6),
        ],
        corroborated=1,
    )

    # Scenario 2: name with multiple referents, dominant context may switch
    ing.ingest_memory(
        raw_text="Alex from work approved the Q3 deployment checklist.",
        source_type="direct_user",
        sidecars=[
            SidecarInput("context_tag", "person", "alex", 0.9),
            SidecarInput("context_tag", "domain", "work", 0.9),
        ],
        corroborated=1,
    )
    ing.ingest_memory(
        raw_text="Alex from college sent me an old song mix last weekend.",
        source_type="quoted",
        sidecars=[
            SidecarInput("context_tag", "person", "alex", 0.9),
            SidecarInput("context_tag", "domain", "personal", 0.8),
            SidecarInput("inferred", "temporal", "historical", 0.55),
        ],
        uncorroborated=1,
    )

    # Scenario 3: imported low-trust memory
    ing.ingest_memory(
        raw_text="Imported note says the landlord definitely agreed to a permanent rent freeze.",
        source_type="imported_external",
        source_reference="csv_import_2025_11_14",
        sidecars=[
            SidecarInput("context_tag", "domain", "housing", 0.7),
            SidecarInput("context_tag", "source", "imported", 0.8),
            SidecarInput("inferred", "intent", "legal_commitment", 0.45),
            SidecarInput("summary", None, "Possible rent freeze commitment", 0.4),
        ],
        uncorroborated=2,
    )

    # Scenario 4 (new): ambiguity + resonance on unlinked memories.
    ing.ingest_memory(
        raw_text="Checklist note: alex requested song-level QA before release.",
        source_type="direct_user",
        sidecars=[
            SidecarInput("context_tag", "person", "alex", 0.85),
            SidecarInput("context_tag", "domain", "work", 0.7),
            SidecarInput("context_tag", "artifact", "song_mix", 0.7),
        ],
        corroborated=1,
    )


def run_demo() -> None:
    store = SQLiteMemoryStore(":memory:")
    store.init_schema()
    seed_sample_data(store)

    retriever = MemoryRetriever(store, include_resonance=True)
    queries = [
        "recipe mom lemon soup",
        "alex approved checklist",
        "imported rent freeze agreement",
        "alex song checklist",  # new ambiguity/resonance query
    ]

    for query in queries:
        result, reflection = retriever.retrieve(query)
        print("\n=== QUERY:", query)
        print("Primary contexts:", [(c.axis, c.value, round(c.score, 2)) for c in result.primary_contexts])
        print("Alternate contexts:", [(c.axis, c.value, round(c.score, 2)) for c in result.alternate_contexts])
        for item in result.items:
            print(
                f"- memory_id={item.memory_id} score={item.final_score:.3f} "
                f"trust={item.trust_score:.2f} resonance={item.resonance_score:.2f} axes={item.agreeing_axes}"
                f" text={item.raw_text}"
            )
        print("Reflection triggered:", reflection.triggered, "reasons=", reflection.reason_codes)
        if reflection.suggested_clarifying_questions:
            print("Questions:")
            for q in reflection.suggested_clarifying_questions:
                print("  *", q)


if __name__ == "__main__":
    run_demo()
