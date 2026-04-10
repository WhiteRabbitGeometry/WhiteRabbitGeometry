# Memory Coordinate System (v1) Design

## Architecture Overview

This prototype implements a **context-gated recall** architecture rather than ordinary global search. It separates immutable raw memories from revisable interpretation layers and performs retrieval in stages:

1. **Store canonical memory** as immutable raw text.
2. **Attach sidecars** (context links, trust annotations, optional summaries, inferred metadata) in separate tables.
3. **Infer and rank contexts** for a query before searching memory text.
4. **Narrow candidate memories** to those connected to top contexts.
5. **Retrieve and score** memories with trust-aware weighting.
6. **Run reflection circuit** when confidence is weak, ambiguity is high, or trust is low.

This keeps source fidelity while allowing adaptive interpretation.

## Major Components

- `memory_system.models`
  - Dataclasses for query and retrieval/reflection outputs.
- `memory_system.storage`
  - SQLite schema + persistence layer.
  - Enforces raw-memory immutability and sidecar separation.
- `memory_system.context_axes`
  - Context dimensions and simple query-to-context inference.
- `memory_system.gematria`
  - Deterministic partition axis (no semantic claims).
- `memory_system.trust`
  - Source-type trust baseline + corroboration adjustments.
- `memory_system.retrieval`
  - Context-first retrieval flow.
- `memory_system.reflection`
  - Uncertainty handling and clarifying-question generation.
- `memory_system.ingest`
  - Ingestion helpers for canonical memory + sidecars.
- `memory_system.examples`
  - End-to-end sample scenarios.

## v1 Assumptions / Defaults

- SQLite is the only storage backend.
- Raw memory text is immutable once inserted.
- Sidecars are append-only (updates create new sidecar records or adjustments).
- Query understanding is lightweight rule-based, not LLM-dependent.
- Embeddings are optional and not used as primary retrieval.
- Trust and context scoring are transparent heuristics.
- Reflection triggers when:
  - top context scores are close,
  - low trust dominates top results,
  - weak evidence/support,
  - multiple referent contexts compete.

## Missing / Future Decisions

- Better temporal reasoning and recurrence modeling.
- Rich association graph traversal and decay/reinforcement dynamics.
- Pluggable context inference engines.
- Optional embedding index for within-context ranking only.
- Conflict resolution policy over long histories.
- User feedback loop for updating dominant context priors.
- Fine-grained provenance (line-level citation inside raw memory).
