from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .gematria import derive_layered_values
from .storage import SQLiteMemoryStore


@dataclass(frozen=True)
class SidecarInput:
    sidecar_type: str
    axis: Optional[str]
    value: str
    confidence: float = 0.8


class MemoryIngestor:
    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def ingest_memory(
        self,
        raw_text: str,
        source_type: str,
        source_reference: str | None = None,
        sidecars: Iterable[SidecarInput] | None = None,
        corroborated: int = 0,
        uncorroborated: int = 0,
    ) -> int:
        memory_id = self.store.insert_raw_memory(raw_text, source_type, source_reference)

        # v2: persist layered gematria in sidecar table (word/sentence/entry)
        # as deterministic partition data only.
        for g in derive_layered_values(raw_text):
            self.store.add_gematria_sidecar(
                memory_id=memory_id,
                layer=g.layer,
                position=g.position,
                token=g.token,
                gematria_value=g.value,
                bucket=g.bucket,
            )

        for sidecar in sidecars or []:
            self.store.add_sidecar(
                memory_id=memory_id,
                sidecar_type=sidecar.sidecar_type,
                axis=sidecar.axis,
                value=sidecar.value,
                confidence=sidecar.confidence,
            )

        for _ in range(corroborated):
            self.store.add_corroboration(memory_id, "corroborated")
        for _ in range(uncorroborated):
            self.store.add_corroboration(memory_id, "uncorroborated")

        return memory_id
