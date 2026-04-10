from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


class SQLiteMemoryStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_memory (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_reference TEXT,
                created_at TEXT NOT NULL,
                immutable INTEGER NOT NULL DEFAULT 1,
                -- v1 fields retained for backward compatibility; v2 gematria lives in sidecar table.
                gematria_value INTEGER NOT NULL DEFAULT 0,
                gematria_bucket INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS metadata_sidecar (
                sidecar_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                sidecar_type TEXT NOT NULL,
                axis TEXT,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES raw_memory(memory_id)
            );

            CREATE TABLE IF NOT EXISTS gematria_sidecar (
                gematria_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                layer TEXT NOT NULL, -- word|sentence|entry
                position INTEGER NOT NULL,
                token TEXT NOT NULL,
                gematria_value INTEGER NOT NULL,
                bucket INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES raw_memory(memory_id)
            );

            CREATE TABLE IF NOT EXISTS corroboration (
                corroboration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES raw_memory(memory_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sidecar_axis_value ON metadata_sidecar(axis, value);
            CREATE INDEX IF NOT EXISTS idx_sidecar_memory ON metadata_sidecar(memory_id);
            CREATE INDEX IF NOT EXISTS idx_gematria_layer_bucket ON gematria_sidecar(layer, bucket);
            CREATE INDEX IF NOT EXISTS idx_gematria_memory ON gematria_sidecar(memory_id);
            """
        )
        self.conn.commit()

    def insert_raw_memory(self, raw_text: str, source_type: str, source_reference: str | None = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO raw_memory(raw_text, source_type, source_reference, created_at, immutable)
            VALUES (?, ?, ?, ?, 1)
            """,
            (raw_text, source_type, source_reference, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_sidecar(self, memory_id: int, sidecar_type: str, axis: str | None, value: str, confidence: float) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO metadata_sidecar(memory_id, sidecar_type, axis, value, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (memory_id, sidecar_type, axis, value, confidence, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_gematria_sidecar(
        self, memory_id: int, layer: str, position: int, token: str, gematria_value: int, bucket: int
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO gematria_sidecar(memory_id, layer, position, token, gematria_value, bucket, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, layer, position, token, gematria_value, bucket, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_corroboration(self, memory_id: int, kind: str, note: str = "") -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO corroboration(memory_id, kind, note, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (memory_id, kind, note, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_memory(self, memory_id: int) -> sqlite3.Row | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM raw_memory WHERE memory_id = ?", (memory_id,))
        return cur.fetchone()

    def get_memories(self, memory_ids: Iterable[int] | None = None) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        if memory_ids is None:
            cur.execute("SELECT * FROM raw_memory")
            return cur.fetchall()

        ids = list(memory_ids)
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        cur.execute(f"SELECT * FROM raw_memory WHERE memory_id IN ({placeholders})", ids)
        return cur.fetchall()

    def update_raw_memory_forbidden(self, memory_id: int, new_text: str) -> None:
        raise RuntimeError("raw_memory is immutable; create sidecars instead")

    def candidate_memory_ids_by_contexts(self, contexts: Iterable[tuple[str, str]]) -> List[int]:
        contexts = list(contexts)
        if not contexts:
            cur = self.conn.cursor()
            cur.execute("SELECT memory_id FROM raw_memory")
            return [int(r[0]) for r in cur.fetchall()]

        placeholders = " OR ".join(["(axis = ? AND value = ?)"] * len(contexts))
        params: list[str] = []
        for axis, value in contexts:
            params.extend([axis, value])

        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT memory_id
            FROM metadata_sidecar
            WHERE {placeholders}
            """,
            params,
        )
        return [int(r[0]) for r in cur.fetchall()]

    def get_sidecars_for_memory(self, memory_id: int) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM metadata_sidecar WHERE memory_id = ? ORDER BY sidecar_id", (memory_id,))
        return cur.fetchall()

    def get_gematria_for_memory(self, memory_id: int, layer: str | None = None) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        if layer:
            cur.execute(
                "SELECT * FROM gematria_sidecar WHERE memory_id = ? AND layer = ? ORDER BY position",
                (memory_id, layer),
            )
        else:
            cur.execute("SELECT * FROM gematria_sidecar WHERE memory_id = ? ORDER BY layer, position", (memory_id,))
        return cur.fetchall()

    def get_context_frequencies(self) -> dict[tuple[str, str], int]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT axis, value, COUNT(*) as n
            FROM metadata_sidecar
            WHERE axis IS NOT NULL
            GROUP BY axis, value
            """
        )
        freq: dict[tuple[str, str], int] = {}
        for row in cur.fetchall():
            freq[(row["axis"], row["value"])] = int(row["n"])
        return freq

    def get_memory_ids_for_context(self, axis: str, value: str) -> list[int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT memory_id FROM metadata_sidecar WHERE axis = ? AND value = ?",
            (axis, value),
        )
        return [int(r[0]) for r in cur.fetchall()]

    def get_corroboration_counts(self, memory_id: int) -> tuple[int, int]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT kind, COUNT(*) as n
            FROM corroboration
            WHERE memory_id = ?
            GROUP BY kind
            """,
            (memory_id,),
        )
        cor = 0
        uncor = 0
        for row in cur.fetchall():
            if row["kind"] == "corroborated":
                cor = int(row["n"])
            elif row["kind"] == "uncorroborated":
                uncor = int(row["n"])
        return cor, uncor

    def search_text_in_memories(self, memory_ids: Iterable[int], query: str) -> list[sqlite3.Row]:
        ids = list(memory_ids)
        if not ids:
            return []

        tokens = [tok.strip().lower() for tok in query.split() if tok.strip()]
        if not tokens:
            return []

        placeholders = ",".join(["?"] * len(ids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM raw_memory WHERE memory_id IN ({placeholders})", ids)
        rows = cur.fetchall()

        matched = []
        for row in rows:
            text = row["raw_text"].lower()
            overlap = sum(1 for t in tokens if t in text)
            if overlap > 0:
                matched.append(row)
        return matched

    def get_memory_ids_by_entry_bucket(self, bucket: int) -> list[int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT memory_id FROM gematria_sidecar WHERE layer = 'entry' AND bucket = ?",
            (bucket,),
        )
        return [int(r[0]) for r in cur.fetchall()]
