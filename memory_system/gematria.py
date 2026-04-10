"""Deterministic gematria utilities.

Gematria here is used only as a deterministic sharding/partition axis, not semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


WORD_RE = re.compile(r"[a-zA-Z]+")
SENTENCE_RE = re.compile(r"[^.!?]+")


def simple_english_gematria(text: str) -> int:
    """Compute A1Z26 value for alphabetic chars.

    Non-letters are ignored.
    """

    total = 0
    for ch in text.lower():
        if "a" <= ch <= "z":
            total += ord(ch) - 96
    return total


def partition_bucket(value: int, modulo: int = 17) -> int:
    """Map deterministic value to a bucket."""

    if modulo <= 0:
        raise ValueError("modulo must be positive")
    return value % modulo


@dataclass(frozen=True)
class GematriaLayerValue:
    layer: str  # word|sentence|entry
    position: int
    token: str
    value: int
    bucket: int


def derive_layered_values(text: str, modulo: int = 17) -> List[GematriaLayerValue]:
    """Derive word/sentence/entry gematria layers for sidecar storage.

    This stays deterministic and explicitly non-semantic.
    """

    values: list[GematriaLayerValue] = []

    words = WORD_RE.findall(text)
    for idx, word in enumerate(words):
        v = simple_english_gematria(word)
        values.append(GematriaLayerValue("word", idx, word, v, partition_bucket(v, modulo)))

    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    for idx, sentence in enumerate(sentences):
        v = simple_english_gematria(sentence)
        values.append(GematriaLayerValue("sentence", idx, sentence, v, partition_bucket(v, modulo)))

    entry_value = simple_english_gematria(text)
    values.append(GematriaLayerValue("entry", 0, "<entry>", entry_value, partition_bucket(entry_value, modulo)))
    return values
