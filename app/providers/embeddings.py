from __future__ import annotations

import re
import hashlib
import math


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize_text(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(value)]


def hashed_text_embedding(value: str, *, dimensions: int = 256) -> list[float]:
    """Build a deterministic local vector from words and Chinese n-grams.

    This is a dependency-free fallback for fuzzy lexical recall. A future
    semantic embedder can rebuild the side index without changing memory rows.
    """
    normalized = " ".join(str(value or "").lower().split())
    features = tokenize_text(normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for run in chinese_runs:
        features.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        features.extend(run[index : index + 3] for index in range(max(0, len(run) - 2)))
    vector = [0.0] * max(8, int(dimensions))
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % len(vector)
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector
