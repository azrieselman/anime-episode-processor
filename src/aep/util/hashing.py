"""Hashing helpers used for stage cache keys and quick file fingerprints.

We deliberately use BLAKE2b (Python stdlib, fast, 256-bit) for stage keys, and a small
sample-based "fast hash" for very large source files where a full hash would be wasteful.
The cache key contract is: same hash → same output, given same tool versions and params.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def hash_json(obj: Any) -> str:
    """Stable hash of a JSON-serializable structure."""
    return hash_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def fast_file_fingerprint(path: Path, *, sample_bytes: int = 1 << 20) -> str:
    """Fingerprint based on size + first/last MB. Adequate for "did this file change?"
    checks; NOT cryptographically meaningful. We use this for source-file cache keys
    because hashing 12 GB MKVs on every job is silly.
    """
    p = Path(path)
    stat = p.stat()
    h = hashlib.blake2b(digest_size=16)
    h.update(stat.st_size.to_bytes(8, "little"))
    h.update(int(stat.st_mtime_ns).to_bytes(8, "little"))
    with p.open("rb") as f:
        head = f.read(sample_bytes)
        h.update(head)
        if stat.st_size > sample_bytes * 2:
            f.seek(-sample_bytes, 2)
            tail = f.read(sample_bytes)
            h.update(tail)
    return h.hexdigest()
