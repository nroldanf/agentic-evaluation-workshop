"""A persistent, file-backed cache for LangGraph node-level caching.

LangGraph's built-in `InMemoryCache` only lives for the current process, so a
one-shot CLI run gets no benefit from it. `DiskCache` stores each cached node
result as a file on disk instead, so re-running the graph in a *later* process
with unchanged node inputs (same cache key) returns the stored result and skips
the (slow, local) LLM call entirely.

It implements the same `BaseCache` contract as `InMemoryCache`:
  - one entry per `(namespace, key)`, serialized with the cache's `serde`;
  - optional per-entry TTL (expired entries are dropped lazily on read);
  - atomic writes (temp file + `os.replace`) guarded by a lock, so the four
    nodes of the parallel graph can write concurrently without corruption.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace


class DiskCache(BaseCache):
    """A `BaseCache` that persists entries as files under a directory."""

    def __init__(self, path: str | os.PathLike[str], *, serde: Any | None = None):
        super().__init__(serde=serde)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _ns_prefix(self, ns: Namespace) -> str:
        """Stable, filesystem-safe prefix for a namespace tuple."""
        return hashlib.sha1(repr(tuple(ns)).encode()).hexdigest()[:16]

    def _file(self, ns: Namespace, key: str) -> Path:
        return self.path / f"{self._ns_prefix(ns)}-{key}.cache"

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        with self._lock:
            now = time.time()
            values: dict[FullKey, Any] = {}
            for ns_tuple, key in keys:
                ns = Namespace(ns_tuple)
                f = self._file(ns, key)
                if not f.exists():
                    continue
                try:
                    enc, val, expiry = pickle.loads(f.read_bytes())
                except Exception:
                    # Corrupt/unreadable entry: treat as a miss and drop it.
                    f.unlink(missing_ok=True)
                    continue
                if expiry is not None and now >= expiry:
                    f.unlink(missing_ok=True)
                    continue
                values[(ns, key)] = self.serde.loads_typed((enc, val))
            return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        return self.get(keys)

    def set(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        with self._lock:
            now = time.time()
            for (ns_tuple, key), (value, ttl) in pairs.items():
                ns = Namespace(ns_tuple)
                expiry = now + ttl if ttl is not None else None
                enc, val = self.serde.dumps_typed(value)
                data = pickle.dumps((enc, val, expiry))
                f = self._file(ns, key)
                tmp = f.with_name(f"{f.name}.tmp.{os.getpid()}.{threading.get_ident()}")
                tmp.write_bytes(data)
                os.replace(tmp, f)  # atomic within the same directory

    async def aset(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        self.set(pairs)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        with self._lock:
            if namespaces is None:
                for f in self.path.glob("*.cache"):
                    f.unlink(missing_ok=True)
            else:
                for ns in namespaces:
                    prefix = self._ns_prefix(Namespace(ns))
                    for f in self.path.glob(f"{prefix}-*.cache"):
                        f.unlink(missing_ok=True)

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        self.clear(namespaces)
