"""
app/cache.py
------------
Fast-path cache for repeated (query, article) pairs — satisfies the scope
doc's "fast-path cache that serves pre-validated answers in under 300 ms".

Backed by `diskcache` (a pure-Python, SQLite-backed cache — no separate
server process needed) so cached results now genuinely persist across
process restarts. This matters more than it might sound: with the
original plain in-memory dict, `uvicorn --reload` — which restarts the
whole Python process on every code change during development — silently
wiped the cache each time, so a "warm cache" latency measurement taken
during development was often secretly measuring a cold miss that just
happened to be fast for other reasons.

Falls back to the original in-memory `OrderedDict` implementation if
`diskcache` isn't installed (or fails to initialize for any reason — e.g.
a read-only filesystem), so the service still works — just without
cross-restart persistence — in a stripped-down environment. Every public
function below (`cache_get`, `cache_set`, `cache_clear`) behaves
identically regardless of which backend ends up active; call
`backend_name()` if you need to know which one you got (surfaced on the
`/health` endpoint for visibility).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("troubleshoot-engine.cache")


def make_cache_key(original_query: str, article_title: str) -> str:
    """
    The cache key is a hash of (query, article title) rather than the raw
    strings themselves, so we don't hold arbitrarily long user text as a
    dict/database key. The article title is included (not the full
    content) because in this kit the same query text should always arrive
    paired with the same article, but keying on both is defensive and cheap.
    """
    raw = f"{original_query.strip().lower()}||{article_title.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --- diskcache-backed implementation (preferred) ----------------------------
_disk_cache = None
_backend = "memory"  # overwritten below if diskcache initializes successfully

try:
    import diskcache

    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
    _disk_cache = diskcache.Cache(settings.cache_dir, size_limit=settings.cache_size_limit_bytes)
    _backend = "diskcache"
    logger.info("Cache backend: diskcache at %s", settings.cache_dir)
except Exception:  # noqa: BLE001 - missing package, permissions, corrupt db — any of these degrade gracefully
    logger.warning(
        "diskcache unavailable (not installed, or couldn't open %s) — "
        "falling back to an in-memory cache that won't survive a process "
        "restart. Install requirements.txt's diskcache entry to fix this.",
        settings.cache_dir, exc_info=True,
    )
    _disk_cache = None
    _backend = "memory"

# --- in-memory fallback (only used if diskcache is unavailable) -------------
# OrderedDict lets us evict the least-recently-inserted entry in O(1) when
# the cache grows past its cap, without pulling in an extra dependency.
_memory_store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()


def _memory_get(key: str) -> Optional[Any]:
    entry = _memory_store.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() > expires_at:
        _memory_store.pop(key, None)  # lazily evict expired entries on read
        return None
    return value


def _memory_set(key: str, value: Any) -> None:
    if key in _memory_store:
        _memory_store.pop(key)  # re-insert to refresh its position (simple LRU-ish behaviour)
    _memory_store[key] = (time.monotonic() + settings.cache_ttl_s, value)
    while len(_memory_store) > settings.cache_max_entries:
        _memory_store.popitem(last=False)  # evict the oldest entry


# --- public API: identical regardless of which backend is active -----------
def cache_get(key: str) -> Optional[Any]:
    if not settings.cache_enabled:
        return None
    if _backend == "diskcache":
        try:
            return _disk_cache.get(key, default=None)
        except Exception:  # noqa: BLE001 - a corrupted cache entry/db shouldn't break a request
            logger.warning("diskcache read failed — treating as a cache miss.", exc_info=True)
            return None
    return _memory_get(key)


def cache_set(key: str, value: Any) -> None:
    if not settings.cache_enabled:
        return
    if _backend == "diskcache":
        try:
            _disk_cache.set(key, value, expire=settings.cache_ttl_s)
            return
        except Exception:  # noqa: BLE001 - a failed write shouldn't break the response that triggered it
            logger.warning("diskcache write failed — this result won't be cached.", exc_info=True)
            return
    _memory_set(key, value)


def cache_clear() -> None:
    """Exposed for tests — wipes the cache between test cases."""
    if _backend == "diskcache":
        _disk_cache.clear()
    else:
        _memory_store.clear()


def backend_name() -> str:
    """Which backend is actually active — surfaced on /health for visibility."""
    return _backend
