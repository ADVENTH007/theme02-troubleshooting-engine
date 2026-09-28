"""
app/config.py
--------------
Single place that reads environment variables and turns them into typed
Python values. Every other module imports the `settings` object from here
instead of calling `os.environ.get(...)` itself, so there is exactly one
place to look when you need to change a default or add a new knob.
"""

from __future__ import annotations           # lets us use `int | None` style hints on Python 3.10+

import os                                     # standard library: read environment variables
import tempfile                               # default cache location outside the repo
from dataclasses import dataclass, field      # dataclass = a class that is just typed fields + a generated __init__

try:
    # Loads variables from a `.env` file in the current working directory
    # into os.environ, if one exists — WITHOUT this, copying .env.example
    # to .env and filling it in would silently do nothing, since nothing
    # else in this codebase reads .env directly. Optional dependency: if
    # python-dotenv isn't installed for some reason, we skip it rather than
    # crash — env vars set directly in the shell/container still work fine.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_bool(name: str, default: bool) -> bool:
    """Read an env var as a boolean. Accepts '1', 'true', 'yes' (any case) as True."""
    raw = os.environ.get(name)                # raw is None if the variable was never set
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_str(name: str, default: str) -> str:
    """
    Read an env var as a string, treating an EMPTY value the same as an
    unset one. This matters for a `.env` file like:
        CACHE_DIR=
    which sets the variable to "" (present, but empty) rather than leaving
    it unset — a plain `os.environ.get(name, default)` would return that
    empty string instead of falling back to `default`, which is never what
    you want for a path/name field.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


def _get_float(name: str, default: float) -> float:
    """Read an env var as a float, falling back to `default` if unset or unparsable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    """Read an env var as an int, falling back to `default` if unset or unparsable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)   # frozen=True -> instances are immutable once created (safe to import/share)
class Settings:
    # --- LLM integration (optional; the engine works without it, see llm_client.py) ---
    # If set, the engine will call this provider to *polish* rule-based output
    # (better phrasing, better step segmentation for messy articles). If unset,
    # the engine still returns fully valid, schema-conformant JSON using the
    # deterministic rule-based path only.
    llm_provider: str = field(default_factory=lambda: _get_str("LLM_PROVIDER", "none"))
    # e.g. "none" | "anthropic" | "gemini"
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _get_str("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _get_str("GEMINI_MODEL", "gemini-1.5-flash"))
    llm_timeout_s: float = field(default_factory=lambda: _get_float("LLM_TIMEOUT_S", 8.0))

    # --- Semantic matching (sentence-transformers) ---
    # Both relevance.py and deeplink_matcher.py use this local embedding
    # model as their PREFERRED scorer, automatically falling back to the
    # original TF-IDF / difflib keyword methods if `sentence-transformers`
    # isn't installed or the model weights can't be loaded (see
    # app/embeddings.py's module docstring for the full fail-open design).
    embedding_model_name: str = field(
        default_factory=lambda: _get_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    # Minimum cosine similarity (embedding space) for the deeplink matcher's
    # semantic fallback tier to trust a catalog entry over the dummy
    # placeholder. NOTE: this default is a reasonable starting point from
    # general MiniLM usage, but was NOT empirically tuned against this
    # catalog (no network access to install/run the model while building
    # this) — run `python scripts/calibrate_semantic_matching.py` after
    # installing requirements-embeddings.txt and adjust this if real/dummy
    # matches aren't separating cleanly on your machine.
    deeplink_match_threshold_embeddings: float = field(
        default_factory=lambda: _get_float("DEEPLINK_MATCH_THRESHOLD_EMBEDDINGS", 0.50)
    )

    # A semantic match that shares NO meaningful words with the catalog entry
    # is only trusted at or above this cosine similarity (see the lexical
    # guard in app/deeplink_matcher.py's tier 2). Guards against a small
    # embedding model rating two unrelated pieces of generic "Settings"
    # phrasing as similar. Default is an informed estimate, NOT calibrated —
    # scripts/calibrate_semantic_matching.py exists to tune it.
    deeplink_semantic_no_overlap_min_score: float = field(
        default_factory=lambda: _get_float("DEEPLINK_SEMANTIC_NO_OVERLAP_MIN_SCORE", 0.65)
    )

    # --- Deeplink matching thresholds (difflib/jaccard fallback tier) ---
    # Minimum similarity score (0..1) a catalog entry must reach before we
    # trust it as a real match. Below this, we fall back to the dummy
    # placeholder deeplink instead of attaching a wrong/misleading one.
    # The matcher (app/deeplink_matcher.py) produces two very different score
    # bands: ~0.85-0.95 for a direct "the article names this exact Settings
    # toggle" hit, and <0.35 for its weaker bag-of-words fallback. The
    # threshold sits between those bands so only genuine label matches (or
    # unusually strong fallback overlap) are trusted; anything softer than
    # that becomes the dummy placeholder rather than a guess.
    deeplink_match_threshold: float = field(default_factory=lambda: _get_float("DEEPLINK_MATCH_THRESHOLD", 0.55))

    # --- Relevance / retrieval-confidence scoring ---
    # Below this cosine-similarity score between the user's query and the
    # SIIS article, we still answer (never fabricate a refusal) but the
    # returned `score` field is capped low so the caller/UI knows to treat
    # the guidance as a best-effort, not a confident diagnosis. This
    # threshold was tuned for the TF-IDF fallback; if you've installed
    # requirements-embeddings.txt, re-check it with
    # scripts/calibrate_semantic_matching.py — embedding similarity scores
    # don't necessarily sit in the same numeric range as TF-IDF ones.
    weak_retrieval_threshold: float = field(default_factory=lambda: _get_float("WEAK_RETRIEVAL_THRESHOLD", 0.15))

    # --- Fast-path cache (diskcache-backed; falls back to in-memory if
    # `diskcache` isn't installed — see app/cache.py) ---
    cache_enabled: bool = field(default_factory=lambda: _get_bool("CACHE_ENABLED", True))
    cache_ttl_s: int = field(default_factory=lambda: _get_int("CACHE_TTL_S", 3600))   # 1 hour default
    cache_max_entries: int = field(default_factory=lambda: _get_int("CACHE_MAX_ENTRIES", 2048))
    # Defaults to a system temp directory (not inside the repo) so the
    # cache "just works" without writing into your project folder or
    # hitting a permissions surprise in a locked-down environment; override
    # with CACHE_DIR if you want it to persist somewhere specific.
    cache_dir: str = field(
        default_factory=lambda: _get_str(
            "CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "troubleshoot_engine_cache"),
        )
    )
    cache_size_limit_bytes: int = field(
        default_factory=lambda: _get_int("CACHE_SIZE_LIMIT_BYTES", 100_000_000)  # 100 MB
    )

    # --- Misc ---
    deeplinks_path: str = field(
        default_factory=lambda: _get_str(
            "DEEPLINKS_PATH",
            os.path.join(os.path.dirname(__file__), "data", "deeplinks.json"),
        )
    )


# A single shared instance every module imports: `from app.config import settings`
settings = Settings()
