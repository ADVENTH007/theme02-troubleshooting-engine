"""
app/engine.py
-------------
The single function every entry point (the FastAPI route, the batch demo
script, the test suite) calls: `process(original_query, siis_title,
siis_content)` -> a `ContextDeeplinkResponse`-shaped dict.

Pipeline
--------
1. Stage 1 — `query_enrichment.enrich_query`: clean the raw complaint.
2. Relevance — `relevance.score_relevance`: how well does this article
   actually match the (cleaned) query? Becomes `Goal.score`.
3. Parse — `siis_parser.parse_siis_content`: turn the article body into
   ordered, grounded Sections of imperative steps.
4. Stage 2 — `step_structurer.build_actions`: turn Sections into schema
   Action dicts, matching each StepGroup to a deeplink along the way.
5. Validate — the assembled dict is checked against the organiser's
   `schema.ContextDeeplinkResponse` before it's returned, so a malformed
   response can never leave the service.
6. Cache — steps 1-5 are skipped entirely on a cache hit for the same
   (query, article title) pair.
"""

from __future__ import annotations

import time
from typing import Dict

from app.cache import cache_get, cache_set, make_cache_key
from app.config import settings
from app.query_enrichment import enrich_query
from app.relevance import score_relevance
from app.schema import ContextDeeplinkResponse
from app.siis_parser import parse_siis_content
from app.step_structurer import build_actions


def _build_goal(original_query: str, article_title: str, article_content: str) -> Dict:
    enriched = enrich_query(original_query)

    relevance_score = score_relevance(enriched.clean_text, article_title, article_content)
    # Weak-retrieval handling (see relevance.py docstring): we still answer
    # with whatever the given article offers, but the score is capped so a
    # caller can distinguish "confident diagnosis" from "best effort".
    if relevance_score < settings.weak_retrieval_threshold:
        relevance_score = min(relevance_score, 0.35)

    sections = parse_siis_content(article_content)
    actions = build_actions(
        sections,
        query=enriched.clean_text,
        article_title=article_title,
        use_llm_refine=(settings.llm_provider != "none"),
    )

    goal_text = f"Follow these steps to resolve: {article_title}"
    if enriched.is_multi_issue:
        goal_text += " (note: the reported complaint mentions more than one symptom; " \
                     "this article addresses the closest overall match)"

    return {
        "goal": goal_text,
        "title": article_title,
        "score": round(relevance_score, 4),
        "actions": actions,
    }


def process(original_query: str, article_title: str, article_content: str) -> Dict:
    """
    Returns a plain dict shaped exactly like `schema.ContextDeeplinkResponse`
    (i.e. `{"contexts": [Goal, ...]}`), already validated against that
    schema. Raises `pydantic.ValidationError` if — despite every effort
    above — the assembled data doesn't fit the contract; the FastAPI layer
    turns that into a clear 500 rather than silently returning bad JSON.

    Thin wrapper around `_process_with_cache_flag` for callers (the batch
    script, most tests) that don't care whether the result came from cache.
    """
    result, _cache_hit = _process_with_cache_flag(original_query, article_title, article_content)
    return result


def _process_with_cache_flag(original_query: str, article_title: str, article_content: str) -> "tuple[Dict, bool]":
    """
    Does the actual work, and — unlike `process()` — also tells the caller
    whether the result was served from `cache.py` rather than freshly
    computed. `process_with_timing()` uses this to report `cache_hit`
    honestly in the HTTP response, instead of relying on a caller
    comparing latency numbers (which, for a very cheap query, can round to
    the same value on both the cold and warm path and give a flaky signal).
    """
    cache_key = make_cache_key(original_query, article_title)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached, True

    goal = _build_goal(original_query, article_title, article_content)
    response_dict = {"contexts": [goal] if goal["actions"] else []}

    # Validate against the organiser's frozen schema, then dump it straight
    # back to a plain dict so every caller (HTTP route, batch script, tests)
    # gets the same JSON-serialisable type. Requires Pydantic v2 (see
    # requirements.txt) — `.model_dump()` is the v2 API; the old v1/v2
    # dual-compatibility branch was dropped in favour of pinning v2
    # strictly, since schema.py only uses plain BaseModel/Enum/Optional
    # fields that need no v1-specific behaviour.
    validated = ContextDeeplinkResponse(**response_dict)
    result = validated.model_dump()

    cache_set(cache_key, result)
    return result, False


def process_with_timing(original_query: str, article_title: str, article_content: str) -> Dict:
    """Convenience wrapper that also reports latency and cache status —
    used by main.py and the batch demo script to satisfy the scope doc's
    'report latency' ask."""
    started = time.perf_counter()
    contexts, cache_hit = _process_with_cache_flag(original_query, article_title, article_content)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {"latency_ms": round(latency_ms, 2), "cache_hit": cache_hit, **contexts}
