"""
app/query_enrichment.py
------------------------
Stage 1 of the "two-stage LLM engine" the scope doc describes: normalise a
raw, possibly messy user complaint into a clean technical query before it's
used for relevance scoring / deeplink matching.

This module is intentionally rule-based (fast, deterministic, zero
dependency) rather than an LLM call — query cleanup here is simple enough
(strip list numbering/quotes, detect bundled complaints) that a model call
would only add latency without adding accuracy. If you want an LLM in this
stage too, `app/llm_client.py` is the place to wire it in; `engine.py`
calls this module's `enrich_query()` first regardless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Strips a leading "1. ", "2) ", etc. list marker some queries arrive with
# (see input.txt line 16/17 in the sample kit).
_LEADING_ORDINAL_RE = re.compile(r"^\s*\d+\s*[.)]\s*")

# Splits a single line that bundles several quoted complaints back-to-back,
# e.g.:  1. "Screen is cracked." 2. "Touch doesn't work." 3. "Hard to see."
# Captures each quoted sentence.
_BUNDLED_QUOTES_RE = re.compile(r'\d+\.\s*"([^"]+)"')

# Repeated asterisks are how some queries mask a model name they weren't
# sure of (e.g. "Samsung S***** Ultra") — not useful signal for relevance
# scoring or deeplink matching, so we collapse them to a single placeholder
# token rather than let them pollute the TF-IDF vocabulary.
_MASK_RUN_RE = re.compile(r"\*{2,}")


@dataclass
class EnrichedQuery:
    """Result of Stage 1 cleanup."""
    clean_text: str                 # the query, cleaned, used for scoring/matching
    is_multi_issue: bool = False    # True if the raw text bundled >1 complaint
    sub_issues: List[str] = field(default_factory=list)  # the individual complaints, if bundled


def _strip_noise(text: str) -> str:
    text = _LEADING_ORDINAL_RE.sub("", text.strip())
    text = text.strip().strip('"').strip()
    text = _MASK_RUN_RE.sub("[model]", text)
    return re.sub(r"\s+", " ", text).strip()


def enrich_query(raw_query: str) -> EnrichedQuery:
    """
    Clean a raw query and detect whether it actually bundles multiple
    distinct complaints into one string (only one SIIS article is provided
    per request in this kit's contract, so we can't re-retrieve per
    sub-issue — but flagging it lets the engine be honest in the response
    when a single article is unlikely to cover every symptom mentioned).
    """
    bundled = _BUNDLED_QUOTES_RE.findall(raw_query)
    if len(bundled) >= 2:
        sub_issues = [_strip_noise(q) for q in bundled]
        return EnrichedQuery(
            clean_text=" ".join(sub_issues),
            is_multi_issue=True,
            sub_issues=sub_issues,
        )

    return EnrichedQuery(clean_text=_strip_noise(raw_query), is_multi_issue=False, sub_issues=[])
