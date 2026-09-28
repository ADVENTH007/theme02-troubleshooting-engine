"""
tests/test_relevance.py
--------------------------
Run with: pytest tests/test_relevance.py -v

These tests always exercise the TF-IDF fallback path (which needs no
optional dependency). The semantic-path tests are skipped automatically
via `pytest.importorskip` if `sentence-transformers` isn't installed —
run them after installing requirements-embeddings.txt to confirm that
upgrade actually works on your machine (see also
scripts/calibrate_semantic_matching.py for a more thorough manual check).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.relevance import _score_relevance_tfidf, score_relevance


def test_tfidf_scores_a_strong_keyword_match_higher_than_a_weak_one():
    query = "My Galaxy S22 screen turns completely blank or white."
    strong_article_title = "Blank or white display on a Samsung phone"
    strong_article_content = "If your screen goes blank or turns white, try these steps."
    weak_article_title = "Screen mirroring to your TV"
    weak_article_content = "Learn how to mirror your phone's screen to a Smart TV."

    strong_score = _score_relevance_tfidf(query, strong_article_title, strong_article_content)
    weak_score = _score_relevance_tfidf(query, weak_article_title, weak_article_content)
    assert strong_score > weak_score


def test_score_relevance_returns_zero_for_empty_query():
    assert score_relevance("", "Some title", "Some content") == 0.0


def test_score_relevance_returns_zero_for_empty_article():
    assert score_relevance("My screen is black.", "", "") == 0.0


def test_score_relevance_is_bounded_between_zero_and_one():
    score = score_relevance(
        "My Galaxy S22 screen turns completely blank or white.",
        "Blank or black display on a Samsung phone or tablet",
        "If your screen goes blank or black, try these troubleshooting steps.",
    )
    assert 0.0 <= score <= 1.0


# --- semantic-path tests: skipped automatically if the optional --------
# --- dependency (requirements-embeddings.txt) isn't installed ----------
sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="requirements-embeddings.txt not installed — semantic relevance tests skipped",
)


def test_semantic_scoring_handles_a_pure_paraphrase():
    """
    The whole point of the semantic upgrade: a query and an article
    describing the same issue in completely different words should score
    highly, even though they share almost no vocabulary. If this fails,
    something is wrong with the embedding model/config on this machine —
    also run scripts/calibrate_semantic_matching.py for a fuller picture.
    """
    from app.relevance import _score_relevance_embeddings

    query = "My screen stays completely black and I can't see anything."
    article_title = "Display is unresponsive"
    article_content = "If the display fails to show any image, follow these steps."

    score = _score_relevance_embeddings(query, article_title, article_content)
    assert score > 0.4, (
        f"Expected a clear paraphrase to score reasonably high, got {score:.3f}. "
        "Run scripts/calibrate_semantic_matching.py to investigate."
    )
