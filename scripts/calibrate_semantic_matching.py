#!/usr/bin/env python3
"""
scripts/calibrate_semantic_matching.py
-----------------------------------------
Run this AFTER installing requirements-embeddings.txt to check whether the
default thresholds in app/config.py (DEEPLINK_MATCH_THRESHOLD_EMBEDDINGS,
WEAK_RETRIEVAL_THRESHOLD) actually separate real matches from non-matches
on your machine.

Why this script exists
------------------------
The semantic-matching upgrade (app/embeddings.py, and its use in
app/relevance.py + app/deeplink_matcher.py) was built in an environment
with no network access to install `sentence-transformers` or download
model weights — so the threshold defaults are informed estimates from
general knowledge of this model family, NOT values empirically tuned
against this specific catalog and these specific sample queries. This is
exactly the kind of tuning the original difflib/TF-IDF thresholds DID get
(see the git history / comments in app/deeplink_matcher.py for that
earlier process) — this script lets you do the same tuning pass for the
semantic scorers, on a machine that can actually load the model.

What it does
-------------
1. Deeplink matching: runs a set of steps that SHOULD match a real
   catalog entry, and a set that SHOULD fall back to the dummy
   placeholder (including the two known "phrase collision" edge cases
   discovered while tuning the original scorer), and prints the semantic
   similarity score for each.
2. Relevance scoring: runs the actual query/article pairs from
   samples/siis_responses.json through both the TF-IDF and (if available)
   embedding scorers side by side, so you can see whether the paraphrase
   cases the embeddings upgrade specifically targets are actually scoring
   higher than they did before.

Read the printed scores, look for a threshold value that cleanly
separates the two groups in each section, and update
DEEPLINK_MATCH_THRESHOLD_EMBEDDINGS / WEAK_RETRIEVAL_THRESHOLD in your
.env accordingly (see .env.example).

Usage:
    pip install -r requirements.txt -r requirements-embeddings.txt
    python scripts/calibrate_semantic_matching.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app import embeddings                                    # noqa: E402
from app.deeplink_matcher import _CATALOG, _CATALOG_EMBEDDING_MATRIX  # noqa: E402
from app.relevance import _score_relevance_tfidf, _score_relevance_embeddings  # noqa: E402

# --- deeplink matching test cases -------------------------------------------
# These mirror the ones used to tune the original difflib scorer (see
# tests/test_deeplink_matcher.py), plus two genuine paraphrase cases the
# semantic upgrade is specifically meant to fix.
SHOULD_MATCH = [
    "Select Back up data to secure your personal files.",
    "Enable the switch next to Touch sensitivity.",
    "Adjust the screen brightness.",
    "Turn on Adaptive brightness.",
    "The screen stays completely black and won't respond.",       # paraphrase target
    "My display isn't showing anything at all.",                  # paraphrase target
]
SHOULD_STAY_DUMMY = [
    "Contact Samsung Support or visit an authorized Samsung Service Center.",
    "Provide device details regarding the peeling film and green display lines to initiate service.",
    "Visit Samsung Repair Services for more information.",         # known "format"-substring edge case
    "Press and hold the Power button or Side button.",             # known phrase-collision edge case
    "Send your device, along with proof of purchase, to an authorized Samsung Service Center.",
]


def calibrate_deeplink_matching() -> None:
    print("=" * 78)
    print("DEEPLINK MATCHING — semantic tier")
    print("=" * 78)
    if _CATALOG_EMBEDDING_MATRIX is None:
        print(
            "Semantic tier is NOT active (embeddings.is_available() is False).\n"
            "Install requirements-embeddings.txt and re-run this script."
        )
        return

    print(f"{'expected':<10} {'score':>7}  step / best-matched entry")
    print("-" * 78)
    for step in SHOULD_MATCH:
        step_vec = embeddings.embed_texts([step])[0]
        sims = embeddings.cosine_similarity_matrix(step_vec, _CATALOG_EMBEDDING_MATRIX)
        idx = int(sims.argmax())
        print(f"{'MATCH':<10} {sims[idx]:>7.3f}  {step!r} -> {_CATALOG[idx].message!r}")
    for step in SHOULD_STAY_DUMMY:
        step_vec = embeddings.embed_texts([step])[0]
        sims = embeddings.cosine_similarity_matrix(step_vec, _CATALOG_EMBEDDING_MATRIX)
        idx = int(sims.argmax())
        print(f"{'DUMMY':<10} {sims[idx]:>7.3f}  {step!r} -> {_CATALOG[idx].message!r}")

    print(
        "\nLook for a threshold that sits ABOVE every 'DUMMY' score and BELOW\n"
        "every 'MATCH' score above. Set that as DEEPLINK_MATCH_THRESHOLD_EMBEDDINGS\n"
        "in your .env (current default: see app/config.py)."
    )


def calibrate_relevance_scoring() -> None:
    print("\n" + "=" * 78)
    print("RELEVANCE SCORING — TF-IDF vs. semantic, on the real 20 sample rows")
    print("=" * 78)

    samples_path = REPO_ROOT / "samples" / "siis_responses.json"
    with open(samples_path, "r", encoding="utf-8") as f:
        rows = json.load(f)["responses"]

    semantic_on = embeddings.is_available()
    if not semantic_on:
        print(
            "Semantic scoring is NOT active (embeddings.is_available() is False).\n"
            "Showing TF-IDF scores only — install requirements-embeddings.txt for a side-by-side."
        )

    header = f"{'id':<10} {'tfidf':>7}" + (f" {'semantic':>9}" if semantic_on else "")
    print(header)
    print("-" * len(header))
    for row in rows:
        query = row["original_query"]
        title = row["siis_response"]["title"]
        content = row["siis_response"]["content"]
        tfidf_score = _score_relevance_tfidf(query, title, content)
        line = f"{row['id']:<10} {tfidf_score:>7.3f}"
        if semantic_on:
            try:
                semantic_score = _score_relevance_embeddings(query, title, content)
                line += f" {semantic_score:>9.3f}"
            except Exception as exc:  # noqa: BLE001 - diagnostic script, show the error inline
                line += f" {'ERROR: ' + str(exc):>9}"
        print(line)

    print(
        "\nRows where the article is a poor match for the query (see README's\n"
        "'weak-retrieval' example, row_8) should score LOW under both methods.\n"
        "If semantic scores cluster much higher/lower overall than TF-IDF's,\n"
        "adjust WEAK_RETRIEVAL_THRESHOLD in your .env accordingly."
    )


if __name__ == "__main__":
    calibrate_deeplink_matching()
    calibrate_relevance_scoring()
