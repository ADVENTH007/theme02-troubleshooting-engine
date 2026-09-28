"""
app/relevance.py
-----------------
Computes the `Goal.score` (0..1) field: how well the SIIS article that was
handed to us actually matches the user's query.

The brief explicitly flags this as part of the challenge: "Some SIIS
matches look off-topic (e.g. row_1 query is a blank screen in Gmail, but
the article returned is about email server errors) — handling weak
retrieval is likely part of the challenge." We can't re-run retrieval
(only one article is given per query), but we CAN be honest about how
confident that pairing looks, so a caller/UI knows to treat a low score as
"best effort, verify with the user" rather than a confident diagnosis.

Two-tier scoring
-----------------
1. Semantic (preferred): cosine similarity between sentence-embedding
   vectors of the query and the article. Catches paraphrase/synonym
   matches a keyword method misses entirely — e.g. a query saying "screen
   stays black" against an article saying "display is unresponsive" share
   almost no vocabulary but are obviously the same issue.
2. TF-IDF (automatic fallback): the original implementation, used only if
   `app/embeddings.py` reports the semantic model isn't available in this
   environment. Kept so the service never hard-fails on this — see
   `app/embeddings.py`'s module docstring for the full fail-open design,
   and its "Known-untested disclosure" for exactly what has and hasn't
   been run end-to-end.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity

from app import embeddings

# Only the first this-many characters of the article content are embedded.
# MiniLM-class models have a fairly short effective context window; the
# vast majority of SIIS articles are shorter than this anyway, and
# troubleshooting intent is almost always established in the opening
# paragraph(s) — the later "contact support" boilerplate at the end of a
# long article contributes little to "is this the right article".
_MAX_CONTENT_CHARS_FOR_EMBEDDING = 2000


def _score_relevance_embeddings(query: str, article_title: str, article_content: str) -> float:
    """Raises on any failure — the caller (`score_relevance`) catches this
    and falls back to TF-IDF, so this function is free to assume the happy
    path throughout."""
    reference_text = f"{article_title}. {article_content[:_MAX_CONTENT_CHARS_FOR_EMBEDDING]}"
    vectors = embeddings.embed_texts([query, reference_text])
    if vectors is None:
        raise RuntimeError("embedding model unavailable")
    query_vec, article_vec = vectors[0], vectors[1]
    # Both rows are L2-normalized (see embeddings.embed_texts), so a plain
    # dot product between them already IS their cosine similarity.
    similarity = float(query_vec @ article_vec)
    return max(0.0, min(1.0, similarity))


def _score_relevance_tfidf(query: str, article_title: str, article_content: str) -> float:
    """
    Classic TF-IDF + cosine similarity between the query and the article's
    title+content. No network/model download required — pure scikit-learn
    on the two strings we already have in hand. This is the original
    implementation, kept as the automatic fallback (see module docstring).
    """
    reference_text = f"{article_title} {article_content}".strip()
    if not query.strip() or not reference_text:
        return 0.0

    # TF-IDF needs at least two "documents" to compute meaningful weights;
    # we feed it exactly the two we have: the query and the article.
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([query, reference_text])
    except ValueError:
        # Happens if, after stop-word removal, one side has zero vocabulary
        # (e.g. a query that is only stop words) — treat as "no signal".
        return 0.0

    similarity = _sk_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return max(0.0, min(1.0, float(similarity)))


def score_relevance(query: str, article_title: str, article_content: str) -> float:
    """
    Returns a similarity score in [0, 1]. Higher = the article is judged a
    better match for the query. Tries the semantic embedding scorer first;
    falls back to TF-IDF if the embedding model isn't available (missing
    dependency) or raises for any reason at call time (e.g. a transient
    encoding error) — this function is designed to always return a usable
    number, never to propagate an exception up to the engine.
    """
    query = (query or "").strip()
    article_title = article_title or ""
    article_content = article_content or ""
    if not query or not (article_title.strip() or article_content.strip()):
        return 0.0

    if embeddings.is_available():
        try:
            return _score_relevance_embeddings(query, article_title, article_content)
        except Exception:  # noqa: BLE001 - any runtime hiccup falls through to TF-IDF below
            pass

    return _score_relevance_tfidf(query, article_title, article_content)
