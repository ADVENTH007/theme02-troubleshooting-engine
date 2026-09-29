# Smart Guided Troubleshooting Engine

A REST API that turns a user's raw complaint + a pre-fetched knowledge-store
(SIIS) article into a structured, deeplink-backed guided troubleshooting
flow — validated against the organiser-supplied `schema.py` contract on
every response.

> **Looking for the full picture?** `BLUEPRINT.md` documents everything — history, every file, every algorithm, all
> tooling, config, limitations and an upgrade roadmap — and `docs/SOURCE_APPENDIX.md` contains the full text of every file
> so the project can be recreated from documentation alone.

Built for **Theme 02** of the Samsung PRISM GenAI Hackathon. See `samples/BRIEF.md`
for the original task notes this was built from.

---
DEMO VIDEO
https://drive.google.com/file/d/1zW33K9jar1KeqxphiHLQ4EdJkWwT7QR_/view?usp=sharing
DEMO VIDEO 2nd HALF
https://drive.google.com/file/d/1pHcMfgeNfQGqrj76E_5qRu-cffjS2x2Z/view?usp=sharing

## Quickstart

```bash
# 1. Install core dependencies (fully working service)
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 2. Run the API
python -m uvicorn app.main:app --reload --port 8000
```

```bash
# 3. Try it (in another terminal)
curl -s -X POST http://127.0.0.1:8000/v1/troubleshoot \
     -H "Content-Type: application/json" \
     -d '{
           "id": "row_1",
           "original_query": "My Galaxy S22 screen turns completely blank or white.",
           "siis_response": {
             "title": "Blank or black display on a Samsung phone or tablet",
             "content": "...(paste an article body from samples/siis_responses.json)..."
           }
         }' | python -m json.tool
```


curl -s -X POST "http://127.0.0.1:8000/v1/troubleshoot" -H "Content-Type: application/json" -d @sample_request.json | python -m json.tool


Opening the bare server URL in a browser redirects to
**http://127.0.0.1:8000/docs** — FastAPI's interactive Swagger UI, where
you can paste a request body and hit "Try it out" directly.

### Run without installing anything globally (Docker)

```bash
docker build --build-arg INSTALL_EMBEDDINGS=true -t samsung-troubleshoot-engine .
docker run -p 8000:8000 samsung-troubleshoot-engine
```

### Batch-run all 20 sample queries at once (no server needed)

```bash
python scripts/run_batch_demo.py
# writes outputs/batch_results.json + prints a per-row summary table
```

### Run the test suite

```bash
python -m pytest -v
```

---

## Optional Upgrades & Pre-Warming

The core install gives you a fully working service using TF-IDF/keyword-based matching throughout. Two independent upgrades are layered on top, both designed to **fail open**: if you don't install them (or they fail to load), the service automatically falls back to its original, already-tested behavior rather than crashing.

| Install this | Unlocks | Trade-off |
|---|---|---|
| `pip install -r requirements-embeddings.txt` | **Semantic matching** — `app/relevance.py` and `app/deeplink_matcher.py` switch from TF-IDF/`difflib` keyword matching to a local sentence-embedding model (`all-MiniLM-L6-v2`). Features startup model pre-warming via `embeddings.prewarm()` to eliminate cold-start latency. | Pulls in `torch` (~100s MB) + downloads ~90 MB model weights on first run. |
| `pip install -r requirements-llm.txt` | Native Anthropic/Google SDKs for the *already-optional* LLM refinement hook (`LLM_PROVIDER=anthropic\|gemini`). | Lightweight SDKs. |

Check active backends on a running server anytime via:
```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

---

## Performance Metrics & Benchmarks

The service enforces strict latency constraints across different processing states:

| Execution Path | Target Latency | Benchmark Performance |
|---|---|---|
| **Cached Path (repeat queries)** | `< 300 ms` | `~ 0.02 ms` |
| **Warm Processing Path (Semantic Tier 2)** | `< 200 ms` | `~ 20 - 50 ms` |
| **Fallback Path (Keyword Tier 3)** | `< 100 ms` | `~ 19 - 56 ms` |

---

## Pipeline & Architecture Summary

```
original_query
      │
      ▼
[1] query_enrichment.py   — clean noisy text, detect bundled multi-issue queries
      │
      ▼
[2] relevance.py          — semantic (preferred) or TF-IDF (fallback)
      │                      similarity(query, article) → Goal.score
      ▼
[3] siis_parser.py        — turn raw article markdown into ordered Sections
      │                      of grounded, verbatim imperative steps
      ▼
[4] step_structurer.py    — Section → Action dict (actionName, description,
      │                      category, stepGroups[])
      ▼
[5] deeplink_matcher.py   — 3-tier match against the 578-entry catalog:
      │                      exact UI-label → semantic (preferred) or
      │                      keyword (fallback) → dummy placeholder
      ▼
[6] schema.py validation  — assembled dict checked against frozen Pydantic schema
      │
      ▼
[7] cache.py              — result cached (diskcache / memory) by (query, title)
```

---

## Configuration Reference

Set environment variables in `.env` (loaded automatically via `python-dotenv`):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `none` | `none` \| `anthropic` \| `gemini` |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformers embedding model |
| `DEEPLINK_MATCH_THRESHOLD_EMBEDDINGS` | `0.50` | Min similarity score for semantic match |
| `DEEPLINK_MATCH_THRESHOLD` | `0.55` | Min similarity score for keyword/difflib match |
| `CACHE_ENABLED` / `CACHE_TTL_S` | `true` / `3600` | Fast-path caching controls |

---

## Submission Checklist Compliance

- **Organiser Contract:** Verified against the unmodified `schema.py`.
- **Deeplink Rules:** URIs copied verbatim from catalog; fallback to `bixby://dummy_positive` with concise self-written labels.
- **Zero Hallucination:** Step instructions are strictly grounded in original SIIS text.
- **Git Tagging:** Ready for tag `PRISM_GENAI_HACKATHON_Y2026`.