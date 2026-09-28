"""
app/main.py
-----------
The REST API entry point. Run it with:

    uvicorn app.main:app --reload --port 8000

Then either open http://127.0.0.1:8000/docs (FastAPI's interactive Swagger
UI, generated automatically from the Pydantic models below) or POST
directly, e.g.:

    curl -X POST http://127.0.0.1:8000/v1/troubleshoot \\
         -H "Content-Type: application/json" \\
         -d @samples/one_request.json
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from app.cache import backend_name as cache_backend_name
from app.deeplink_matcher import active_matching_tier
from app import embeddings
from app.engine import process_with_timing
from app.models import TroubleshootRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("troubleshoot-engine")

app = FastAPI(
    title="Smart Guided Troubleshooting Engine",
    description=(
        "Turns a user complaint + a pre-fetched SIIS knowledge-store article "
        "into a structured, deeplink-backed guided troubleshooting flow."
    ),
    version="1.0.0",
)

# Wide-open CORS so the demo/judging UI (wherever it's hosted) can call this
# API directly from a browser. Tighten `allow_origins` to your real frontend
# domain before any real production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """
    There's no meaningful content to serve at the bare root path — the
    actual API lives at POST /v1/troubleshoot. Rather than let that return
    FastAPI's default `{"detail": "Not Found"}` (confusing the first time
    you open the server's URL in a browser), redirect straight to the
    interactive Swagger docs, which is almost certainly what you wanted.
    """
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict:
    """
    Liveness probe — used by Docker HEALTHCHECK and by judges' smoke tests
    to confirm the container actually came up. Also reports which backend
    is active for the two components that have an optional heavier upgrade
    (semantic matching, persistent caching), so you can immediately tell
    from a running server whether `requirements-embeddings.txt` actually
    loaded successfully, without digging through logs.
    """
    return {
        "status": "ok",
        "cache_backend": cache_backend_name(),
        "deeplink_matching_tier": active_matching_tier(),
        "semantic_relevance_available": embeddings.is_available(),
    }


@app.post("/v1/troubleshoot")
async def troubleshoot(request: TroubleshootRequest) -> dict:
    """
    Main endpoint. Accepts exactly the shape `siis_responses.json` uses per
    row — {id, original_query, siis_response: {title, content}} — and
    returns `{id, latency_ms, cache_hit, contexts: [...]}`, where `contexts`
    is schema-valid per the organiser's `schema.ContextDeeplinkResponse`.
    """
    try:
        result = process_with_timing(
            original_query=request.original_query,
            article_title=request.siis_response.title,
            article_content=request.siis_response.content,
        )
    except ValidationError as exc:
        # The engine validates its own output against schema.py before
        # returning; if that ever fails, surface it as a clear 500 instead
        # of silently shipping a malformed response.
        logger.exception("Engine produced a response that failed schema validation")
        raise HTTPException(status_code=500, detail=f"internal schema validation error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - last-resort guard, logged with full trace
        logger.exception("Unhandled error while processing /v1/troubleshoot")
        raise HTTPException(status_code=500, detail=f"internal error: {exc}") from exc

    return {"id": request.id, **result}
