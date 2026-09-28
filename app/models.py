"""
app/models.py
--------------
Defines the shape of what the API *accepts* (TroubleshootRequest) and
re-exports what it *returns* (ContextDeeplinkResponse, from the organiser-
supplied schema.py, which we do not modify — the brief says to validate
every response against it, so we treat it as a frozen contract).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Re-export the organiser's response schema unchanged. Importing it here
# (rather than duplicating it) guarantees our API can never drift from the
# contract we're graded against.
from app.schema import ContextDeeplinkResponse  # noqa: F401  (re-exported for convenience)


class SiisResponse(BaseModel):
    """The pre-fetched knowledge-store article the caller already looked up."""
    model_config = ConfigDict(strict=True)

    title: str
    content: str


class TroubleshootRequest(BaseModel):
    """
    The payload POST /v1/troubleshoot must accept.
    Shape matches `siis_responses.json` exactly: {id, original_query, siis_response}.

    `strict=True`: Pydantic v2's strict mode disables its usual lenient
    type coercion (e.g. a JSON number silently becoming a string field).
    This is a deliberate trade-off — it rejects a small amount of loosely
    typed input, in exchange for guaranteeing every value that reaches the
    engine is exactly the type the code expects, with no silent
    conversions. If a caller ever needs to send e.g. a numeric `id`, they
    should send it as a JSON string ("id": "123") rather than a bare
    number — this is the one behavioural change from the previous
    (lenient) version worth knowing about.
    """
    model_config = ConfigDict(strict=True)

    id: Optional[str] = Field(default=None, description="Caller-supplied correlation id, echoed back if present")
    original_query: str = Field(..., description="The raw, user-typed complaint")
    siis_response: SiisResponse = Field(..., description="The knowledge-store article to ground the answer in")


class TroubleshootResponseEnvelope(BaseModel):
    """
    Thin wrapper so the HTTP response also carries back the request id and a
    latency figure, which the scope doc calls out ('report step accuracy,
    latency and cost per query'). `contexts` is the actual graded payload.
    """
    id: Optional[str] = None
    latency_ms: float
    cache_hit: bool = False
    contexts: list = Field(default_factory=list)
