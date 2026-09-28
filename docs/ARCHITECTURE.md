# Architecture — Smart Guided Troubleshooting Engine

## Problem statement (in our own words)

Support agents currently translate a customer's vague complaint ("screen
flickers and battery dies fast") into troubleshooting steps by hand — about
15 minutes per call, at millions-of-calls scale — and the customer still
has to manually hunt through Settings to act on the advice. This service
automates that translation: given a complaint and a pre-fetched knowledge
article, it returns a structured, ordered set of steps, each one linked
directly to the exact Settings screen it refers to (or a manual fallback
when no such screen exists), so the fix is one tap away instead of a
support-article scavenger hunt.

## System diagram

```mermaid
flowchart TD
    Startup["Server Startup / Lifespan"] -->|Pre-warm model| EmbPre["embeddings.prewarm()"]
    
    A["POST /v1/troubleshoot<br/>{id, original_query, siis_response}"] --> B[query_enrichment.py]
    B -->|cleaned query, multi-issue flag| C[relevance.py]
    C -->|"semantic (preferred) or<br/>TF-IDF (fallback) similarity"| D["Goal.score"]
    A --> E[siis_parser.py]
    E -->|"ordered Sections<br/>(heading + grounded steps)"| F[step_structurer.py]
    F -->|optional polish| G["llm_client.py<br/>(Anthropic / Gemini, opt-in)"]
    F --> H[deeplink_matcher.py]
    H -->|"578-entry catalog, 3 tiers:<br/>direct label -> semantic (preferred)<br/>or keyword (fallback)"| I["actionableDeeplink /<br/>validationDeeplink"]
    F --> J[categorizer.py]
    J -->|auto / manual / critical| K[Action]
    D --> L[Goal]
    K --> L
    L --> M["schema.py validation<br/>(ContextDeeplinkResponse)"]
    M --> N["cache.py<br/>(diskcache, falls back to in-memory)"]
    N --> O["HTTP 200<br/>{id, latency_ms, cache_hit, contexts}"]