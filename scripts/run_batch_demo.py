#!/usr/bin/env python3
"""
scripts/run_batch_demo.py
---------------------------
Feeds every row of samples/siis_responses.json through the engine directly
(no HTTP server needed) and writes the full results JSON plus a short
console summary. This is the fastest way to (a) sanity-check the whole
pipeline after a change, and (b) generate on-screen output for a demo
video without needing to run curl/Postman against a live server.

Usage:
    python scripts/run_batch_demo.py
    python scripts/run_batch_demo.py --out outputs/my_run.json
    python scripts/run_batch_demo.py --input samples/siis_responses.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `import app.xxx` works when run directly

from app.engine import process_with_timing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(REPO_ROOT / "samples" / "siis_responses.json"),
        help="path to a siis_responses.json-shaped file",
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "outputs" / "batch_results.json"),
        help="where to write the full results JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)["responses"]
    if args.limit:
        rows = rows[: args.limit]

    results = []
    total_latency_ms = 0.0
    total_actions = 0
    total_real_deeplinks = 0
    total_dummy_deeplinks = 0

    print(f"Processing {len(rows)} rows from {args.input} ...\n")
    print(f"{'id':<10} {'latency(ms)':>12} {'score':>7} {'actions':>8} {'real_dl':>8} {'dummy_dl':>9}")
    print("-" * 60)

    for row in rows:
        result = process_with_timing(
            original_query=row["original_query"],
            article_title=row["siis_response"]["title"],
            article_content=row["siis_response"]["content"],
        )
        results.append({"id": row["id"], "original_query": row["original_query"], **result})

        n_actions = sum(len(g["actions"]) for g in result["contexts"])
        n_real = sum(
            1 for g in result["contexts"] for a in g["actions"] for sg in a["stepGroups"]
            if sg["actionableDeeplink"] and sg["actionableDeeplink"]["deeplink"] != "bixby://dummy_positive"
        )
        n_dummy = sum(len(a["stepGroups"]) for g in result["contexts"] for a in g["actions"]) - n_real
        score = result["contexts"][0]["score"] if result["contexts"] else None

        total_latency_ms += result["latency_ms"]
        total_actions += n_actions
        total_real_deeplinks += n_real
        total_dummy_deeplinks += n_dummy

        score_display = f"{score:.3f}" if score is not None else "  n/a"
        print(f"{row['id']:<10} {result['latency_ms']:>12.2f} {score_display:>7} {n_actions:>8} {n_real:>8} {n_dummy:>9}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_dl = total_real_deeplinks + total_dummy_deeplinks
    real_pct = (100 * total_real_deeplinks / total_dl) if total_dl else 0.0
    print("-" * 60)
    print(f"rows processed        : {len(rows)}")
    print(f"avg latency           : {total_latency_ms / len(rows):.2f} ms")
    print(f"total actions         : {total_actions}")
    print(f"real vs dummy deeplink: {total_real_deeplinks} real / {total_dummy_deeplinks} dummy ({real_pct:.1f}% real)")
    print(f"\nfull results written to: {args.out}")


if __name__ == "__main__":
    main()
