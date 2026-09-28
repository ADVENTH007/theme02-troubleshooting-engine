"""
tests/test_query_enrichment.py
--------------------------------
Run with: pytest tests/test_query_enrichment.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.query_enrichment import enrich_query


def test_strips_leading_list_numbering_and_quotes():
    result = enrich_query('1. "My Galaxy S24 screen goes completely blank."')
    assert result.clean_text == "My Galaxy S24 screen goes completely blank."
    assert result.is_multi_issue is False


def test_masked_model_name_is_normalized():
    result = enrich_query("My Samsung S***** Ultra screen flashes.")
    assert "*" not in result.clean_text
    assert "[model]" in result.clean_text


def test_detects_bundled_multi_complaint_query():
    raw = (
        '1. "My Galaxy Z Flip 7 screen is cracked again right where it folds." '
        '2. "The touch doesn\'t work on certain parts of the screen." '
        '3. "I can hardly see anything on the display."'
    )
    result = enrich_query(raw)
    assert result.is_multi_issue is True
    assert len(result.sub_issues) == 3
    assert result.sub_issues[0].startswith("My Galaxy Z Flip 7 screen is cracked")


def test_plain_single_complaint_is_not_flagged_multi_issue():
    result = enrich_query("My Galaxy S22 screen turns completely blank or white.")
    assert result.is_multi_issue is False
    assert result.sub_issues == []
