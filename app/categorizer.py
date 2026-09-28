"""
app/categorizer.py
-------------------
Decides which `actionCategory` (auto / manual / critical) an Action gets.
This mirrors the reasoning implied by sample_output.json: "Back Up Phone
Data" (a safe, reversible Settings toggle we found a real deeplink for) is
`auto`; "Schedule Screen Repair Service" (no deeplink exists for visiting a
physical service center) is `manual`.
"""

from __future__ import annotations

import re
from typing import List

# Any of these words appearing in the step text mark the action as
# irreversible / high-risk, regardless of whether a deeplink was found —
# these should never be silently auto-executed.
_CRITICAL_KEYWORDS = (
    "factory reset", "erase all data", "erase everything", "wipe",
    "delete all", "format the device", "reset network settings",
    "remove your google account", "unregister",
)

# Phrases that signal "this step happens outside the phone's Settings app"
# (visiting a person, mailing a device, using external hardware) — these can
# never be automated via a deeplink, so they default to `manual` even if a
# generic placeholder deeplink was attached.
_MANUAL_ONLY_KEYWORDS = (
    "service center", "service centre", "walk-in", "mail-in",
    "contact samsung support", "samsung premium care", "usb mouse",
    "usb adapter", "send your device", "ship your device", "warranty",
)


def _contains_keyword(text: str, keyword: str) -> bool:
    """
    Word-boundary substring check. Plain `keyword in text` would let, e.g.,
    the critical keyword "wipe" false-positive-match inside the ordinary
    word "swipe" — \b anchors prevent that while still matching multi-word
    phrases like "factory reset" correctly.
    """
    return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None


def categorize_action(step_texts: List[str], has_real_deeplink: bool) -> str:
    """
    Parameters
    ----------
    step_texts: every step string across all stepGroups of this Action
                (used to scan for risk/manual-only keywords).
    has_real_deeplink: True if at least one stepGroup in this Action got a
                       genuine catalog deeplink match (not the dummy
                       placeholder) — a proxy for "this can be done
                       in-Settings, one tap".

    Returns
    -------
    One of "critical", "manual", "auto" (matches schema.actionCategory).
    """
    combined = " ".join(step_texts).lower()

    if any(_contains_keyword(combined, keyword) for keyword in _CRITICAL_KEYWORDS):
        return "critical"

    if any(_contains_keyword(combined, keyword) for keyword in _MANUAL_ONLY_KEYWORDS):
        return "manual"

    return "auto" if has_real_deeplink else "manual"
