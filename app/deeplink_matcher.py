"""
app/deeplink_matcher.py
------------------------
Loads the 578-entry masked deeplink catalog once at import time and exposes
`match_step_group(steps, lead_in)`, which finds the catalog entry whose
Settings screen best matches a StepGroup's instructions. If nothing clears
the confidence bar, it hands back the generic `bixby://dummy_positive`
placeholder with a short, freshly written label — exactly as the brief
specifies.

Matching approach — three tiers, cheapest/most-specific signal first
----------------------------------------------------------------------
1. Direct label match — does the step text literally name the catalog
   entry's real UI toggle label (`validation.key`, e.g. "Touch
   sensitivity")? This is the strongest possible signal: the article and
   the catalog are both describing the same Settings screen by its own
   name. Whole-word-boundary regex, so a short label like "Format"
   doesn't spuriously match inside an unrelated word like "information".
   Unchanged by the semantic upgrade below — an exact UI-label match is a
   stronger signal than any similarity score could give it.

2. Semantic fallback (preferred, when available) — for steps where no
   catalog entry's exact label appears, embed the step text and every
   catalog entry (precomputed once at startup) with a local
   sentence-transformers model, then pick the entry with the highest
   cosine similarity. This is what lets a step saying "screen stays
   black" correctly match a catalog entry described as "display
   unresponsive" — no shared vocabulary, but clearly the same concept.

3. Bag-of-words fallback (automatic, if tier 2 is unavailable) — the
   original Jaccard-token-overlap + `difflib` character-ratio scorer.
   Used only when `app/embeddings.py` reports the semantic model isn't
   available in this environment, so the service still works (just with
   the original, more literal matching behaviour) without it.

See `app/embeddings.py`'s docstring for the fail-open design and its
"Known-untested disclosure" for exactly what has and hasn't been run
end-to-end for tier 2 — the environment this was built in had no way to
install/run `sentence-transformers`.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app import embeddings
from app.config import settings
from app.siis_parser import _starts_with_verb  # reuse the parser's verb test for label building

logger = logging.getLogger("troubleshoot-engine.deeplink_matcher")

# Small stopword list tuned for this domain — deliberately short; we'd
# rather keep a slightly-too-common word than accidentally strip a word
# that actually carries meaning in a Settings-menu label.
_STOPWORDS = {
    "the", "a", "an", "to", "on", "in", "of", "for", "and", "or", "your",
    "you", "via", "settings", "device", "page", "opens", "enables",
    "disables", "updates", "configures", "specified", "value", "screen",
    "option", "is", "this", "that", "with", "will", "can", "please",
}

_WORD_RE = re.compile(r"[a-z0-9']+")

# Words/phrases that hint the step wants something switched ON vs OFF —
# used to break ties between an "Enable X" / "Disable X" catalog pair that
# would otherwise score identically on plain word overlap. Matched with
# \b word boundaries (see _phrase_present) so e.g. "turn on" doesn't
# false-positive-match the "on" inside an unrelated word like "connection".
_ON_HINTS = ("enable", "turn on", "allow", "activate", "switch on")
_OFF_HINTS = ("disable", "turn off", "remove", "deactivate", "switch off")

# Words/phrases that hint the step wants a READ-ONLY check, not a toggle —
# e.g. "Ensure your phone is connected to a stable Wi-Fi or mobile data
# network" contains the literal phrase "mobile data" (a real
# validation.key), but the intent is "verify connectivity", not "turn
# mobile data on/off". Without this, tier 1's direct-label match has no
# way to know an Enable/Disable action is the WRONG kind of action for a
# diagnostic instruction — it would attach whichever of the tied Enable/
# Disable entries happened to come first in the catalog, which for Mobile
# data is "Disable" (an action that would actively break the very
# connection the user is trying to verify). See _polarity_bonus.
_CHECK_HINTS = ("verify", "ensure", "check", "confirm", "make sure", "see if")

# Words/phrases that show a step wants a VALUE changed (a slider, a level, an
# interval) — the only kind of step an "Adjust X" / "Increase X" / "Decrease X"
# catalog entry can correctly serve. Those entries have generic, single-word
# keys ("Media", "Call", "System", "Notifications") and descriptions like
# "Updates the notification vibration to a specified value". Without an
# intent check, ANY step that merely mentions "system" or "call" could match
# one — leaking "...vibration to a specified value" text into unrelated,
# purely navigational steps. See _is_disqualified.
_ADJUST_HINTS = (
    "adjust", "increase", "decrease", "raise", "lower", "reduce", "change",
    "set", "slide", "drag", "turn up", "turn down",
)

# Verbs that prefix a catalog `message`, e.g. "Enable Touch sensitivity" ->
# stripping this prefix leaves the actual feature name, "Touch sensitivity".
_VERB_PREFIX_RE = re.compile(r"^(enable|disable|view|adjust|check|diagnose)\s+", re.IGNORECASE)
# Strips a trailing/inner parenthetical qualifier, e.g.
# "Back up data (Samsung Cloud)" -> "Back up data".
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")

_DUMMY_DEEPLINK = "bixby://dummy_positive"


def _phrase_present(phrase: str, text: str) -> bool:
    """Word-boundary phrase search — see comment on _ON_HINTS/_OFF_HINTS."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def _tokenize(text: str) -> set:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _raw_feature_label(message: str, validation: Optional[dict]) -> str:
    """The short, human name of the exact Settings toggle this entry
    represents. `validation.key` (the literal switch/field label Samsung's
    own QA harness reads back) is preferred; entries with no validation
    payload fall back to the message text with its leading verb stripped."""
    if validation and validation.get("key"):
        return validation["key"]
    return _VERB_PREFIX_RE.sub("", message)


# 535 of the catalog's 578 entries share a near-identical boilerplate
# phrase in their `description` ("... via device Settings on the device.",
# "... settings page in device Settings on the device."). For a sentence-
# embedding model, that shared boilerplate dominates a large share of a
# SHORT string's embedding — compressing genuinely different entries into
# a narrow similarity band and diluting the one thing that actually
# distinguishes them (their feature name and what it does). Real symptom
# this caused: a step about clearing an app's cache/storage matched
# "Enable Allow apps to be pinned" — an unrelated toggle — because both
# texts were dominated by the same generic "Settings/device/app" phrasing.
# `qna_description` is consistently the most content-specific, boilerplate-
# free field in the catalog (e.g. "Pins an app to the screen so others can
# only use that app..." vs. the generic "Enables app pinning via device
# Settings on the device."), so prefer it; only fall back to `description`
# for the handful of entries where `qna_description` is empty.
def _build_searchable_text(feature_label_clean: str, description: str, qna_description: str) -> str:
    distinguishing_content = qna_description.strip() or description.strip()
    return f"{feature_label_clean}. {distinguishing_content}".strip(". ")


@dataclass
class CatalogEntry:
    id: str
    deeplink: str
    description: str
    message: str
    original_type: Optional[str]
    qna_description: str
    validation: Optional[dict]
    search_tokens: set             # precomputed: bag of words for the tier-3 keyword fallback
    feature_label_clean: str       # precomputed: lowercased UI label, no parenthetical, for tier-1 direct match
    feature_label_pattern: "re.Pattern"  # precomputed: compiled \b...\b regex for feature_label_clean
    combined_lower: str            # precomputed: "message description" lowercased, for tier-3 difflib
    searchable_text: str           # precomputed: the text tier-2 actually embeds (see _load_catalog)
    is_on_entry: bool              # precomputed: message.lower() starts with "enable"
    is_off_entry: bool             # precomputed: message.lower() starts with "disable"
    is_view_entry: bool            # precomputed: message.lower() starts with "view" — a read-only screen, not a toggle
    is_adjust_entry: bool          # precomputed: message starts with adjust/increase/decrease — sets a VALUE


def _load_catalog(path: str) -> List[CatalogEntry]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries: List[CatalogEntry] = []
    for row in raw.get("deeplinks", []):
        if row.get("id") == "DL-DUMMY":
            continue  # the generic placeholder is handled specially, not matched into
        message = row.get("message", "")
        description = row.get("description", "")
        qna_description = row.get("qna_description", "")
        validation = row.get("validation")

        searchable = " ".join(filter(None, [
            message, description, qna_description, (validation or {}).get("key", ""),
        ]))

        label_clean = _PARENTHETICAL_RE.sub("", _raw_feature_label(message, validation)).strip().lower()
        # A label under 4 characters is too generic to trust as a standalone
        # signal (see _direct_label_match) — compile a pattern that can
        # never match rather than special-casing "empty label" at query time.
        pattern = re.compile(r"\b" + re.escape(label_clean) + r"\b") if len(label_clean) >= 4 else re.compile(r"(?!)")

        lowered_message = message.lower()
        entries.append(CatalogEntry(
            id=row["id"],
            deeplink=row["deeplink"],
            description=description,
            message=message,
            original_type=row.get("originalType"),
            qna_description=qna_description,
            validation=validation,
            search_tokens=_tokenize(searchable),
            feature_label_clean=label_clean,
            feature_label_pattern=pattern,
            combined_lower=f"{message} {description}".lower(),
            searchable_text=_build_searchable_text(label_clean, description, qna_description),
            is_on_entry=lowered_message.startswith("enable"),
            is_off_entry=lowered_message.startswith("disable"),
            is_view_entry=lowered_message.startswith("view"),
            is_adjust_entry=lowered_message.startswith(("adjust", "increase", "decrease")),
        ))
    return entries


# Module-level singletons — built once per process, reused for every request.
_CATALOG: List[CatalogEntry] = _load_catalog(settings.deeplinks_path)

# Precompute an embedding for every catalog entry ONCE at startup, if the
# semantic model is available (see app/embeddings.py). This is what lets
# tier 2 score "one step vs. all 578 entries" as a single vectorized numpy
# matrix-vector product per query instead of 578 individual model calls.
_CATALOG_EMBEDDING_MATRIX: Optional[np.ndarray] = None
if embeddings.is_available():
    try:
        _CATALOG_EMBEDDING_MATRIX = embeddings.embed_texts([e.searchable_text for e in _CATALOG])
    except Exception:  # noqa: BLE001 - any failure here just disables tier 2, never crashes startup
        logger.warning(
            "Failed to precompute catalog embeddings — the deeplink matcher "
            "will use its keyword-based fallback tier instead.", exc_info=True,
        )
        _CATALOG_EMBEDDING_MATRIX = None

# Precomputed once: True at index i if _CATALOG[i] is an Enable/Disable
# toggle entry. Used to disqualify toggle actions from the vectorized
# semantic search when the step reads as a read-only check/verify
# instruction — see _is_toggle_disqualified_for_check_intent's docstring
# for the real bug (a "verify your connection" step matching "Disable
# Mobile data") this prevents, applied identically across all three tiers.
_TOGGLE_MASK: np.ndarray = np.array([e.is_on_entry or e.is_off_entry for e in _CATALOG])
# Same idea for value-setting ("Adjust/Increase/Decrease X") entries, which are
# disqualified for any step without adjust intent — see _is_disqualified.
_ADJUST_MASK: np.ndarray = np.array([e.is_adjust_entry for e in _CATALOG])


@dataclass(frozen=True)
class _StepIntent:
    """
    What kind of action a step is asking for, computed ONCE per step. (The
    previous version re-ran the same handful of word-boundary regexes inside
    the per-entry loop, i.e. up to 578x per step for identical answers.)
    """
    wants_on: bool
    wants_off: bool
    wants_check: bool
    wants_adjust: bool


def _step_intent(step_text_lower: str) -> _StepIntent:
    return _StepIntent(
        wants_on=any(_phrase_present(h, step_text_lower) for h in _ON_HINTS),
        wants_off=any(_phrase_present(h, step_text_lower) for h in _OFF_HINTS),
        wants_check=any(_phrase_present(h, step_text_lower) for h in _CHECK_HINTS),
        wants_adjust=any(_phrase_present(h, step_text_lower) for h in _ADJUST_HINTS),
    )


def _polarity_bonus(intent: _StepIntent, entry: CatalogEntry) -> float:
    """
    +0.05 when the step's intent agrees with the entry's kind, -0.08 when an
    on/off step meets the opposite toggle (these come in Enable/Disable pairs
    sharing one label, so getting the direction right matters most). Toggle
    entries never reach here for a check-intent step, and adjust entries never
    reach here without adjust intent — both are disqualified upstream.
    """
    if intent.wants_check and entry.is_view_entry:
        return 0.05
    if intent.wants_on and entry.is_on_entry:
        return 0.05
    if intent.wants_off and entry.is_off_entry:
        return 0.05
    if (intent.wants_on and entry.is_off_entry) or (intent.wants_off and entry.is_on_entry):
        return -0.08
    if intent.wants_adjust and entry.is_adjust_entry:
        return 0.05
    if not (intent.wants_on or intent.wants_off) and entry.is_on_entry:
        # Deliberate tie-break. An Enable/Disable pair shares one label, so a
        # step that names the setting but states no direction ("Select Back up
        # data...") scores both identically — and without this, whichever came
        # first in the catalog file won, i.e. arbitrarily and, for some pairs,
        # in the riskier direction (Disable Back up data). Absent a stated
        # direction, prefer Enable: turning something on is the conservative
        # default for a troubleshooting flow.
        return 0.01
    return 0.0


def _is_disqualified(intent: _StepIntent, entry: CatalogEntry) -> bool:
    """
    True if this entry must NEVER be matched to this step, however well the
    words overlap. A matcher-wide safety rule, applied identically in all
    three tiers so no tier can route around it.

    Rule 1 — a check/verify step never matches a TOGGLE. Real bug: "Ensure
    your phone is connected to a stable Wi-Fi or mobile data network"
    contains the literal phrase "mobile data" (a genuine validation.key), and
    the catalog only has Enable/Disable Mobile data (no read-only entry). A
    score penalty is not enough — a direct match starts at 0.85, so even a
    penalised one still clears the threshold when nothing competes — and the
    winner was "Disable Mobile data": an action that would BREAK the very
    connection the user is trying to verify. Disqualify, don't penalise.

    Rule 2 — a step with no adjust intent never matches a VALUE-SETTING
    ("Adjust/Increase/Decrease X") entry. Those have generic single-word keys
    ("Media", "Call", "System") and descriptions like "Updates the ... to a
    specified value"; without this rule, any step merely mentioning "system"
    or "call" could pull one in, leaking that text into unrelated steps.
    """
    if intent.wants_check and (entry.is_on_entry or entry.is_off_entry):
        return True
    if entry.is_adjust_entry and not intent.wants_adjust:
        return True
    return False


def _direct_label_match(step_text_lower: str, entry: CatalogEntry) -> Optional[float]:
    """Tier 1: the article's own instruction names the exact Settings
    toggle by its real UI label. See CatalogEntry.feature_label_* for how
    the pattern was precomputed."""
    if entry.feature_label_pattern.search(step_text_lower):
        coverage = len(entry.feature_label_clean) / max(len(step_text_lower), 1)
        return 0.85 + min(0.10, coverage)
    return None


def _label_overlaps_target(entry: CatalogEntry, deepest_target_lower: Optional[str]) -> bool:
    """
    A direct label match must overlap the step's DEEPEST target — the last
    screen the sentence navigates to — not merely appear somewhere earlier in
    it. Real case: "Go to Settings, tap Lock screen and AOD, and then tap
    Extend Unlock" contains the label "Lock screen" (a real catalog key), but
    the sentence is about Extend Unlock; matching "View Lock screen" attached
    a description about lock-screen NOTIFICATIONS to an Extend Unlock step.
    The catalog has no Extend Unlock entry, so the honest result is the
    dummy placeholder — a parent-screen link with the wrong description is
    worse than none. If no target can be extracted, don't block the match.
    """
    if not deepest_target_lower:
        return True
    label = entry.feature_label_clean
    return label in deepest_target_lower or deepest_target_lower in label


def _tier1_direct_match(
    step_text_lower: str, intent: _StepIntent, deepest_target_lower: Optional[str]
) -> Tuple[Optional[CatalogEntry], float]:
    best_entry: Optional[CatalogEntry] = None
    best_score = 0.0
    for entry in _CATALOG:
        direct = _direct_label_match(step_text_lower, entry)  # cheap regex first; usually None
        if direct is None or _is_disqualified(intent, entry):
            continue
        if not _label_overlaps_target(entry, deepest_target_lower):
            continue
        score = direct + _polarity_bonus(intent, entry)
        if score > best_score:
            best_score, best_entry = score, entry
    return best_entry, best_score


def _tier2_semantic_match(
    step_text: str, intent: _StepIntent, step_tokens: set
) -> Optional[Tuple[CatalogEntry, float]]:
    """
    Vectorized semantic fallback: embed the step once, score it against every
    precomputed catalog embedding in one matrix operation. Returns None
    (meaning "tier 2 unavailable / untrustworthy, try tier 3") rather than
    raising, so an embedding hiccup never breaks a request.
    """
    if _CATALOG_EMBEDDING_MATRIX is None:
        return None
    try:
        step_vec = embeddings.embed_texts([step_text])
        if step_vec is None:
            return None
        similarities = embeddings.cosine_similarity_matrix(step_vec[0], _CATALOG_EMBEDDING_MATRIX)
        # Apply the same disqualification rules as tiers 1 and 3 (see _is_disqualified).
        if intent.wants_check:
            similarities[_TOGGLE_MASK] = -1.0
        if not intent.wants_adjust:
            similarities[_ADJUST_MASK] = -1.0
        best_idx = int(np.argmax(similarities))
        raw_score = float(similarities[best_idx])
        if raw_score < 0.0:
            return None  # every entry was disqualified
        entry = _CATALOG[best_idx]

        # Lexical sanity guard (a lightweight hybrid of semantic + keyword
        # signals). A small general-purpose embedding model will happily rate
        # two unrelated pieces of generic "Settings" phrasing as similar — real
        # cases seen: cache-clearing steps matching "Allow apps to be pinned".
        # Those false positives share NO meaningful vocabulary with the entry.
        # So a semantic match with zero word overlap is only trusted at very
        # high confidence; below that it falls through to tier 3. The catch:
        # this also rejects genuine no-shared-words paraphrases unless they
        # score high — the trade-off is deliberate, and the bypass score is
        # configurable and worth calibrating (scripts/calibrate_semantic_matching.py).
        if (_jaccard(step_tokens, entry.search_tokens) == 0.0
                and raw_score < settings.deeplink_semantic_no_overlap_min_score):
            return None
        return entry, raw_score + _polarity_bonus(intent, entry)
    except Exception:  # noqa: BLE001 - any runtime hiccup falls through to tier 3
        logger.warning("Semantic deeplink match failed for a step — falling back to keyword matching.", exc_info=True)
        return None


# token -> indices of catalog entries containing it. Tier 3 only ever scores
# entries sharing at least one token with the step (an entry with zero overlap
# scores 0 and was skipped anyway), so this replaces "test all 578 entries per
# step" with "test the few dozen that could possibly match". Built once.
_TOKEN_INDEX: Dict[str, List[int]] = {}
for _i, _entry in enumerate(_CATALOG):
    for _tok in _entry.search_tokens:
        _TOKEN_INDEX.setdefault(_tok, []).append(_i)


_TIER3_RERANK_TOP_K = 8


def _tier3_keyword_match(
    step_text_lower: str, step_tokens: set, intent: _StepIntent
) -> Tuple[Optional[CatalogEntry], float]:
    """Original Jaccard + difflib scorer — only reached when tier 2 isn't
    available or wasn't trustworthy. The expensive difflib comparison only
    runs once the cheap token-overlap check has found something in common,
    since most of the 578 entries share zero vocabulary with a given step."""
    # Stage A (cheap): Jaccard for every candidate sharing >= 1 token.
    scored: List[Tuple[float, int]] = []
    for idx in sorted({i for tok in step_tokens for i in _TOKEN_INDEX.get(tok, ())}):  # sorted: deterministic ties
        entry = _CATALOG[idx]
        if _is_disqualified(intent, entry):
            continue
        jaccard = _jaccard(step_tokens, entry.search_tokens)
        if jaccard > 0.0:
            scored.append((jaccard, idx))
    # Stage B (expensive): difflib only on the best few. Profiling showed
    # SequenceMatcher.ratio() was ~90% of cold-path time (~100 calls per step,
    # because common words like "tap"/"notification" make many entries
    # candidates). Re-ranking only the top-K by Jaccard is an APPROXIMATION —
    # an entry just outside the top K could in principle have won on ratio —
    # traded for a large latency cut. Raise _TIER3_RERANK_TOP_K for accuracy.
    scored.sort(key=lambda t: (-t[0], t[1]))
    best_entry: Optional[CatalogEntry] = None
    best_score = 0.0
    for jaccard, idx in scored[:_TIER3_RERANK_TOP_K]:
        entry = _CATALOG[idx]
        ratio = difflib.SequenceMatcher(None, step_text_lower, entry.combined_lower).ratio()
        score = (0.6 * jaccard) + (0.4 * ratio) + _polarity_bonus(intent, entry)
        if score > best_score:
            best_score, best_entry = score, entry
    return best_entry, best_score


def _best_match(step_text: str) -> Tuple[Optional[CatalogEntry], float, str]:
    """
    Runs the three tiers in order and returns (entry, score, tier_name).
    `tier_name` tells the caller which threshold applies (see
    match_step_group) — tier 1 always clears any reasonable threshold on its
    own, but tiers 2 and 3 produce scores on different numeric scales, so each
    is compared against its own configured threshold.
    """
    step_text_lower = step_text.lower()
    intent = _step_intent(step_text_lower)      # computed once, reused by every tier
    step_tokens = _tokenize(step_text)          # likewise

    deepest = _extract_ui_target([step_text])
    entry, score = _tier1_direct_match(step_text_lower, intent, deepest.lower() if deepest else None)
    if entry is not None:
        return entry, score, "direct"

    semantic_result = _tier2_semantic_match(step_text, intent, step_tokens)
    if semantic_result is not None:
        entry, score = semantic_result
        return entry, score, "embeddings"

    entry, score = _tier3_keyword_match(step_text_lower, step_tokens, intent)
    return entry, score, "keyword"


# Matches "tap X", "select X", "open X", "go to X" ... and captures X up to the
# next punctuation. Used to find the concrete on-screen thing a step points at.
_UI_TARGET_RE = re.compile(
    r"\b(?:tap|select|open|choose|touch|click|go to|navigate to)\s+(?:on\s+)?"
    r"(?:the\s+)?(?:switch next to\s+)?(?P<target>[^.,;:]+)",
    re.IGNORECASE,
)
_TARGET_CLAUSE_CUT_RE = re.compile(r"\s+(?:and then|then|and|to|if|when|or|so|until)\b.*$", re.IGNORECASE)
_TARGET_LEADING_RE = re.compile(r"^(?:the|your|a|an)\s+", re.IGNORECASE)
_TARGET_TRAILING_RE = re.compile(r"\s+(?:icon|button|switch|option|tab|toggle)$", re.IGNORECASE)
# A target that names a container or a confirmation rather than a specific
# screen tells the reader nothing ("Settings", "Apps", "OK") — skip it and
# use the previous, more specific one instead.
_GENERIC_TARGETS = {
    "settings", "setting", "apps", "app", "device", "phone", "menu", "screen",
    "home", "ok", "yes", "no", "done", "confirm", "cancel", "next", "back",
}
_TRAILING_FILLER = {"a", "an", "the", "on", "to", "and", "or", "of", "your", "in", "for", "with", "at"}
_INSTRUCTION_CUT_RE = re.compile(r"\s+(?:and then|then|and|or|so that|to)\b.*$|[,;:].*$", re.IGNORECASE)


# Words that describe WHERE on a screen something is, not WHAT it is: "Tap the
# top of the pop-up window" points at no named screen, so "top" must not become
# a label ("Open top settings screen").
_POSITIONAL_WORDS = {"top", "bottom", "left", "right", "center", "centre", "side", "edge", "corner", "middle"}


def _clean_target(raw: str) -> str:
    # A breadcrumb path ("Settings > Security and privacy > Screen lock") names
    # the deepest screen LAST — keep only that segment.
    if ">" in raw:
        raw = raw.split(">")[-1]
    target = _TARGET_CLAUSE_CUT_RE.sub("", raw.strip())
    target = _TARGET_LEADING_RE.sub("", target)
    target = _TARGET_TRAILING_RE.sub("", target)
    words = target.strip(" \"'()\u201c\u201d\u2018\u2019").split()[:3]
    while words and words[-1].lower() in _TRAILING_FILLER:
        words.pop()
    return " ".join(words)


def _extract_ui_target(steps: List[str]) -> Optional[str]:
    """The most specific on-screen thing the group points at — the LAST
    non-generic tap/select/open target in the group (the deepest screen the
    steps navigate to). Taken verbatim from the article's own words."""
    for step in reversed(steps):
        for match in reversed(list(_UI_TARGET_RE.finditer(step))):
            target = _clean_target(match.group("target"))
            if not target or target.lower() in _GENERIC_TARGETS:
                continue
            if target.split()[0].lower() in _POSITIONAL_WORDS:
                continue
            return target
    return None


def _summarize_instruction(sentence: str) -> str:
    """A short imperative phrase from the article's own wording, for manual
    steps that don't point at any Settings screen. Strips stacked lead-in
    connectors ("Now, please ..."), skips an introductory clause so the label
    is the INSTRUCTION not its lead-in ("To find the panel, swipe left ..."
    -> "Swipe left ..."), cuts at the next clause break, keeps <= 8 words."""
    text = sentence.strip()
    for _ in range(3):
        stripped = _LEADING_CONNECTOR_RE_FOR_LABELS.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    head, sep, tail = text.partition(",")
    if sep and not _starts_with_verb(head.strip().lower()) and _starts_with_verb(tail.strip().lower()):
        text = tail.strip()
    text = _INSTRUCTION_CUT_RE.sub("", text).strip(" .")
    words = text.split()[:8]
    while words and words[-1].lower() in _TRAILING_FILLER:
        words.pop()
    if not words:
        return ""
    phrase = " ".join(words)
    return phrase[0].upper() + phrase[1:]


_LEADING_CONNECTOR_RE_FOR_LABELS = re.compile(
    r"^(?:first|next|then|now|alternatively|otherwise|finally|please|also)\b[,:]?\s*", re.IGNORECASE
)


def _make_dummy_labels(steps: List[str], lead_in: Optional[str]) -> Tuple[str, str]:
    """
    Write the (description, message) for the generic placeholder, per the
    catalog's own DL-DUMMY instructions: write it yourself, short, naming the
    concrete screen from the steps. It is derived from THIS group's own
    wording — the deepest screen its steps navigate to, or, for manual steps
    that touch no Settings screen, a short phrase from the instruction itself
    — so two unrelated steps can never share a label.
    """
    target = _extract_ui_target(steps)
    if target:
        return f"Opens the {target} screen in device Settings", f"Open {target} settings screen"

    first = steps[0] if steps else (lead_in or "")
    phrase = _summarize_instruction(first)
    if phrase:
        # No Settings screen applies (e.g. contacting support, using a PC):
        # say what the step is rather than claim it opens a screen.
        return f"Manual step, no Settings screen: {phrase.lower()}", phrase
    return "Opens the relevant device settings screen", "Open relevant device settings"


def _threshold_for_tier(tier: str) -> float:
    if tier == "embeddings":
        return settings.deeplink_match_threshold_embeddings
    # "direct" scores (0.85+) always clear this too — it's really only the
    # deciding threshold for "keyword".
    return settings.deeplink_match_threshold


def match_step_group(steps: List[str], lead_in: Optional[str] = None) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Given the plain-language steps of one StepGroup (plus its optional
    mini-heading lead-in), returns (actionableDeeplink_dict, validationDeeplink_dict).
    Either element of the tuple can be None (schema marks both Optional).

    We match against the SINGLE candidate string that scores highest
    (relative to ITS OWN tier's threshold — see _threshold_for_tier), on
    the theory that a StepGroup's Settings destination is usually named by
    its most specific instruction (e.g. "Tap the switch next to Touch
    sensitivity"), not by combining every step into one noisy blob.
    """
    context_text = " ".join(filter(None, [lead_in, *steps]))
    if not context_text.strip():
        return None, None

    candidates = ([lead_in] if lead_in else []) + list(steps)
    best_entry: Optional[CatalogEntry] = None
    best_score = 0.0
    best_tier = "keyword"
    for candidate in candidates:
        if not candidate:
            continue
        entry, score, tier = _best_match(candidate)
        # Compare against how far ABOVE that tier's own threshold this
        # candidate scored, not the raw score — a raw 0.6 "keyword" score
        # (its threshold is 0.55) shouldn't beat a raw 0.55 "embeddings"
        # score (its threshold is 0.50) just because the number is bigger;
        # what matters is which candidate is the more confident match.
        if entry is None:
            continue
        margin = score - _threshold_for_tier(tier)
        best_margin = best_score - _threshold_for_tier(best_tier) if best_entry is not None else float("-inf")
        if margin > best_margin:
            best_entry, best_score, best_tier = entry, score, tier

    if best_entry is None or best_score < _threshold_for_tier(best_tier):
        description, message = _make_dummy_labels(steps, lead_in)
        actionable = {
            "deeplink": _DUMMY_DEEPLINK,
            "description": description,
            "message": message,
            "originalType": "placeholder",
        }
        return actionable, None  # no catalog validation entry exists for a placeholder

    actionable = {
        "deeplink": best_entry.deeplink,
        "description": best_entry.description,
        "message": best_entry.message,
        "originalType": best_entry.original_type,
    }
    validation = None
    if best_entry.validation and best_entry.validation.get("deeplink"):
        v = best_entry.validation
        validation = {
            "deeplink": v["deeplink"],
            "key": v.get("key", best_entry.message),
            "resultType": v.get("resultType"),
            "condition": v.get("condition"),
            "value": v.get("value"),
        }
    return actionable, validation


def is_real_match(actionable: Optional[dict]) -> bool:
    """True if `actionable` points at a genuine catalog entry (not the
    dummy placeholder) — used by categorizer.py to help decide auto vs manual."""
    return bool(actionable) and actionable.get("deeplink") != _DUMMY_DEEPLINK


def active_matching_tier() -> str:
    """Which tier-2 backend is actually active — surfaced on /health for
    visibility into whether the semantic upgrade is really loaded."""
    return "embeddings" if _CATALOG_EMBEDDING_MATRIX is not None else "keyword-fallback"
