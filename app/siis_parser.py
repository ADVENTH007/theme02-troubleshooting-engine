"""
app/siis_parser.py
-------------------
Turns the messy, semi-structured `siis_response.content` string into a clean
list of `Section` objects: a heading (candidate `actionName`) plus an
ordered list of `StepBlock`s (candidate `stepGroups[].steps`).

The SIIS text is NOT uniformly formatted (verified by inspecting all 20
sample rows): headings appear at `#`, `##` or `###` levels, sometimes
numbered ("## Step 1: ..."), sometimes not ("## Customize the Edge panel"),
and three of the twenty articles have NO markdown headings at all — just
flowing prose. This parser is deliberately conservative: it never invents
text. Every string that ends up in a `steps` list is copied verbatim
(after whitespace trimming) from the article.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Vocabulary used to decide "is this sentence an instruction (a step) or is
# it just explanatory narrative?". This is a heuristic, not a parser of
# natural language grammar — good enough for troubleshooting-article prose,
# which is written in a very consistent imperative style.
# ---------------------------------------------------------------------------
_IMPERATIVE_VERBS = (
    "tap", "navigate", "go to", "open", "press", "hold", "swipe", "touch",
    "select", "enable", "disable", "turn on", "turn off", "insert", "connect",
    "check", "verify", "ensure", "restart", "reboot", "try", "adjust", "clear",
    "remove", "update", "install", "back up", "backup", "contact", "visit",
    "schedule", "send", "enter", "increase", "decrease", "use", "switch",
    "wait", "charge", "plug", "shine", "provide", "inspect", "locate",
    "confirm", "look", "sign in", "log in", "restore", "reset", "download",
    "scan", "place",
    # Added after a compound-step test showed real instructions being silently
    # dropped because their verb wasn't whitelisted (e.g. "Follow the on-screen
    # prompts"). Destructive verbs (erase / wipe / delete / uninstall) are
    # included ON PURPOSE: a destructive instruction must be CAPTURED so
    # categorizer.py can flag it `critical`, never silently discarded.
    "follow", "choose", "click", "scroll", "drag", "slide", "set", "toggle",
    "activate", "uninstall", "delete", "erase", "wipe", "disconnect", "unplug",
    "allow", "keep", "review", "retry", "sync", "pair", "make sure",
    "force stop", "power off", "power on", "long press", "sign out",
    "log out", "add", "create", "close", "exit", "launch", "start", "stop",
    "unlock", "lock", "move", "copy", "share", "take", "put", "leave",
    "rotate", "reinsert", "re-insert", "reconnect",
)

# Precompiled per-verb "starts with this exact word/phrase" patterns. \b
# after the verb is essential: a naive `"clearing...".startswith("clear")`
# is True, which would wrongly treat the descriptive gerund "Clearing the
# cache and data can resolve temporary glitches." as if it were the
# imperative "Clear the cache." — two very different sentences that happen
# to share a prefix. \b correctly rejects "clearing" (no boundary between
# "clear" and the following "i") while still accepting "clear the cache".
_VERB_PATTERNS = [re.compile(r"^" + re.escape(verb) + r"\b") for verb in _IMPERATIVE_VERBS]
# Leading transition words that introduce an instruction without being its
# verb ("Then tap X", "Additionally, update Y", "Please restart"). This was a
# FIXED tuple of a few words; anything not on it ("Additionally,",
# "Afterwards,", ...) made the whole sentence fail the verb check and get
# silently dropped as "narrative". A pattern covers the family, and the
# comma-clause handling in _looks_like_step covers everything else.
_LEADING_CONNECTOR_RE = re.compile(
    r"^(?:first|next|then|now|alternatively|otherwise|finally|lastly|also|"
    r"additionally|afterwards?|and then|and|but|so|please|if needed|"
    r"if necessary|note)\b[,:]?\s*",
    re.IGNORECASE,
)

# How far _looks_like_step will look past an introductory clause. Only
# clauses that END IN A COMMA are skipped ("For best results, restart...",
# "Once that is done, check..."), never arbitrary words — skipping arbitrary
# leading words would turn narrative like "If you can sign in and access
# your email..." into a false step, because "sign in" appears mid-sentence.
_MAX_INTRO_CLAUSES = 1
_MAX_INTRO_CLAUSE_CHARS = 70

# An independent main clause ("You can customize the panel, download new
# panels, and remove apps...") is narrative, not a lead-in — its commas
# separate list items, not an introductory clause from an instruction. So a
# clause opening with a subject pronoun is never skipped.
_MAIN_CLAUSE_START_RE = re.compile(r"^(?:you|your|it|they|this|that|these|those|we|there|he|she|i)\b")

# Verb-initial but not an actionable step: pointers to other content
# ("check out our guide"). Matching the verb list alone would admit them.
_NOT_A_STEP_RE = re.compile(r"^(?:check out|refer to)\b")

# A step starting with one of these opens a NEW stepGroup rather than being
# appended to the current one: "Alternatively, ..." introduces a different
# way to do the same thing, i.e. a separate procedure (see
# _build_section_from_sentences).
_NEW_GROUP_MARKER_RE = re.compile(r"^(?:alternatively|otherwise)\b", re.IGNORECASE)

# A sentence ending in ':' that itself is short and NOT an instruction is
# treated as a "mini-heading" inside a section (e.g. "To clear the app's
# cache:") — it starts a new StepBlock instead of becoming a step itself.
_MINI_HEADING_RE = re.compile(r":\s*$")

# Matches a markdown heading line: 1-6 leading '#' characters, a space, then
# the heading text. Captured group 1 = the '#' run (its length = level),
# group 2 = the raw heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)

# Strips a leading "Step 3:" / "Step 3." / "3." / "3)" style ordinal prefix
# off a heading, so "## Step 1: Check Email Access" -> "Check Email Access".
_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*(?:step\s*)?\d+\s*[:.\)-]\s*", re.IGNORECASE
)

# A handful of the source SIIS articles have a text-extraction artifact
# where the space after a sentence-ending punctuation mark was dropped,
# e.g. "...split screen mode.Once you've turned on..." (verified in the
# sample data). We repair *only* this exact pattern — inserting a single
# space, never adding/removing/reordering any word — so the sentence
# splitter above can find the real sentence boundary. This is whitespace
# repair, not content generation, so it doesn't violate "don't invent
# steps": every word in the output is still one that was in the article.
_MISSING_SPACE_RE = re.compile(r"([.!?:])(?=[A-Z])")

# The organiser's export wraps every article body in a fixed metadata
# prefix: "<categories> <Title> ( <categories>): <body>". We only want
# <body>. This regex finds the FIRST "): " and treats everything after it
# as the article body, which is safe because the title/category list never
# itself contains "): " in the sample data.
_METADATA_PREFIX_RE = re.compile(r"^.*?\):\s*", re.DOTALL)

# Splits a wall of prose into rough sentences. Not linguistically perfect
# (doesn't special-case "Mr." etc.) but is good enough for troubleshooting
# copy, which favours short, plain sentences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


@dataclass
class StepBlock:
    """One `StepGroup`'s worth of instructions, plus the mini-heading (if
    any) that introduced it — kept around for deeplink-matching context."""
    lead_in: Optional[str]         # e.g. "To clear the app's cache" (mini-heading, may be None)
    steps: List[str] = field(default_factory=list)


@dataclass
class Section:
    """One candidate `Action`: a heading plus one or more StepBlocks."""
    heading: Optional[str]         # cleaned heading text, or None for headless articles
    description_hint: Optional[str]  # first descriptive (non-step) sentence right after the heading, if any
    blocks: List[StepBlock] = field(default_factory=list)

    @property
    def all_steps(self) -> List[str]:
        return [s for block in self.blocks for s in block.steps]


def _strip_metadata_prefix(raw_content: str) -> str:
    """Remove the '<categories> Title ( categories): ' export wrapper."""
    match = _METADATA_PREFIX_RE.match(raw_content)
    return raw_content[match.end():] if match else raw_content


def _repair_missing_spaces(text: str) -> str:
    """Insert the space that a text-extraction bug dropped after sentence
    punctuation (see _MISSING_SPACE_RE above). Whitespace-only repair."""
    return _MISSING_SPACE_RE.sub(r"\1 ", text)


def _clean_heading(raw_heading: str) -> str:
    """'Step 1: Check Email Access on a PC' -> 'Check Email Access on a PC'."""
    return _ORDINAL_PREFIX_RE.sub("", raw_heading).strip().rstrip(":").strip()


def _starts_with_verb(lowered: str) -> bool:
    if _NOT_A_STEP_RE.match(lowered):
        return False
    return any(pattern.match(lowered) for pattern in _VERB_PATTERNS)


def _looks_like_step(sentence: str) -> bool:
    """
    Heuristic: does this sentence read as an instruction to perform?

    Checks the start of the sentence, then — if that fails — peels off up to
    _MAX_INTRO_CLAUSES short comma-terminated introductory clauses and
    re-checks, so "For best results, restart your phone" and "Once that is
    done, check the update screen" both count, without a hand-maintained
    list of every possible lead-in phrase.
    """
    candidate = sentence.strip().lower()
    if not candidate:
        return False
    for _ in range(_MAX_INTRO_CLAUSES + 1):
        # Peel off (possibly stacked) connectors: "Then, please tap ..."
        for _ in range(2):
            stripped = _LEADING_CONNECTOR_RE.sub("", candidate, count=1)
            if stripped == candidate:
                break
            candidate = stripped
        if _starts_with_verb(candidate):
            return True
        if _MAIN_CLAUSE_START_RE.match(candidate):
            return False  # a main clause, not an introductory lead-in
        comma = candidate.find(",")
        if comma == -1 or comma > _MAX_INTRO_CLAUSE_CHARS:
            return False
        candidate = candidate[comma + 1:].strip()
    return False


_SINGLE_WORD_VERBS = {v for v in _IMPERATIVE_VERBS if " " not in v}
_FLATTENED_LIST_MIN_VERBS = 3


def _split_flattened_list(sentence: str) -> List[str]:
    """
    Some source articles flatten a bullet list into one run-on line with no
    punctuation between items ("Drag and drop apps from the Taskbar Drag and
    drop apps from the Edge Panel Select apps from the Recents menu ...").
    A flattened list is recognisable by THREE OR MORE capitalised imperative
    verbs appearing mid-sentence; a normal sentence with a couple of
    capitalised UI labels ("tap either Open in split screen view or Open in
    pop-up view") stays under that bar. Splitting at those boundaries only
    inserts breaks — no word is added, removed or reordered — so, like the
    missing-space repair, it never introduces content that wasn't in the
    article.
    """
    words = sentence.split(" ")
    boundaries = [
        i for i, w in enumerate(words)
        if i > 0 and w[:1].isupper() and w.lower().strip(".,:;") in _SINGLE_WORD_VERBS
    ]
    if len(boundaries) < _FLATTENED_LIST_MIN_VERBS:
        return [sentence]
    parts, start = [], 0
    for boundary in boundaries:
        parts.append(" ".join(words[start:boundary]))
        start = boundary
    parts.append(" ".join(words[start:]))
    return [p.strip() for p in parts if p.strip()]


def _split_into_sentences(paragraph: str) -> List[str]:
    """Split one paragraph (already newline-free) into candidate sentences."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    out: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        sentence = sentence.strip()
        if sentence:
            out.extend(_split_flattened_list(sentence))
    return out


def _lines_and_sentences(body: str) -> List[str]:
    """
    Flatten a section body into an ordered list of candidate sentences.
    SIIS articles mix "one instruction per line" (common) with "several
    sentences crammed into one paragraph line" (also common) — so we split
    on newlines first, then further split any resulting line that still
    contains multiple sentences.
    """
    out: List[str] = []
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        out.extend(_split_into_sentences(line))
    return out


def _build_section_from_sentences(heading: Optional[str], sentences: List[str]) -> Optional[Section]:
    """
    Walk a flat sentence list for one heading and group it into StepBlocks,
    splitting on "mini-heading" sentences (short, colon-terminated, not
    themselves instructions). Returns None if no real steps were found —
    per the brief, we never invent an Action with zero grounded steps.
    """
    section = Section(heading=heading, description_hint=None)
    current_block = StepBlock(lead_in=None)
    seen_first_sentence = False

    for sentence in sentences:
        is_step = _looks_like_step(sentence)
        # A short colon-terminated line introduces a list, so it is a
        # sub-heading even when it ALSO starts with a verb ("Please try the
        # following steps:"). Checking this before the step test matters now
        # that the step classifier is broader — otherwise such lead-ins would
        # be demoted from sub-headings to junk steps.
        is_mini_heading = bool(_MINI_HEADING_RE.search(sentence)) and len(sentence) <= 80

        if is_mini_heading:
            # Flush the current block if it already collected steps, then
            # start a fresh one introduced by this mini-heading.
            if current_block.steps:
                section.blocks.append(current_block)
            current_block = StepBlock(lead_in=sentence.rstrip(":").strip())
            continue

        if is_step:
            # "Alternatively, ..." / "Otherwise, ..." starts a different
            # procedure, so it opens a new stepGroup (each group gets its own
            # deeplink match) instead of being appended to the current one.
            if _NEW_GROUP_MARKER_RE.match(sentence) and current_block.steps:
                section.blocks.append(current_block)
                current_block = StepBlock(lead_in=None)
            current_block.steps.append(sentence)
            continue

        # Descriptive / narrative sentence. The very first one we see (before
        # any step) is a good candidate for the Action's human-readable
        # `description` field, e.g. "Clearing the cache and data ... glitches."
        if not seen_first_sentence and not current_block.steps and section.description_hint is None:
            section.description_hint = sentence
        seen_first_sentence = True

    if current_block.steps:
        section.blocks.append(current_block)

    return section if section.all_steps else None


def parse_siis_content(raw_content: str) -> List[Section]:
    """
    Main entry point. Returns an ordered list of Sections, each of which is
    a grounded candidate for one `Action` in the final response. Sections
    with no extractable steps are dropped (never fabricated).
    """
    body = _strip_metadata_prefix(raw_content)
    body = _repair_missing_spaces(body)

    heading_matches = list(_HEADING_RE.finditer(body))

    sections: List[Section] = []

    if not heading_matches:
        # Headless article (e.g. "Some things to check first"): treat the
        # whole body as one section with no heading of its own — the caller
        # falls back to the SIIS article title for the Action name.
        sentences = _lines_and_sentences(body)
        section = _build_section_from_sentences(heading=None, sentences=sentences)
        if section:
            sections.append(section)
        return sections

    # Walk consecutive heading matches, each one's body running up to the
    # start of the next heading (or end of string for the last one).
    for idx, match in enumerate(heading_matches):
        heading_text = _clean_heading(match.group(2))
        start = match.end()
        end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(body)
        section_body = body[start:end]

        sentences = _lines_and_sentences(section_body)
        section = _build_section_from_sentences(heading=heading_text, sentences=sentences)
        if section:
            sections.append(section)

    return sections
