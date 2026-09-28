"""
tests/test_siis_parser.py
--------------------------
Run with: pytest tests/test_siis_parser.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from app.siis_parser import parse_siis_content


def test_strips_metadata_prefix_and_finds_headings():
    raw = (
        "Smartphone,Tablet Some Title ( Smartphone,Tablet): # Intro heading\n"
        "Some narrative text.\n"
        "## Step 1: Check Wi-Fi\n"
        "Navigate to Settings.\n"
        "Tap Connections.\n"
    )
    sections = parse_siis_content(raw)
    assert len(sections) == 1
    assert sections[0].heading == "Check Wi-Fi"
    assert sections[0].all_steps == ["Navigate to Settings.", "Tap Connections."]


def test_never_invents_a_section_with_zero_steps():
    raw = (
        "Cat Title ( Cat): ## Just Narrative\n"
        "This section has no imperative instructions at all, just prose "
        "explaining background context for the reader.\n"
    )
    sections = parse_siis_content(raw)
    assert sections == []


def test_headless_article_still_extracts_steps():
    raw = (
        "Cat Title ( Cat): Turn off your phone or tablet. "
        "Insert the ejector tool into the small hole next to the SIM tray. "
        "Shine a flashlight into the SIM slot.\n"
    )
    sections = parse_siis_content(raw)
    assert len(sections) == 1
    assert sections[0].heading is None
    assert len(sections[0].all_steps) == 3


def test_mini_heading_splits_into_separate_stepgroups():
    raw = (
        "Cat Title ( Cat): ## Clear the App's Cache and Data\n"
        "Clearing the cache and data can resolve temporary glitches.\n"
        "To clear the app's cache:\n"
        "Navigate to Settings.\n"
        "Tap Apps.\n"
        "To clear the app's data:\n"
        "Navigate to Settings.\n"
        "Tap Clear data.\n"
    )
    sections = parse_siis_content(raw)
    assert len(sections) == 1
    section = sections[0]
    assert section.description_hint == "Clearing the cache and data can resolve temporary glitches."
    assert len(section.blocks) == 2
    assert section.blocks[0].lead_in == "To clear the app's cache"
    assert section.blocks[1].lead_in == "To clear the app's data"


def test_missing_space_after_period_is_repaired_before_splitting():
    """
    Regression test for a real text-extraction artifact in the source
    data: a dropped space after sentence-ending punctuation used to merge
    two sentences into one run-on string (e.g. "...split screen.Once
    you've..."). After the whitespace repair, the first sentence must end
    cleanly at its own period rather than swallowing the next sentence.
    """
    raw = (
        "Cat Title ( Cat): ## Multi window\n"
        "Tap the switches next to Swipe for split screen.Once you've turned "
        "on these features, a two finger swipe will work.\n"
    )
    sections = parse_siis_content(raw)
    steps = sections[0].all_steps
    assert steps == ["Tap the switches next to Swipe for split screen."]
    assert not any("Once you've" in s for s in steps)


def test_real_sample_row1_shapes_as_expected():
    """Sanity check against one of the actual kit samples (email server article)."""
    content = (
        "Smartphone,Others Mobile,Mobile Accessories,Tablet Email server not "
        "responding on Samsung phone or tablet ( Smartphone,Others Mobile,"
        "Mobile Accessories,Tablet): # Troubleshooting Email Connection Issues "
        "on Your Samsung Phone\nIf you're having trouble accessing your email "
        "on your Samsung phone, here are some steps you can take to resolve "
        "the issue.\n## Step 1: Check Email Access on a PC\nFirst, try "
        "accessing your email on a personal computer. This helps determine "
        "if the problem is with your phone's connection or your email "
        "account itself.\n"
    )
    sections = parse_siis_content(content)
    assert len(sections) == 1
    assert sections[0].heading == "Check Email Access on a PC"
    assert sections[0].all_steps == ["First, try accessing your email on a personal computer."]


# --- regression tests for the "dropped steps" bug (compound / continuation steps) ---

def _steps(body_lines):
    raw = "Cat Title ( Cat): ## Step 3: Update things\n" + "\n".join(body_lines) + "\n"
    sections = parse_siis_content(raw)
    return [s for sec in sections for s in sec.all_steps]


def test_continuation_steps_with_lead_in_words_are_not_dropped():
    """
    Real bug: a sentence only counted as a step if it began with a whitelisted
    verb after a FIXED list of connectors, so "Additionally, update Knox Core",
    "For best results, restart...", "Once that is done, check..." and
    "Follow the on-screen prompts" were silently discarded as narrative.
    """
    steps = _steps([
        "Open the Galaxy Store app.",
        "Additionally, update Knox Core from the same list.",
        "For best results, restart your phone once both updates finish.",
        "Once that is done, check the software update screen again.",
        "Follow the on-screen prompts to finish.",
    ])
    assert len(steps) == 5
    assert any(s.startswith("Additionally, update Knox Core") for s in steps)


def test_narrative_sentences_are_still_not_steps():
    """The loosened classifier must not admit descriptive prose."""
    steps = _steps([
        "Tap Done.",
        "This helps determine if the problem is with your connection.",
        "If you can sign in and access your email on a PC, your phone might not be connected.",
        "Clearing the cache and data can resolve temporary glitches.",
        "You can customize the panel, download new panels, and remove apps if needed.",
        "Certain apps, like Netflix or YouTube, allow you to cast videos to a big screen.",
        "If you need help, check out our guide.",
    ])
    assert steps == ["Tap Done."]


def test_escalation_and_destructive_steps_are_captured():
    """Manual-escalation and destructive steps drive the manual/critical
    categories — silently dropping them made those categories unreachable."""
    steps = _steps([
        "In this case, please contact the Samsung Support Center for further assistance.",
        "Once you've backed up your data, please visit a Samsung walk-in service center.",
        "Afterward, tap Delete all.",
    ])
    assert len(steps) == 3


def test_alternatively_starts_a_new_stepgroup():
    raw = (
        "Cat Title ( Cat): ## Step 2: Check Wi-Fi\n"
        "Swipe down to open Quick settings.\n"
        "Touch and hold the Wi-Fi icon.\n"
        "Alternatively, go to Settings, tap Connections, and then tap Wi-Fi.\n"
    )
    section = parse_siis_content(raw)[0]
    assert len(section.blocks) == 2
    assert section.blocks[1].steps[0].startswith("Alternatively, go to Settings")


def test_colon_terminated_lead_in_stays_a_subheading_not_a_step():
    raw = (
        "Cat Title ( Cat): ## Step 1: Fix flicker\n"
        "To resolve this issue, please try the following steps:\n"
        "Tap Settings.\n"
    )
    section = parse_siis_content(raw)[0]
    assert section.all_steps == ["Tap Settings."]
    assert section.blocks[0].lead_in == "To resolve this issue, please try the following steps"


def test_flattened_bullet_list_is_split_into_separate_steps():
    steps = _steps([
        "Drag and drop apps from the Taskbar Drag and drop apps from the Edge Panel "
        "Select apps from the Recents menu Drag and drop a URL or hyperlink",
    ])
    assert steps == [
        "Drag and drop apps from the Taskbar",
        "Drag and drop apps from the Edge Panel",
        "Select apps from the Recents menu",
        "Drag and drop a URL or hyperlink",
    ]
