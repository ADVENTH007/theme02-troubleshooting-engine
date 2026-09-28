"""
tests/test_deeplink_matcher.py
--------------------------------
Run with: pytest tests/test_deeplink_matcher.py -v

These tests lock in the two failure modes that were actually found (and
fixed) while building this matcher against the real catalog:
  1. A step naming an entry's exact UI label should match it directly.
  2. Word-boundary matching must not let a short label match INSIDE an
     unrelated word (e.g. "format" inside "information").
  3. Steps that have no genuine Settings-screen equivalent (visiting a
     service center, using a USB mouse) must fall back to the dummy
     placeholder rather than being force-matched to something plausible-
     looking but wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.deeplink_matcher import is_real_match, match_step_group


def test_direct_label_match_finds_the_real_catalog_entry():
    actionable, validation = match_step_group(
        ["Enable the switch next to Touch sensitivity."]
    )
    assert is_real_match(actionable)
    assert "touch sensitivity" in actionable["message"].lower()
    assert validation is not None
    assert validation["key"].lower() == "touch sensitivity"


def test_word_boundary_prevents_substring_false_positive():
    """
    Regression test: "Visit Samsung Repair Services for more information."
    must NOT match the catalog's "Format" screenshot-format toggle just
    because "format" is a substring of "information".
    """
    actionable, _ = match_step_group(
        ["Visit Samsung Repair Services for more information."]
    )
    if is_real_match(actionable):
        assert "format" not in actionable["message"].lower()


def test_unmatchable_step_falls_back_to_dummy_placeholder():
    actionable, validation = match_step_group(
        ["Contact Samsung Support or visit an authorized Samsung Service Center."]
    )
    assert actionable["deeplink"] == "bixby://dummy_positive"
    assert validation is None
    # per the brief: 5-7 word, freshly written description/message
    assert 3 <= len(actionable["message"].split()) <= 10


def test_dummy_placeholder_has_no_validation_deeplink():
    actionable, validation = match_step_group(["Some completely made-up instruction."])
    assert actionable["deeplink"] == "bixby://dummy_positive"
    assert validation is None


def test_polarity_prefers_matching_on_off_intent():
    actionable_on, _ = match_step_group(["Turn on Adaptive brightness."])
    actionable_off, _ = match_step_group(["Turn off Adaptive brightness."])
    assert is_real_match(actionable_on)
    assert is_real_match(actionable_off)
    assert actionable_on["message"].lower().startswith("enable")
    assert actionable_off["message"].lower().startswith("disable")


def test_check_intent_step_never_matches_a_toggle_action():
    """
    Regression test for a real bug found via the live Swagger UI: the step
    "Ensure your phone is connected to a stable Wi-Fi or mobile data
    network" contains the literal phrase "mobile data" (a genuine
    validation.key), so a naive matcher wires it to "Disable Mobile data"
    — an action that would actively BREAK the very connection the user is
    trying to verify. A read-only/check-intent step must never be matched
    to a setting-changing toggle action, regardless of how well the words
    otherwise overlap.
    """
    actionable, _ = match_step_group([
        "Ensure your phone is connected to a stable Wi-Fi or mobile data network.",
        "Touch and hold the Wi-Fi icon to check your connection status.",
    ])
    if is_real_match(actionable):
        assert "mobile data" not in actionable["message"].lower()
        assert not actionable["message"].lower().startswith(("enable", "disable"))


# --- regression tests for the "leaking / wrong description" bugs -------------

from app.deeplink_matcher import _make_dummy_labels


def test_value_setting_entries_need_adjust_intent():
    """
    "Adjust/Increase/Decrease X" catalog entries have generic single-word keys
    (Media, Call, System) and descriptions like "Updates the ... vibration to a
    specified value". A purely navigational step must never pull one in.
    """
    actionable, _ = match_step_group(["Tap Notifications.", "Select Sound and vibration."])
    if is_real_match(actionable):
        assert not actionable["message"].startswith(("Adjust", "Increase", "Decrease"))


def test_adjust_intent_still_matches_value_setting_entries():
    actionable, _ = match_step_group(["Adjust the screen brightness."])
    assert is_real_match(actionable)
    assert "brightness" in actionable["message"].lower()


def test_ambiguous_direction_prefers_enable_over_disable():
    """No stated direction used to tie, and the catalog-order winner could be
    'Disable Back up data' — an arbitrary, riskier default."""
    actionable, _ = match_step_group(["Select Back up data to secure your personal files."])
    assert is_real_match(actionable)
    assert actionable["message"].lower().startswith("enable")


def test_explicit_direction_still_wins_over_the_enable_default():
    actionable, _ = match_step_group(["Turn off Adaptive brightness."])
    assert actionable["message"].lower().startswith("disable")


def test_direct_match_must_overlap_the_deepest_target():
    """
    "...tap Lock screen and AOD, and then tap Extend Unlock" contains the real
    label "Lock screen", but the sentence is about Extend Unlock. Attaching
    the parent screen's entry (whose description is about lock-screen
    NOTIFICATIONS) was a wrong-description leak.
    """
    actionable, _ = match_step_group(
        ["Go to Settings, tap Lock screen and AOD, and then tap Extend Unlock."]
    )
    if is_real_match(actionable):
        assert "lock screen" not in actionable["message"].lower()


def test_dummy_label_names_the_step_own_target():
    description, message = _make_dummy_labels(["Tap the Menu icon, and then tap Updates."], None)
    assert message == "Open Updates settings screen"
    assert "Updates" in description


def test_dummy_labels_differ_between_unrelated_steps():
    a = _make_dummy_labels(["Tap Storage.", "Tap Clear cache."], None)
    b = _make_dummy_labels(["Tap Display.", "Tap Screen timeout."], None)
    assert a != b


def test_manual_step_label_is_the_instruction_not_its_lead_in():
    _, message = _make_dummy_labels(["To find the panel, swipe left on the Edge handle."], None)
    assert message.lower().startswith("swipe left")
    _, message = _make_dummy_labels(["Now, please connect your phone to the charger."], None)
    assert message.lower().startswith("connect your phone")


def test_positional_words_are_not_used_as_screen_names():
    _, message = _make_dummy_labels(["Tap the top of the pop-up window to use additional options."], None)
    assert "top settings" not in message.lower()
