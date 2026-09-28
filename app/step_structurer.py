"""
app/step_structurer.py
------------------------
Stage 2 of the engine: takes the `Section`/`StepBlock` objects produced by
`siis_parser.py` and turns each one into a schema-shaped `Action` dict
(actionName, description, category, stepGroups[]) — attaching a real or
placeholder deeplink to every StepGroup along the way.

Everything returned by this module is a plain `dict`, deliberately, so it
can be handed straight to `schema.Action(**action_dict)` for validation
without any further transformation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.categorizer import categorize_action
from app.deeplink_matcher import is_real_match, match_step_group
from app.llm_client import refine_steps
from app.siis_parser import Section


def _default_description(section: Section, article_title: str) -> str:
    """
    Every Action needs a short human-readable `description` (schema
    requires it). Prefer the descriptive sentence the parser already
    captured right after the section heading (it's grounded article text);
    fall back to a generic templated line naming the section/article.
    """
    if section.description_hint:
        return section.description_hint
    label = section.heading or article_title
    return f"Steps to address: {label}."


def build_action(
    section: Section,
    *,
    query: str,
    article_title: str,
    use_llm_refine: bool,
) -> Optional[Dict]:
    """
    Converts one parsed `Section` into one schema `Action` dict, or None if
    the section ends up with zero real steps (should be rare — siis_parser
    already drops empty sections, but a defensive check costs nothing).
    """
    action_name = section.heading or article_title
    description = _default_description(section, article_title)

    step_groups: List[Dict] = []
    any_real_deeplink = False
    all_step_texts: List[str] = []

    for block in section.blocks:
        steps = list(block.steps)
        if use_llm_refine:
            # Optional quality pass — see app/llm_client.py. Falls back to
            # the original rule-based steps on any failure/timeout/absence
            # of an API key, so this is always safe to leave enabled.
            refined = refine_steps(query, article_title, action_name, steps)
            if refined:
                steps = refined
        if not steps:
            continue

        actionable, validation = match_step_group(steps, block.lead_in)
        if is_real_match(actionable):
            any_real_deeplink = True

        step_groups.append({
            "steps": steps,
            "actionableDeeplink": actionable,
            "validationDeeplink": validation,
        })
        all_step_texts.extend(steps)

    if not step_groups:
        return None

    return {
        "actionName": action_name,
        "description": description,
        "stepGroups": step_groups,
        "category": categorize_action(all_step_texts, any_real_deeplink),
    }


def build_actions(
    sections: List[Section],
    *,
    query: str,
    article_title: str,
    use_llm_refine: bool,
) -> List[Dict]:
    """Converts every parsed Section into an Action dict, dropping empties."""
    actions: List[Dict] = []
    for section in sections:
        action = build_action(
            section, query=query, article_title=article_title, use_llm_refine=use_llm_refine
        )
        if action:
            actions.append(action)
    return actions
