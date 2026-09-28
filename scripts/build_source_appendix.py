#!/usr/bin/env python3
"""
scripts/build_source_appendix.py
----------------------------------
Regenerates docs/SOURCE_APPENDIX.md: the FULL text of every hand-written file
in this repository, in one Markdown document.

Why this exists
----------------
BLUEPRINT.md explains what every file is for and how the pieces fit; this
appendix is the other half — the literal contents — so the whole project can
be recreated from documentation alone (copy each fenced block into the path
in its heading). It is generated rather than hand-maintained, so it can never
drift out of date: re-run it after ANY code change.

    python scripts/build_source_appendix.py

Deliberately excluded (not hand-written, or supplied by the organisers):
  * samples/*.json and app/data/deeplinks.json  — organiser-provided data
  * outputs/, __pycache__/, .pytest_cache/, *.pyc, *.zip — generated artifacts
  * docs/SOURCE_APPENDIX.md itself
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "SOURCE_APPENDIX.md"

# Extension -> Markdown code-fence language.
FENCE_LANG = {
    ".py": "python", ".md": "markdown", ".txt": "text", ".yml": "yaml",
    ".yaml": "yaml", ".json": "json", ".toml": "toml",
}
# Files with no extension worth naming explicitly.
FENCE_BY_NAME = {"Dockerfile": "dockerfile", ".gitignore": "gitignore",
                 ".dockerignore": "gitignore", ".env.example": "bash"}

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", "outputs", ".git", ".venv", "venv"}
EXCLUDED_NAMES = {"SOURCE_APPENDIX.md", ".gitkeep"}
EXCLUDED_SUFFIXES = {".pyc", ".zip"}
# Organiser-supplied data, not source we wrote.
EXCLUDED_PATHS = {
    "app/data/deeplinks.json", "samples/deeplinks.json",
    "samples/siis_responses.json", "samples/sample_output.json",
    "samples/input.txt", "samples/BRIEF.md", "samples/schema.py",
}


def _should_include(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if any(part in EXCLUDED_DIRS for part in path.relative_to(REPO_ROOT).parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
        return False
    return rel not in EXCLUDED_PATHS


def _fence_for(path: Path) -> str:
    return FENCE_BY_NAME.get(path.name) or FENCE_LANG.get(path.suffix, "text")


def _sort_key(path: Path):
    """Root files first, then app/, scripts/, tests/, docs/ — a sensible
    reading (and recreation) order."""
    rel = path.relative_to(REPO_ROOT)
    order = {"": 0, "app": 1, "scripts": 2, "tests": 3, "docs": 4}
    top = rel.parts[0] if len(rel.parts) > 1 else ""
    return (order.get(top, 9), rel.as_posix())


def main() -> None:
    files = sorted((p for p in REPO_ROOT.rglob("*") if p.is_file() and _should_include(p)), key=_sort_key)

    lines = [
        "# Source Appendix",
        "",
        "Full text of every hand-written file in this repository. **Generated** by",
        "`scripts/build_source_appendix.py` — do not edit by hand; re-run the script.",
        "",
        "To recreate the project: create each path below and paste its fenced block in.",
        "Organiser-supplied data (`app/data/deeplinks.json`, `samples/*`) is not reproduced;",
        "copy it from the original kit. See `BLUEPRINT.md` for what each file is for.",
        "",
        "## Contents",
        "",
    ]
    for f in files:
        lines.append(f"- `{f.relative_to(REPO_ROOT).as_posix()}`")
    lines.append("")

    for f in files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        text = f.read_text(encoding="utf-8")
        # Use a fence longer than any backtick run inside the file (README/docs
        # contain ``` blocks themselves), so nesting never breaks the Markdown.
        fence = "```"
        while fence in text:
            fence += "`"
        lines += ["---", "", f"## `{rel}`", "", f"{fence}{_fence_for(f)}", text.rstrip("\n"), fence, ""]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    total_lines = sum(chunk.count("\n") + 1 for chunk in lines)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}  ({len(files)} files, {total_lines} lines)")


if __name__ == "__main__":
    main()
