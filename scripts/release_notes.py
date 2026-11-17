"""Release helpers used by the release workflow.

``parse_release_title`` recognises a release commit (``chore(release): vX.Y.Z``) and returns the
version. ``changelog_section`` returns the body of ``## [X.Y.Z]`` in a Keep a Changelog file.
Both fail loudly: a release must never be published with empty or wrong notes.

CLI: ``python scripts/release_notes.py "<commit subject>" CHANGELOG.md out.md`` prints the version
(or nothing if the commit is not a release) and writes the notes to ``out.md``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TITLE = re.compile(r"^chore\(release\): v(\d+\.\d+\.\d+)(?:\s|$)")


def parse_release_title(subject: str) -> str | None:
    """Version from a release commit subject, or None if it is not a release commit."""
    match = _TITLE.match(subject.strip())
    return match.group(1) if match else None


def changelog_section(changelog: str, version: str) -> str:
    """Text between ``## [version]`` and the next ``## [`` heading, stripped."""
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", flags=re.M | re.S
    )
    match = pattern.search(changelog)
    if not match:
        raise ValueError(f"CHANGELOG has no section for {version}")
    body = match.group(1).strip()
    if not body:
        raise ValueError(f"CHANGELOG section for {version} is empty")
    return body


def main(argv: list[str]) -> int:
    subject, changelog_path, out_path = argv
    version = parse_release_title(subject)
    if version is None:
        return 0
    notes = changelog_section(Path(changelog_path).read_text(encoding="utf-8"), version)
    report = Path(f"docs/reports/v{version}.md")
    if report.exists():
        url = f"https://github.com/itsmarcusduong/vi-rag-agent/blob/v{version}/{report.as_posix()}"
        notes += f"\n\nMilestone report: [{report.as_posix()}]({url})"
    Path(out_path).write_text(notes + "\n", encoding="utf-8")
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
