"""Release helpers used by the release workflow.

``parse_release_title`` recognises a release commit (``chore(release): vX.Y.Z``) and returns the
version. ``changelog_section`` returns the body of ``## [X.Y.Z]`` in a Keep a Changelog file.
Both fail loudly: a release must never be published with empty or wrong notes.

CLI: ``python scripts/release_notes.py "<commit subject>" CHANGELOG.md out.md [existing tags...]``
prints the version (or nothing if the commit is not a release), refuses a version that is not
greater than every existing tag, and writes the notes to ``out.md``.
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


def version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = (int(x) for x in version.split("."))
    return major, minor, patch


def check_newer(version: str, existing_tags: list[str]) -> None:
    """Refuse a version that is not strictly greater than every existing ``vX.Y.Z`` tag."""
    existing = [t[1:] for t in existing_tags if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    if existing:
        latest = max(existing, key=version_tuple)
        if version_tuple(version) <= version_tuple(latest):
            raise ValueError(f"version {version} is not greater than the latest tag v{latest}")


def main(argv: list[str]) -> int:
    subject, changelog_path, out_path, *tags = argv
    version = parse_release_title(subject)
    if version is None:
        return 0
    check_newer(version, tags)
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
