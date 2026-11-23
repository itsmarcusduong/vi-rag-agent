"""Guards against a stale README: release, extras and links must match the repository."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def latest_release() -> str:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.MULTILINE).group(1)


def extras() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    return set(re.findall(r"^([a-z][\w-]*)\s*=", section, flags=re.MULTILINE)) - {"dev"}


def test_status_names_the_latest_release():
    status = next(line for line in README.splitlines() if "**Status" in line)
    assert f"v{latest_release()}" in status, "update the README status line after a release"


def test_every_extra_is_documented():
    missing = {e for e in extras() if f".[{e}]" not in README}
    assert not missing, f"README does not mention extras: {sorted(missing)}"


def test_relative_links_exist():
    links = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", README)
    broken = [t for t in links if "://" not in t and not (ROOT / t).exists()]
    assert not broken, f"broken README links: {broken}"


def test_extras_parser_reads_pyproject():
    assert {"retrieval", "dense", "data", "generation"} <= extras()
