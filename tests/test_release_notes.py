import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "release_notes", Path(__file__).parents[1] / "scripts" / "release_notes.py"
)
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.1] - 2026-11-18

### Added
- Release workflow.

## [0.2.0] - 2026-11-18

### Added
- BM25.

## [0.10.0] - 2026-12-01
- Not to be confused with 0.1.0.
"""


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("chore(release): v0.2.1", "0.2.1"),
        ("chore(release): v0.2.1 (#7)", "0.2.1"),
        ("chore(release): v1.10.3\n\nbody", "1.10.3"),
        ("chore: v0.2.1", None),
        ("feat(release): v0.2.1", None),
        ("chore(release): 0.2.1", None),
        ("chore(release): v0.2", None),
        ("chore(release): v0.2.1-rc1", None),
        ("Revert chore(release): v0.2.1", None),
    ],
)
def test_parse_release_title(subject, expected):
    assert release_notes.parse_release_title(subject) == expected


def test_changelog_section_extracts_only_that_version():
    body = release_notes.changelog_section(CHANGELOG, "0.2.1")
    assert body == "### Added\n- Release workflow."
    assert "BM25" not in body


def test_changelog_section_last_section_and_no_prefix_match():
    assert (
        release_notes.changelog_section(CHANGELOG, "0.10.0") == "- Not to be confused with 0.1.0."
    )
    with pytest.raises(ValueError, match="no section"):
        release_notes.changelog_section(CHANGELOG, "0.1.0")


def test_changelog_section_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        release_notes.changelog_section("## [0.3.0]\n\n## [0.2.0]\n- x\n", "0.3.0")


def test_main_writes_notes_and_skips_non_release(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    out = tmp_path / "notes.md"
    assert release_notes.main(["feat: something", "CHANGELOG.md", str(out)]) == 0
    assert not out.exists() and capsys.readouterr().out == ""
    (tmp_path / "docs" / "reports").mkdir(parents=True)
    (tmp_path / "docs" / "reports" / "v0.2.1.md").write_text("report", encoding="utf-8")
    release_notes.main(["chore(release): v0.2.1 (#7)", "CHANGELOG.md", str(out)])
    assert capsys.readouterr().out.strip() == "0.2.1"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("### Added") and "docs/reports/v0.2.1.md" in text
