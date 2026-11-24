"""Print the end of a downloaded Kaggle kernel log, so failures can be read in the Actions log.

Kaggle writes ``<kernel>.log`` as a JSON list of ``{"stream_name", "time", "data"}`` chunks;
plain text is printed as is.
Usage: ``python scripts/kaggle_log_tail.py <dir> [n_lines]``. Always exits 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def log_text(raw: str) -> str:
    try:
        chunks = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(chunks, list):
        return raw
    return "".join(c.get("data", "") for c in chunks if isinstance(c, dict))


def tail(text: str, n: int) -> list[str]:
    return text.splitlines()[-n:] if n > 0 else []


def main(argv: list[str]) -> int:
    folder = Path(argv[1]) if len(argv) > 1 else Path("kout")
    n = int(argv[2]) if len(argv) > 2 else 200
    logs = sorted(folder.rglob("*.log")) if folder.is_dir() else []
    if not logs:
        print(f"no kernel log under {folder}")
    for log in logs:
        print(f"===== last {n} lines of {log} =====")
        print("\n".join(tail(log_text(log.read_text(encoding="utf-8", errors="replace")), n)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
