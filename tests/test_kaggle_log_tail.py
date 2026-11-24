import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "kaggle_log_tail", Path(__file__).parents[1] / "scripts" / "kaggle_log_tail.py"
)
klt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(klt)


def test_json_chunks_are_joined_and_tailed(tmp_path, capsys):
    chunks = [
        {"stream_name": "stdout", "time": 1.0, "data": "a\nb\n"},
        {"stream_name": "stderr", "time": 2.0, "data": "Traceback\nOSError: boom\n"},
    ]
    (tmp_path / "k.log").write_text(json.dumps(chunks), encoding="utf-8")
    assert klt.main(["x", str(tmp_path), "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == ["Traceback", "OSError: boom"]
    assert "a" not in out


def test_plain_text_and_missing_folder(tmp_path, capsys):
    (tmp_path / "k.log").write_text("x\ny\nz", encoding="utf-8")
    klt.main(["x", str(tmp_path), "1"])
    assert capsys.readouterr().out.splitlines()[-1] == "z"
    assert klt.main(["x", str(tmp_path / "missing")]) == 0
    assert "no kernel log" in capsys.readouterr().out


def test_tail_zero_lines():
    assert klt.tail("a\nb", 0) == []
