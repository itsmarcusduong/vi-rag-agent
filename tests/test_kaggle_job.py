import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "kaggle_job", Path(__file__).parents[1] / "scripts" / "kaggle_job.py"
)
kaggle_job = importlib.util.module_from_spec(_spec)
sys.modules["kaggle_job"] = kaggle_job  # dataclasses look the module up by name
_spec.loader.exec_module(kaggle_job)

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_build_writes_valid_script_and_metadata(tmp_path):
    out = kaggle_job.build("m2_bm25", SHA, "someone", tmp_path / "k")
    code = (out / "run.py").read_text(encoding="utf-8")
    ast.parse(code)  # generated kernel must be valid Python
    assert repr(SHA) in code and "scripts/run_bm25.py" in code
    assert "/tmp/repo" in code  # repository is cloned outside /kaggle/working (not in outputs)
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/vi-rag-agent-m2-bm25"
    assert meta["title"] == "vi-rag-agent-m2-bm25"
    assert meta["is_private"] == "true" and meta["enable_internet"] == "true"
    assert meta["enable_gpu"] == "false"
    assert meta["code_file"] == "run.py" and meta["kernel_type"] == "script"


def test_every_registered_job_points_to_existing_files():
    root = Path(__file__).parents[1]
    for name, job in kaggle_job.JOBS.items():
        assert (root / job.script).is_file(), name
        assert all(o.startswith("results/") and o.endswith(".json") for o in job.outputs)


@pytest.mark.parametrize("sha", ["abc123", SHA.upper(), SHA[:-1] + "g"])
def test_build_rejects_non_full_sha(tmp_path, sha):
    with pytest.raises(ValueError):
        kaggle_job.build("m2_bm25", sha, "u", tmp_path)


def test_unknown_job_raises():
    with pytest.raises(KeyError, match="unknown job"):
        kaggle_job.get_job("nope")


def test_info_prints_kernel_gpu_and_outputs(capsys):
    assert kaggle_job.main(["info", "--job", "m2_bm25", "--user", "u"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["kernel=u/vi-rag-agent-m2-bm25", "gpu=false", "outputs=results/m2_bm25.json"]


def test_every_experiment_script_is_registered():
    # Guards against adding scripts/run_*.py and forgetting the registry entry (happened for M4).
    root = Path(__file__).parents[1]
    registered = {job.script for job in kaggle_job.JOBS.values()}
    gpu_scripts = {
        "scripts/run_dense.py",
        "scripts/run_ablation_m3.py",
        "scripts/run_rerank.py",
        "scripts/run_generation.py",
    }
    missing = {s for s in gpu_scripts if (root / s).exists()} - registered
    assert not missing, f"not registered as Kaggle jobs: {sorted(missing)}"


def test_outputs_are_copied_even_when_the_script_fails(tmp_path):
    # M5 lost its stage A/B results because the copy ran only after a successful script.
    code = (kaggle_job.build("m5_generation", SHA, "u", tmp_path / "k") / "run.py").read_text(
        "utf-8"
    )
    tries = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Try)]
    assert any(
        "SCRIPT" in ast.unparse(t.body) and "shutil.copy" in ast.unparse(t.finalbody) for t in tries
    )
