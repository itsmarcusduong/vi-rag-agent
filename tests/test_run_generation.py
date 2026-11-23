"""End-to-end test of scripts/run_generation.py on synthetic data with fake models (no GPU)."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from vi_rag_agent.data import build_benchmark

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("run_generation", SCRIPTS / "run_generation.py")
rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rg)

CITIES = [f"thành{i}" for i in range(24)]
DISHES = [f"món{i}" for i in range(24)]


def records(split, idx, offset):
    out = []
    for j, i in enumerate(idx):
        ctx = f"Bài {i} kể rằng thành phố {CITIES[i]} nổi tiếng với {DISHES[i]} ngon."
        for r in range(2):  # two questions per passage, so clusters have size > 1
            out.append(
                {
                    "id": str(offset + 2 * j + r),
                    "title": f"T{i}",
                    "context": ctx,
                    "question": f"Thành phố {CITIES[i]} nổi tiếng với món gì (câu {r})?",
                    "is_impossible": False,
                    "answers": {"text": [DISHES[i]]},
                }
            )
    return out


def fake_bench():
    return build_benchmark(
        {
            "train": records("train", range(0, 8), 0),
            "validation": records("validation", range(8, 16), 100),
            # passages 4..7 are shared with train, 16..23 are unseen
            "test": records("test", list(range(4, 8)) + list(range(16, 24)), 200),
        }
    )


def fake_encode_all(spec, passages, queries):
    def vec(text):
        rng = np.random.default_rng(int(re.findall(r"thành(\d+)", text)[0]))
        v = rng.normal(size=16)
        return v / np.linalg.norm(v)

    p = np.stack([vec(t) for t in passages])
    q = {n: np.stack([vec(t) for t in ts]) for n, ts in queries.items()}
    return p, q, {"embedding_dim": 16}


class FakeGenerator:
    """Answers from the first passage (right iff it is the gold passage); closed-book is wrong."""

    revision = "fake"
    load_s = 0.0

    def __init__(self, name, fail_on_generate=False):
        self.name, self.fail = name, fail_on_generate

    def generate(self, batch):
        if self.fail:
            raise RuntimeError("CUDA out of memory (fake)")
        outs = []
        for messages in batch:
            m = re.search(r"\[1\][^\n]*?(món\d+)", messages[-1]["content"])
            outs.append(f"Trả lời: {m.group(1)}" if m else "không rõ")
        return outs, [3] * len(batch)

    def close(self):
        pass


def make_generator(name):
    if name == rg.LLMS["L3"]:
        raise OSError("cannot load (fake)")
    return FakeGenerator(name, fail_on_generate=name == rg.LLMS["L2"])


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "load_viquad", fake_bench)
    monkeypatch.setattr(rg, "encode_all", fake_encode_all)
    monkeypatch.setattr(rg, "make_generator", make_generator)
    monkeypatch.setattr(rg, "device", lambda: "cpu")
    monkeypatch.setattr(rg, "gpu_name", lambda: None)
    monkeypatch.setattr(rg, "git_commit", lambda: "test")
    monkeypatch.setattr(rg, "OUT", tmp_path / "out.json")
    ref = tmp_path / "ref.json"
    monkeypatch.setattr(rg, "REFERENCE", ref)

    def _run(target):
        held = {"validation/answerable": {"selected": {"metrics": {"mrr@10": target}}}}
        ref.write_text(json.dumps({"a2_held_out": held}), encoding="utf-8")
        code = rg.main()
        return code, json.loads(rg.OUT.read_text(encoding="utf-8"))

    return _run


def test_gate_failure_stops_before_any_generation(run):
    code, out = run(target=-1.0)
    assert code == 1
    assert out["stage_status"]["gate"].startswith("FAILED")
    assert "stage_a" not in out


def test_full_pipeline_with_failing_models(run):
    _, first = run(target=-1.0)
    code, out = run(target=first["reproduction_gate"]["reproduced_mrr@10"])
    assert code == 0, out["stage_status"]
    assert set(out["stage_status"].values()) == {"OK"}

    # L2 fails while generating, L3 fails to load: both recorded and excluded.
    assert set(out["failures"]) == {"L2", "L3"}
    assert set(out["models"]) == {"L1"}
    assert out["selected_model"]["id"] == "L1"

    s = out["samples"]
    assert s["validation"] == 16 and s["test"] == 24 and s["test_unseen"] == 16

    # Closed-book is always wrong, the oracle passage always right.
    assert out["stage_a"]["L1/closed_book"]["f1"] == 0.0
    assert out["stage_b"]["B4_oracle"]["f1"] == 1.0
    assert out["stage_b"]["B4_oracle"]["em"] == 1.0
    assert set(out["stage_b"]) == {
        "B0_hybrid_top3",
        "B1_hybrid_top1",
        "B2_hybrid_top5",
        "B3_bm25_top3",
        "B4_oracle",
    }

    # The fake only reads passage [1], so k does not change F1: the tie rule picks k = 1.
    assert out["selected_k"]["k"] == 1
    k = out["selected_k"]["k"]
    assert set(out["test"]) >= {"closed_book", f"bm25_top{k}", f"hybrid_top{k}", "paired_unseen"}
    assert out["decision"]["rag_adopted"] is True
    assert len(out["stage_a"]["L1/hybrid_top3"]["examples"]) == 16  # fewer than 30 questions

    # Every reported mean equals the mean of the per-question scores implied by the examples.
    ex = out["stage_b"]["B1_hybrid_top1"]["examples"]
    em = np.mean([float(e["pred"] in e["gold"]) for e in ex])
    assert out["stage_b"]["B1_hybrid_top1"]["em"] == pytest.approx(em, abs=1e-4)


def pq_from(f1s):
    return {f"q{i}": {"em": f, "f1": f} for i, f in enumerate(f1s)}


GROUPS = {f"q{i}": f"p{i}" for i in range(40)}  # one question per cluster


def perturbed(base, up=0.2, down=0.1):
    """Same scores except one question up and one down: higher mean, CI containing 0."""
    vals = list(base)
    vals[0] += up
    vals[1] -= down
    return vals


BASE = [0.5] * 20 + [0.7] * 20


def test_select_k_prefers_smaller_k_when_tied():
    pq = {1: pq_from(BASE), 3: pq_from(perturbed(BASE)), 5: pq_from(BASE)}
    k, reason = rg.select_k(pq, GROUPS)
    assert k == 1 and "ties" in reason


def test_select_k_keeps_best_when_clearly_better():
    low, high = pq_from([0.0] * 40), pq_from([1.0] * 40)
    k, reason = rg.select_k({1: low, 3: low, 5: high}, GROUPS)
    assert k == 5 and reason == "highest validation F1"


def test_select_model_tie_goes_to_smaller_model():
    pq = {"L1/hybrid_top3": pq_from(BASE), "L3/hybrid_top3": pq_from(perturbed(BASE))}
    sel, reason, comp = rg.select_model(["L1", "L3"], pq, GROUPS)
    assert comp["best"] == "L3" and rg.contains_zero(comp["f1_diff"]["ci95"])
    assert sel == "L1", reason


def test_select_model_clear_win_goes_to_best():
    pq = {"L1/hybrid_top3": pq_from(BASE), "L3/hybrid_top3": pq_from([x + 0.1 for x in BASE])}
    sel, _, comp = rg.select_model(["L1", "L3"], pq, GROUPS)
    assert sel == "L3" and comp["f1_diff"]["ci95"][0] > 0


def test_select_model_single_candidate():
    sel, reason, comp = rg.select_model(["L2"], {"L2/hybrid_top3": pq_from([1.0])}, GROUPS)
    assert sel == "L2" and comp is None


@pytest.mark.parametrize(
    ("ci", "expected"),
    [([-0.1, 0.2], True), ([0.0, 0.1], True), ([0.01, 0.1], False), ([-0.2, -0.01], False)],
)
def test_contains_zero(ci, expected):
    assert rg.contains_zero(ci) is expected
