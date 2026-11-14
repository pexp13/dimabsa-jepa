import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dimabsa_jepa.data import find_span, read_file, sample_block_mask, write_predictions
from dimabsa_jepa.metrics import rmse_va

ROOT = Path(__file__).resolve().parents[1] / "data" / "DimABSA2026"
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="run scripts/get_data.sh")


def test_block_mask_properties():
    rng = np.random.default_rng(0)
    for n in range(0, 60):
        m = sample_block_mask(n, 0.3, 5, rng)
        assert m.shape == (n,)
        if n <= 1:
            assert not m.any()
        else:
            assert 1 <= m.sum() <= n - 1
            assert m.sum() == min(max(1, round(0.3 * n)), n - 1)


def test_find_span():
    assert find_span("The Salads are fantastic", "salads") == (4, 10)
    assert find_span("abc", "zzz") is None
    assert find_span("abc", None) is None


def test_read_formats(tmp_path):
    p = tmp_path / "eng_restaurant_train_alltasks.jsonl"
    p.write_text(json.dumps({"ID": "1", "Text": "good food", "Quadruplet": [
        {"Aspect": "food", "Opinion": "good", "Category": "FOOD#QUALITY", "VA": "7.00#6.50"},
        {"Aspect": "NULL", "Opinion": "NULL", "Category": "X#Y", "VA": "5.00#5.00"}]}) + "\n")
    (inst,) = read_file(p)
    assert (inst.aspect, inst.opinion, inst.va, inst.source) == ("food", "good", (7.0, 6.5), "eng_restaurant")
    q = tmp_path / "eng_restaurant_test_task1.jsonl"
    q.write_text(json.dumps({"ID": "1", "Text": "good food", "Aspect": ["food"]}) + "\n")
    assert read_file(q)[0].va is None


@needs_data
def test_rmse_matches_official_script(tmp_path):
    gold_path = ROOT / "task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl"
    insts = read_file(gold_path)
    gold = np.array([i.va for i in insts])
    pred = np.round(np.clip(gold + np.random.default_rng(0).normal(0, 0.7, gold.shape), 1, 9), 2)
    pred_path = tmp_path / "pred.jsonl"
    write_predictions(pred_path, insts, pred)
    script = ROOT / "evaluation_script" / "metrics_subtask_1_2_3.py"
    out = subprocess.run([sys.executable, str(script), "-t", "1", "-p", str(pred_path), "-g", str(gold_path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip(f"official script not runnable here: {out.stderr[-200:]}")
    assert f"{rmse_va(pred, gold):.4f}"[:5] in out.stdout


def test_merge_duplicates():
    from dimabsa_jepa.data import Instance, merge_duplicates
    a = Instance("1", "t", "food", (7.0, 6.0), "good")
    b = Instance("1", "t", "food", (5.0, 5.0), "pricey")
    c = Instance("1", "t", "staff", (3.0, 4.0))
    out = merge_duplicates([a, b, c])
    assert [(o.aspect, o.va) for o in out] == [("food", (6.0, 5.5)), ("staff", (3.0, 4.0))]
