"""Local synthetic checks; no real EEG, CLIP downloads or cluster access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("visual_axes", Path(__file__).with_name("low_dimensional_clip.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def synthetic(n=30, seed=7):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 4))
    targets = m.normalize(latent @ rng.normal(size=(4, 32)))
    x = latent @ rng.normal(size=(4, 8)) + rng.normal(0, 0.01, (n, 8))
    return x, targets


def test_manifest_groups_and_determinism():
    stems = [str(i) for i in range(33)]
    a = m.make_manifest(stems)
    assert a == m.make_manifest(stems)
    assert a != m.make_manifest(stems, seed=1)
    assert sorted(i for f in a["outer"] for i in f["test"]) == list(range(33))
    for f in a["outer"]:
        assert not set(f["train"]) & set(f["test"])
        for inner in f["inner"]:
            assert not set(inner["train"]) & set(inner["validation"])
            assert set(inner["train"] + inner["validation"]) == set(f["train"])
            assert not set(inner["train"] + inner["validation"]) & set(f["test"])


def test_metric_values_and_ties():
    y = np.eye(4)
    r = m.retrieval(y, y)
    assert np.mean(r["pairwise"]) == 1
    assert np.mean(r["top1"]) == 1
    assert np.mean(r["reciprocal_rank"]) == 1
    r = m.retrieval(np.ones((4, 4)), y)
    assert np.mean(r["pairwise"]) == 0.5
    assert np.mean(r["top1"]) == 0.25
    assert np.mean(r["reciprocal_rank"]) == pytest.approx(np.mean(1 / np.arange(1, 5)))
    with pytest.raises(ValueError):
        m.retrieval(y[:1], y[:1])


def test_signal_noise_and_shuffled_labels():
    x, y = synthetic(n=100)
    fitted = m.fit_ridge(x[:70], y[:70], 0.0001, dim=4)
    signal = np.mean(m.retrieval(m.predict(fitted, x[70:]), y[70:])["pairwise"])
    shuffled = m.fit_ridge(x[:70], np.random.default_rng(1).permutation(y[:70]), 0.0001, dim=4)
    scramble = np.mean(m.retrieval(m.predict(shuffled, x[70:]), y[70:])["pairwise"])
    noise = np.random.default_rng(1).normal(size=x.shape)
    noise_model = m.fit_ridge(noise[:70], y[:70], 0.1, dim=4)
    null = np.mean(m.retrieval(m.predict(noise_model, noise[70:]), y[70:])["pairwise"])
    assert signal > 0.9
    assert scramble < 0.7
    assert 0.3 < null < 0.7


@pytest.mark.parametrize("reduced", [False, True])
def test_outer_test_changes_cannot_change_fit_or_selection(reduced):
    x, y = synthetic()
    fold = m.make_manifest([str(i) for i in range(len(x))])["outer"][0]
    a = np.arange(len(x))
    original, scores = m.choose(x, y, fold, reduced, a)
    x2, y2 = x.copy(), y.copy()
    x2[fold["test"]] = 10000
    y2[fold["test"]] = -5000
    changed, scores2 = m.choose(x2, y2, fold, reduced, a)
    assert scores == scores2
    for key in original:
        np.testing.assert_equal(original[key], changed[key])


def test_rank_limits_and_ridge_projection_equivalence():
    x, y = synthetic()
    with pytest.raises(ValueError, match="rank"):
        m.fit_ridge(x[:5], y[:5], 0.1, dim=8)
    model = m.fit_ridge(x, y, 0.1, dim=4)
    full = m.fit_ridge(x, y, 0.1)
    np.testing.assert_allclose(model["weight"], full["weight"] @ model["basis"], atol=1e-12)
    fold = m.make_manifest([str(i) for i in range(len(x))])["outer"][0]
    _, scores = m.choose(x, y, fold, True, np.arange(len(x)))
    assert {r["dimension"] for r in scores} == {2, 4}


def test_selection_ties_prefer_fewer_axes_and_stronger_l2():
    _, y = synthetic()
    x = np.zeros((len(y), 3))
    fold = m.make_manifest([str(i) for i in range(len(x))])["outer"][0]
    model, _ = m.choose(x, y, fold, True, np.arange(len(x)))
    assert model["dimension"] == 2
    assert model["l2"] == 1.0


def test_followup_requires_complete_null_and_three_improvements():
    obs = {"mean_delta": 0.1, "subjects": {str(i): {"delta": 0.1} for i in range(4)}}
    assert m.decision_gate(obs, [], 0) == (None, 4, False)
    assert not m.decision_gate(obs, [0] * 99, 99)[2]
    assert m.decision_gate(obs, [0] * 1000, 1000)[2]
    assert not m.decision_gate(obs, [0.2] * 1000, 1000)[2]
    obs["subjects"]["0"]["delta"] = -0.1
    obs["subjects"]["1"]["delta"] = -0.1
    assert not m.decision_gate(obs, [0] * 1000, 1000)[2]


def test_nested_permutations_are_synchronized_and_reproducible():
    x, y = synthetic(n=20)
    manifest = m.make_manifest([str(i) for i in range(len(x))])
    r = m.evaluate({1: x, 2: x.copy()}, y, manifest, permutation=0)
    assert r["subjects"]["1"]["metrics"] == r["subjects"]["2"]["metrics"]
    q = m.evaluate({1: x}, y, manifest, permutation=0)
    for f, g in zip(r["subjects"]["1"]["folds"], q["subjects"]["1"]["folds"]):
        np.testing.assert_equal(f["training_target_indices"], g["training_target_indices"])
        for arm in ("full", "reduced"):
            np.testing.assert_equal(f["predictions"][arm], g["predictions"][arm])
    for fold, fitted in zip(manifest["outer"], r["subjects"]["1"]["folds"]):
        assert set(fitted["training_target_indices"]) == set(fold["train"])
        assert not set(fitted["training_target_indices"]) & set(fold["test"])


def payload():
    rng = np.random.default_rng(11)
    stems = ["image%d" % i for i in range(20)]
    entries = [{"subject": 1, "image": stem, "label": i, "window": w,
                "eeg": torch.tensor(rng.normal(size=(3, 125)), dtype=torch.float32)}
               for i, stem in enumerate(stems) for w in range(20)]
    return {"dataset": entries}, {"stems": stems, "emb": torch.tensor(rng.normal(size=(20, 768)))}


def test_loader_averages_events_and_validates_assets(tmp_path):
    data, cache = payload()
    dp, cp = tmp_path / "data.pt", tmp_path / "cache.pt"
    torch.save(data, dp)
    torch.save(cache, cp)
    features, targets, audit = m.load_events(dp, cp)
    assert features[1].shape == (20, 3)
    assert targets.shape == (20, 768)
    assert audit["window_counts"][1] == [20] * 20
    event = [e for e in data["dataset"] if e["image"] == audit["stimuli"][0]]
    filtered, _, _ = m.prepare([dict(e, eeg=e["eeg"].numpy()) for e in event], (8, 13), False)
    np.testing.assert_allclose(features[1][0], m.feat_bandpower(filtered).mean(0), atol=1e-5)
    with pytest.raises(ValueError, match="Missing local asset"):
        m.load_events(dp, tmp_path / "absent")
    torch.save({"stems": cache["stems"][:-1], "emb": cache["emb"][:-1]}, cp)
    with pytest.raises(ValueError, match="Missing embeddings"):
        m.load_events(dp, cp)
    _, _, audit = m.load_events(dp, cp, allow_missing_embeddings=True)
    assert audit["uncached_stimuli"] == [cache["stems"][-1]]


@pytest.mark.parametrize("failure", ["duplicate", "shape", "incomplete"])
def test_loader_rejects_malformed_events(tmp_path, failure):
    data, cache = payload()
    if failure == "duplicate":
        data["dataset"].append(data["dataset"][0])
    elif failure == "shape":
        data["dataset"][0]["eeg"] = torch.zeros(4, 125)
    else:
        data["dataset"].pop()
    dp, cp = tmp_path / "data.pt", tmp_path / "cache.pt"
    torch.save(data, dp)
    torch.save(cache, cp)
    with pytest.raises(ValueError):
        m.load_events(dp, cp)


def test_root_cli_help_and_missing_assets(tmp_path):
    script = Path(m.__file__)
    r = subprocess.run([sys.executable, str(script), "--help"], cwd=m.paths.ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    r = subprocess.run([sys.executable, str(script), "--dataset", str(tmp_path / "absent"),
                        "--cache", str(tmp_path / "absent_cache"), "--output", str(tmp_path / "out")],
                       cwd=m.paths.ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "No download was attempted" in r.stderr
    assert not (tmp_path / "out").exists()


def test_end_to_end_synthetic_outputs_and_no_overwrite(tmp_path, monkeypatch):
    data, cache = payload()
    dp, cp, out = tmp_path / "data.pt", tmp_path / "cache.pt", tmp_path / "out"
    torch.save(data, dp)
    torch.save(cache, cp)
    monkeypatch.setattr(sys, "argv", [m.__file__, "--dataset", str(dp), "--cache", str(cp),
                                    "--output", str(out), "--permute", "1"])
    original = m.evaluate

    def checked_evaluate(*args, **kwargs):
        assert (out / "manifest.json").is_file()
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "evaluate", checked_evaluate)
    m.main()
    observed = json.loads((out / "observed.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert observed["kind"] == m.KIND
    assert len(observed["subjects"]["1"]["folds"]) == 5
    assert len(summary["null_deltas"]) == 1
    assert summary["follow_up_gate"] is False
    assert summary["provenance"]["source_hashes"]
    assert summary["group_metrics"]["full"] == summary["subjects"]["1"]["metrics"]["full"]
    with pytest.raises(SystemExit):
        m.main()
