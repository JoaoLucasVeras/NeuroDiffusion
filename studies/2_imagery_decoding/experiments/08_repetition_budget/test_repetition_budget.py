import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("repetition_budget", Path(__file__).with_name("repetition_budget.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
rb.torch.set_num_threads(1)


def session(seed, subject=1, strength=0.8):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(3), 40)
    lat = np.arange(120) * 17000
    X = strength * np.eye(3)[y] + rng.normal(size=(120, 3))
    X, _, _ = rb.standardize(X)
    return {"X": X, "y": y, "ids": [[subject, seed, "FVI", int(i)] for i in range(120)],
            "groups": {k: rb.groups_for(y, lat, k) for k in rb.BUDGETS}, "subject": subject}


@pytest.mark.parametrize("budget,decisions", [(1, 120), (2, 60), (4, 30), (8, 15)])
def test_disjoint_complete_groups(budget, decisions):
    y = np.tile(np.arange(3), 40)
    lat = np.arange(120) * 17000
    g = rb.groups_for(y, lat, budget)
    assert g.shape == (decisions, budget)
    assert len(set(g.ravel())) == 120
    assert np.all(y[g] == y[g[:, :1]])
    assert np.all(np.diff(lat[g], axis=1) > 0)


def test_group_membership_determined_by_timing_not_input_order():
    rng = np.random.default_rng(9)
    y, lat = np.tile(np.arange(3), 40), np.arange(120) * 17000
    original = rb.groups_for(y, lat, 4)
    order = rng.permutation(120)
    reordered = rb.groups_for(y[order], lat[order], 4)
    np.testing.assert_array_equal(lat[original], lat[order][reordered])


def test_bad_counts_and_duplicate_times_rejected():
    y, lat = np.tile(np.arange(3), 40), np.arange(120) * 17000
    with pytest.raises(ValueError):
        rb.groups_for(y[:-1], lat[:-1], 4)
    with pytest.raises(ValueError, match="Duplicate"):
        rb.groups_for(y, np.zeros(120), 4)
    y[0] = 1
    with pytest.raises(ValueError, match="40 events"):
        rb.groups_for(y, lat, 4)


def test_hand_computed_metric():
    y = np.repeat(np.arange(3), 2)
    p = np.array([[.9,.05,.05],[.1,.8,.1],[.6,.3,.1],[.1,.7,.2],[.2,.2,.6],[.6,.3,.1]])
    pooled = rb.pool_scores(p, y, np.arange(6).reshape(3, 2))
    single = rb.pool_scores(p, y, np.arange(6).reshape(6, 1))
    assert pooled["accuracy"] == pytest.approx(2 / 3)
    assert single["accuracy"] == .5


def test_constant_control_and_ties():
    s = session(1)
    for k in rb.BUDGETS:
        result = rb.pool_scores(np.full((120,3), 1 / 3), s["y"], s["groups"][k])
        assert result["accuracy"] == pytest.approx(1 / 3)
        assert set(result["predicted"]) == {0}


def test_invalid_predictions_and_mixed_groups_rejected():
    s = session(1)
    p = np.full((120,3), 1 / 3)
    with pytest.raises(ValueError, match="cover each"):
        rb.pool_scores(p, s["y"], np.zeros((30,4), dtype=int))
    with pytest.raises(ValueError, match="Mixed"):
        rb.pool_scores(p, s["y"], np.arange(120).reshape(30,4))
    p[0,0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        rb.pool_scores(p, s["y"], s["groups"][4])


def test_fit_once_per_direction_and_predictions_saved():
    calls = []
    def estimator(X, y, test, seed):
        calls.append((X.copy(), y.copy(), test.copy(), seed))
        return np.full((len(test),3), 1 / 3)
    sessions = {1: session(1), 2: session(2)}
    result = rb.evaluate_subject(sessions, 2022, estimator=estimator, keep_predictions=True)
    assert len(calls) == 2  # Not eight fits for four budgets.
    assert result["delta"] == 0
    assert len(result["directions"][0]["event_probabilities"]) == 120
    np.testing.assert_array_equal(calls[0][1], sessions[1]["y"])
    assert result["directions"][0]["event_ids"] == sessions[2]["ids"]


def test_training_direction_unaffected_by_test_eeg_or_labels():
    sessions = {1: session(1), 2: session(2)}
    captured = []
    def estimator(X, y, test, seed):
        captured.append((X.copy(), y.copy()))
        return np.full((len(test),3), 1 / 3)
    rb.evaluate_subject(sessions, 2022, estimator=estimator)
    changed = copy.deepcopy(sessions)
    changed[2]["X"] *= 99
    changed[2]["y"] = (changed[2]["y"] + 1) % 3
    rb.evaluate_subject(changed, 2022, estimator=estimator)
    # Session 2 becomes training data in the reverse direction, so compare only 1->2.
    np.testing.assert_array_equal(captured[0][0], captured[2][0])
    np.testing.assert_array_equal(captured[0][1], captured[2][1])


def test_paired_permutations_deterministic_no_input_mutation():
    sessions = {1: session(1), 2: session(2)}
    before = copy.deepcopy(sessions)
    calls = []
    def estimator(X, y, test, seed):
        calls.append(y.copy())
        return np.full((len(test),3), 1 / 3)
    rb.evaluate_subject(sessions, 2022, 0, estimator)
    rb.evaluate_subject(sessions, 2022, 0, estimator)
    np.testing.assert_array_equal(calls[0], calls[2])
    assert not np.array_equal(calls[0], sessions[1]["y"])
    for sid in (1,2):
        np.testing.assert_array_equal(before[sid]["y"], sessions[sid]["y"])


def test_seeded_signal_noise_and_shuffled_labels():
    results = []
    for subject in (1,2,3):
        sessions = {1: session(subject * 10, subject, 1.2), 2: session(subject * 10 + 1, subject, 1.2)}
        results.append(rb.evaluate_subject(sessions, 2022))
    assert np.mean([r["accuracy"]["4"] for r in results]) > .7
    assert np.mean([r["delta"] for r in results]) > .1
    noise, shuffled = [], []
    for subject in (1,2,3):
        sessions = {1: session(subject * 10, subject, 0), 2: session(subject * 10 + 1, subject, 0)}
        noise.append(rb.evaluate_subject(sessions, 2022)["accuracy"]["4"])
        signal = {1: session(subject * 10, subject, 1.2), 2: session(subject * 10 + 1, subject, 1.2)}
        shuffled.append(rb.evaluate_subject(signal, 2022, permutation=0)["accuracy"]["4"])
    assert abs(np.mean(noise) - 1 / 3) < .15
    assert abs(np.mean(shuffled) - 1 / 3) < .15


def test_p_values_bootstrap_and_gate_require_complete_cohort():
    assert rb.p_value(1, [0,1,2]) == .75
    assert rb.p_value(1, []) is None
    assert rb.bootstrap([.2,.2], 2022) == pytest.approx([.2,.2])
    observed = [{"subject": s, "accuracy": {"1": .4,"2": .5,"4": .6,"8": .7}, "delta": .2} for s in rb.COHORT]
    null = [[{"subject": s, "accuracy": {"1": 1/3,"2": 1/3,"4": 1/3,"8": 1/3}, "delta": 0} for s in rb.COHORT]] * 1000
    assert rb.summarize(observed, null, 2022, 1000)["practical_advance"]
    assert not rb.summarize(observed, null[:2], 2022, 2)["statistical_advance"]
    assert not rb.summarize(observed[:1], [n[:1] for n in null], 2022, 1000)["statistical_advance"]


def test_invalid_prepared_format_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "digest", lambda _: "hash")
    monkeypatch.setattr(rb.torch, "load", lambda *a, **kw: {"format": "wrong"})
    with pytest.raises(ValueError, match="format/version"):
        rb.load_inputs(tmp_path, [1], False)


def test_standardization_finite_and_deterministic():
    X = np.array([[1,5],[2,5],[3,5]])
    a, mean, spread = rb.standardize(X)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a.mean(0), 0, atol=1e-10)
    assert mean.tolist() == [2,5]
    with pytest.raises(ValueError):
        rb.standardize([[np.nan]])


def test_checkpoint_consistency_checks():
    observed = [{"subject": 1, "accuracy": {"1": .4, "2": .5, "4": .6, "8": .7}, "delta": .2}]
    rb.validate_results(observed, [], [1], 2)
    with pytest.raises(ValueError, match="cohort"):
        rb.validate_results(observed, [], [2], 2)
    with pytest.raises(ValueError, match="excess"):
        rb.validate_results(observed, [observed], [1], 0)
    wrong = copy.deepcopy(observed)
    wrong[0]["delta"] = .9
    with pytest.raises(ValueError, match="difference"):
        rb.validate_results(wrong, [], [1], 2)


def fake_inputs(compute):
    manifest = [{"subject": 1, "sessions": [{"four_event_span_seconds": [55.]}, {"four_event_span_seconds": [60.]}]}]
    cohort = {1: {1: session(1), 2: session(2)}} if compute else {}
    return manifest, cohort, {"synthetic": np.ones((2,3))}


def test_validation_mode_never_fits_or_extracts(tmp_path, monkeypatch):
    calls = []
    def load(data, subjects, compute):
        calls.append(compute)
        return fake_inputs(compute)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected scientific operation")
    monkeypatch.setattr(rb, "load_inputs", load)
    monkeypatch.setattr(rb, "features", forbidden)
    monkeypatch.setattr(rb, "logreg", forbidden)
    out = tmp_path / "validation"
    monkeypatch.setattr(rb.sys, "argv", ["runner", "--validate-only", "--subjects", "1", "--out", str(out)])
    rb.main()
    assert calls == [False]
    assert json.loads((out / "validation.json").read_text())["scientific_models_fitted"] == 0
    assert not (out / "observed.json").exists()


def test_synthetic_pipeline_manifest_before_fit_and_resume(tmp_path, monkeypatch):
    out = tmp_path / "run"
    calls = []
    monkeypatch.setattr(rb, "load_inputs", lambda data, subjects, compute: fake_inputs(compute))
    monkeypatch.setattr(rb, "digest", lambda _: "unchanged")
    def estimator(X, labels, target, n, l2, **kwargs):
        assert (out / "manifest.json").is_file()
        calls.append(labels.copy())
        p = np.exp(target - target.max(1,keepdims=True))
        return p / p.sum(1,keepdims=True)
    monkeypatch.setattr(rb, "logreg", estimator)
    argv = ["runner", "--run", "--subjects", "1", "--permute", "2", "--out", str(out)]
    monkeypatch.setattr(rb.sys, "argv", argv)
    rb.main()
    assert len(calls) == 6  # 2 observed + 2 directions per permutation.
    initial = json.loads((out / "summary.json").read_text())
    progress = json.loads((out / "progress.json").read_text())
    progress["null"] = progress["null"][:1]
    progress["completed"] = 1
    (out / "progress.json").write_text(json.dumps(progress))
    monkeypatch.setattr(rb.sys, "argv", argv + ["--resume"])
    rb.main()
    assert len(calls) == 8
    assert json.loads((out / "summary.json").read_text()) == initial
    monkeypatch.setattr(rb, "digest", lambda _: "changed")
    with pytest.raises(ValueError, match="Resume rejected"):
        rb.main()
    assert len(calls) == 8
