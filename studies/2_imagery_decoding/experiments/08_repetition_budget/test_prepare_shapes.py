import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_shapes", Path(__file__).with_name("prepare_shapes.py"))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def trial(ordinal=1):
    return {"trial": ordinal, "subject": 1, "session": 1, "task": "FVI", "class": "circle"}


def event(identifier="7", name="circle"):
    return {"trial": identifier, "type": name, "latency": "20000", "onset": "20"}


def test_original_ids_and_timing_preserved():
    trials = [trial()]
    prep.annotate(trials, [event()], "source.bdf", "archive-hash")
    assert trials[0]["trial"] == 7
    assert trials[0]["event_ordinal"] == 1
    assert trials[0]["source_latency_samples"] == 20000
    assert trials[0]["source_onset_seconds"] == 20
    assert trials[0]["source_archive_md5"] == "archive-hash"


def test_label_disagreement_rejected():
    with pytest.raises(ValueError, match="label differs"):
        prep.annotate([trial()], [event(name="square")], "source.bdf", "hash")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate source"):
        prep.annotate([trial(1), trial(2)], [event(), event()], "source.bdf", "hash")


def test_wrong_version_rejected_before_preparation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "figshare_manifest.json").write_text(json.dumps({"id": 30227503, "version": 3}))
    with pytest.raises(ValueError, match="version-4"):
        prep.prepare(raw, tmp_path / "out", [1])
    assert not (tmp_path / "out").exists()


def test_existing_subject_not_overwritten(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "figshare_manifest.json").write_text(json.dumps({"id": 30227503, "version": 4, "files": []}))
    out = tmp_path / "out"
    out.mkdir()
    destination = out / "gao_sub-01.pt"
    destination.write_bytes(b"keep this")
    with pytest.raises(FileExistsError, match="overwrite"):
        prep.prepare(raw, out, [1])
    assert destination.read_bytes() == b"keep this"
