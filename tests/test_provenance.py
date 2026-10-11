"""Evaluation provenance remains usable in source archives without Git."""
from pathlib import Path
import pytest
from mjev.benchmark.provenance import capture, revision, sha256


def test_model_revision_must_be_immutable():
    with pytest.raises(ValueError):
        revision("main")
    assert revision("a" * 40) == "a" * 40


def test_capture_without_git_preserves_source_evidence(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr("mjev.benchmark.provenance.subprocess.check_output", unavailable)
    evidence = capture({"backend": "hf"}, "a" * 40)
    assert evidence["code_commit"] is None
    assert evidence["working_tree_dirty"] is None
    assert evidence["model_revision_status"] == "declared_local_unverified"
    root = Path(__file__).resolve().parents[1]
    for source in ("mjev/hf.py", "docjev/service.py", "demo_docjev.py"):
        assert evidence["source_sha256"][source] == sha256(root / source)
    assert evidence["dependencies"]["torch"]
