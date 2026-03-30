"""Tests for torch_checkpoint_triage.py"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch_checkpoint_triage as triage  # noqa: E402


# ===========================================================================
# inspect_state_dict
# ===========================================================================

class TestInspectStateDict:
    def test_clean_state_dict_no_anomalies(self, simple_state_dict):
        anomalies = triage.inspect_state_dict(simple_state_dict, threshold=100.0)
        assert anomalies == []

    def test_detects_nan(self, poisoned_state_dict):
        anomalies = triage.inspect_state_dict(poisoned_state_dict, threshold=100.0)
        nan_msgs = [a for a in anomalies if "NaN" in a]
        assert len(nan_msgs) >= 1

    def test_detects_inf(self, poisoned_state_dict):
        anomalies = triage.inspect_state_dict(poisoned_state_dict, threshold=100.0)
        inf_msgs = [a for a in anomalies if "Inf" in a]
        assert len(inf_msgs) >= 1

    def test_detects_extreme_weight(self, poisoned_state_dict):
        anomalies = triage.inspect_state_dict(poisoned_state_dict, threshold=10.0)
        extreme_msgs = [a for a in anomalies if "max|w|" in a]
        assert len(extreme_msgs) >= 1

    def test_skips_non_float_tensors(self):
        state = {"int_param": torch.tensor([1, 2, 3])}
        anomalies = triage.inspect_state_dict(state, threshold=100.0)
        assert anomalies == []

    def test_empty_state_dict(self):
        anomalies = triage.inspect_state_dict({}, threshold=100.0)
        assert anomalies == []


# ===========================================================================
# KL divergence
# ===========================================================================

class TestKLDivergence:
    def test_identical_distributions_zero(self):
        p = [0.25, 0.25, 0.25, 0.25]
        kl = triage.kl_divergence(p, p)
        assert abs(kl) < 1e-10

    def test_different_distributions_positive(self):
        p = [0.9, 0.05, 0.025, 0.025]
        q = [0.25, 0.25, 0.25, 0.25]
        kl = triage.kl_divergence(p, q)
        assert kl > 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            triage.kl_divergence([0.5, 0.5], [0.33, 0.33, 0.34])


# ===========================================================================
# tensor_histogram
# ===========================================================================

class TestTensorHistogram:
    def test_output_is_normalized(self):
        t = torch.randn(100)
        hist, bounds = triage.tensor_histogram(t, bins=16)
        assert abs(sum(hist) - 1.0) < 1e-4

    def test_output_has_correct_bins(self):
        t = torch.randn(100)
        hist, _ = triage.tensor_histogram(t, bins=32)
        assert len(hist) == 32

    def test_empty_tensor(self):
        t = torch.tensor([])
        hist, bounds = triage.tensor_histogram(t, bins=8)
        assert len(hist) == 8

    def test_constant_tensor_does_not_crash(self):
        t = torch.ones(50) * 3.0
        hist, bounds = triage.tensor_histogram(t, bins=8)
        assert len(hist) == 8
        assert bounds[0] < bounds[1]


# ===========================================================================
# compute_fingerprint / compare_fingerprints
# ===========================================================================

class TestFingerprinting:
    def test_fingerprint_contains_expected_keys(self, simple_state_dict):
        fp = triage.compute_fingerprint(simple_state_dict)
        assert "weight" in fp
        assert "histogram" in fp["weight"]
        assert "shape" in fp["weight"]
        assert "range" in fp["weight"]

    def test_compare_identical_fingerprints_no_anomalies(self, simple_state_dict):
        fp = triage.compute_fingerprint(simple_state_dict)
        anomalies, divergences = triage.compare_fingerprints(fp, fp, kl_threshold=0.5)
        assert anomalies == []
        for val in divergences.values():
            assert val < 1e-6

    def test_compare_detects_missing_parameter(self, simple_state_dict):
        ref = triage.compute_fingerprint(simple_state_dict)
        candidate = {k: v for k, v in ref.items() if k != "weight"}
        anomalies, _ = triage.compare_fingerprints(ref, candidate, kl_threshold=0.5)
        missing_msgs = [a for a in anomalies if "Missing" in a]
        assert len(missing_msgs) >= 1

    def test_compare_detects_unexpected_parameter(self, simple_state_dict):
        ref = triage.compute_fingerprint(simple_state_dict)
        candidate = dict(ref)
        candidate["injected_layer"] = ref["weight"]
        anomalies, _ = triage.compare_fingerprints(ref, candidate, kl_threshold=0.5)
        unexpected = [a for a in anomalies if "Unexpected" in a]
        assert len(unexpected) >= 1


# ===========================================================================
# extract_state_dict
# ===========================================================================

class TestExtractStateDict:
    def test_from_plain_dict(self, simple_state_dict):
        result = triage.extract_state_dict(simple_state_dict)
        assert result is not None
        assert "weight" in result

    def test_from_dict_with_state_dict_key(self, simple_state_dict):
        wrapped = {"state_dict": simple_state_dict, "epoch": 10}
        result = triage.extract_state_dict(wrapped)
        assert result is not None
        assert "weight" in result

    def test_from_nn_module(self):
        model = nn.Linear(4, 2)
        result = triage.extract_state_dict(model)
        assert result is not None

    def test_returns_none_for_unknown(self):
        result = triage.extract_state_dict("not a state dict")
        assert result is None


# ===========================================================================
# find_checkpoints
# ===========================================================================

class TestFindCheckpoints:
    def test_finds_pt_files(self, tmp_path):
        (tmp_path / "model.pt").touch()
        (tmp_path / "model.pth").touch()
        (tmp_path / "not_a_checkpoint.txt").touch()
        found = list(triage.find_checkpoints(tmp_path))
        assert len(found) == 2

    def test_single_file(self, tmp_path):
        f = tmp_path / "single.pt"
        f.touch()
        found = list(triage.find_checkpoints(f))
        assert len(found) == 1

    def test_ignores_wrong_extensions(self, tmp_path):
        (tmp_path / "data.csv").touch()
        found = list(triage.find_checkpoints(tmp_path))
        assert len(found) == 0


# ===========================================================================
# CLI main
# ===========================================================================

class TestCheckpointMain:
    def test_main_with_clean_checkpoint(self, tmp_path):
        model = nn.Linear(4, 2)
        path = tmp_path / "clean.pt"
        torch.save(model.state_dict(), path)
        exit_code = triage.main([str(path)])
        assert exit_code == 0

    def test_main_json_output(self, tmp_path, capsys):
        model = nn.Linear(4, 2)
        path = tmp_path / "clean.pt"
        torch.save(model.state_dict(), path)
        exit_code = triage.main([str(path), "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert parsed[0]["loaded"] is True

    def test_main_nonexistent_path(self, tmp_path):
        exit_code = triage.main([str(tmp_path / "nonexistent.pt")])
        assert exit_code == 0  # logs error but doesn't crash

    def test_main_with_fingerprint_output(self, tmp_path):
        model = nn.Linear(4, 2)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        fp_dir = tmp_path / "fingerprints"
        exit_code = triage.main([str(path), "--write-fingerprint", str(fp_dir)])
        assert exit_code == 0
        assert (fp_dir / "model.fingerprint.json").exists()
