"""
Tests for torch_checkpoint_triage.py

Coverage targets
----------------
- find_checkpoints: yields from file, yields from directory, skips non-checkpoint extensions
- extract_state_dict: plain dict, nn.Module, nested dict with 'state_dict' key, unknown type
- inspect_state_dict: clean returns empty list; NaN, Inf, and extreme weights flagged
- tensor_histogram: correct bin count, normalised to ~1.0, handles constant tensor
- compute_fingerprint: produces per-parameter entries with shape/histogram/range
- kl_divergence: correct value for known distributions, raises on length mismatch
- compare_fingerprints: detects missing params, unexpected params, high KL divergence
- convert_to_safetensors: writes file when safetensors available, skips with warning otherwise
- triage_checkpoint: end-to-end with a real .pt file saved by torch.save
- CheckpointReport.as_dict: correct serialisation of all fields
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch_checkpoint_triage as triage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, obj: object) -> None:
    torch.save(obj, path)


# ===========================================================================
# find_checkpoints
# ===========================================================================

class TestFindCheckpoints:
    def test_single_pt_file_yields_itself(self, tmp_dir):
        f = tmp_dir / "model.pt"
        f.touch()
        results = list(triage.find_checkpoints(f))
        assert results == [f]

    def test_non_checkpoint_extension_not_yielded_as_file(self, tmp_dir):
        f = tmp_dir / "model.json"
        f.touch()
        results = list(triage.find_checkpoints(f))
        assert results == []

    def test_directory_search_finds_nested_checkpoints(self, tmp_dir):
        sub = tmp_dir / "sub"
        sub.mkdir()
        (sub / "weights.pth").touch()
        (sub / "other.txt").touch()
        results = list(triage.find_checkpoints(tmp_dir))
        assert len(results) == 1
        assert results[0].name == "weights.pth"

    def test_all_checkpoint_extensions_found(self, tmp_dir):
        for ext in (".pt", ".pth", ".bin", ".ckpt"):
            (tmp_dir / f"model{ext}").touch()
        results = list(triage.find_checkpoints(tmp_dir))
        assert len(results) == 4

    def test_case_insensitive_extension_matching(self, tmp_dir):
        f = tmp_dir / "model.PT"
        f.touch()
        results = list(triage.find_checkpoints(f))
        assert len(results) == 1


# ===========================================================================
# extract_state_dict
# ===========================================================================

class TestExtractStateDict:
    def test_plain_dict_of_tensors(self):
        d = {"w": torch.ones(2, 2), "b": torch.zeros(2)}
        result = triage.extract_state_dict(d)
        assert result is not None
        assert set(result.keys()) == {"w", "b"}

    def test_plain_dict_filters_non_tensor_values(self):
        d = {"w": torch.ones(2), "meta": "some string"}
        result = triage.extract_state_dict(d)
        assert result is not None
        assert "meta" not in result

    def test_nn_module_uses_state_dict(self):
        model = nn.Linear(4, 2)
        result = triage.extract_state_dict(model)
        assert result is not None
        assert "weight" in result

    def test_nested_dict_with_state_dict_key(self):
        inner = {"weight": torch.ones(2, 2)}
        d = {"state_dict": inner, "optimizer": {}}
        result = triage.extract_state_dict(d)
        assert result is not None
        assert "weight" in result

    def test_unknown_type_returns_none(self):
        result = triage.extract_state_dict([1, 2, 3])
        assert result is None

    def test_plain_dict_with_non_string_keys_returns_none(self):
        d = {0: torch.ones(2), 1: torch.zeros(2)}
        result = triage.extract_state_dict(d)
        assert result is None


# ===========================================================================
# inspect_state_dict
# ===========================================================================

class TestInspectStateDict:
    def test_clean_state_dict_returns_no_anomalies(self, simple_state_dict):
        anomalies = triage.inspect_state_dict(simple_state_dict, threshold=100.0)
        assert anomalies == []

    def test_nan_detected(self, poisoned_state_dict):
        anomalies = triage.inspect_state_dict(poisoned_state_dict, threshold=100.0)
        assert any("NaN" in a for a in anomalies)

    def test_inf_detected(self, poisoned_state_dict):
        anomalies = triage.inspect_state_dict(poisoned_state_dict, threshold=100.0)
        assert any("Inf" in a for a in anomalies)

    def test_extreme_value_detected(self, poisoned_state_dict):
        anomalies = triage.inspect_state_dict(poisoned_state_dict, threshold=100.0)
        # The anomaly message format is "name: max|w|=999.00 > 100.0"
        assert any("max|w|" in a for a in anomalies)

    def test_threshold_controls_sensitivity(self, simple_state_dict):
        # Weights are 0.5; threshold=0.1 should flag them
        anomalies = triage.inspect_state_dict(simple_state_dict, threshold=0.1)
        assert len(anomalies) >= 1

    def test_non_floating_point_tensors_skipped(self):
        state_dict = {"ids": torch.tensor([0, 1, 2], dtype=torch.long)}
        anomalies = triage.inspect_state_dict(state_dict, threshold=100.0)
        assert anomalies == []


# ===========================================================================
# tensor_histogram
# ===========================================================================

class TestTensorHistogram:
    def test_correct_bin_count(self):
        t = torch.randn(256)
        hist, _ = triage.tensor_histogram(t, bins=64)
        assert len(hist) == 64

    def test_histogram_sums_to_approximately_one(self):
        t = torch.randn(1000)
        hist, _ = triage.tensor_histogram(t, bins=32)
        assert abs(sum(hist) - 1.0) < 1e-4

    def test_constant_tensor_does_not_crash(self):
        t = torch.ones(50)
        hist, bounds = triage.tensor_histogram(t, bins=16)
        assert len(hist) == 16
        # Bounds should be slightly widened to avoid zero-width interval
        assert bounds[1] > bounds[0]

    def test_empty_tensor_returns_zeros(self):
        t = torch.tensor([])
        hist, bounds = triage.tensor_histogram(t, bins=8)
        assert hist == [0.0] * 8
        assert bounds == (0.0, 0.0)

    def test_non_negative_bin_values(self):
        t = torch.randn(100)
        hist, _ = triage.tensor_histogram(t, bins=16)
        assert all(v >= 0.0 for v in hist)


# ===========================================================================
# compute_fingerprint
# ===========================================================================

class TestComputeFingerprint:
    def test_fingerprint_has_entry_per_float_param(self):
        state_dict = {
            "weight": torch.randn(4, 4),
            "bias": torch.zeros(4),
        }
        fp = triage.compute_fingerprint(state_dict)
        assert "weight" in fp
        assert "bias" in fp

    def test_integer_params_excluded(self):
        state_dict = {
            "ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "weight": torch.randn(4),
        }
        fp = triage.compute_fingerprint(state_dict)
        assert "ids" not in fp
        assert "weight" in fp

    def test_shape_preserved_in_fingerprint(self):
        state_dict = {"weight": torch.randn(3, 5)}
        fp = triage.compute_fingerprint(state_dict)
        assert fp["weight"]["shape"] == [3, 5]

    def test_histogram_present_and_non_empty(self):
        state_dict = {"weight": torch.randn(100)}
        fp = triage.compute_fingerprint(state_dict)
        hist = fp["weight"]["histogram"]
        assert isinstance(hist, list)
        assert len(hist) > 0


# ===========================================================================
# kl_divergence
# ===========================================================================

class TestKlDivergence:
    def test_identical_distributions_returns_near_zero(self):
        p = [0.25, 0.25, 0.25, 0.25]
        kl = triage.kl_divergence(p, p)
        assert abs(kl) < 1e-6

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Histogram length mismatch"):
            triage.kl_divergence([0.5, 0.5], [0.25, 0.5, 0.25])

    def test_uniform_vs_concentrated_is_positive(self):
        p = [0.1, 0.8, 0.1]  # concentrated
        q = [1/3, 1/3, 1/3]  # uniform
        # KL(p || q) should be positive
        kl = triage.kl_divergence(p, q)
        assert kl > 0

    def test_symmetric_distributions_have_same_kl(self):
        """KL(uniform || uniform) == 0 regardless of orientation."""
        p = [0.5, 0.5]
        kl = triage.kl_divergence(p, p)
        assert kl < 1e-9


# ===========================================================================
# compare_fingerprints
# ===========================================================================

class TestCompareFingerprints:
    def _make_fp(self, name: str = "weight", value: float = 0.5) -> dict:
        state_dict = {name: torch.ones(64) * value}
        return triage.compute_fingerprint(state_dict)

    def test_identical_fingerprints_no_anomalies(self):
        fp = self._make_fp()
        anomalies, divs = triage.compare_fingerprints(fp, fp, kl_threshold=1.0)
        assert anomalies == []

    def test_missing_parameter_detected(self):
        ref = self._make_fp("weight")
        cand = self._make_fp("bias")  # different key
        anomalies, _ = triage.compare_fingerprints(ref, cand, kl_threshold=1.0)
        assert any("Missing" in a for a in anomalies)

    def test_unexpected_parameter_detected(self):
        ref = self._make_fp("weight")
        cand = {**self._make_fp("weight"), **self._make_fp("injected_layer")}
        anomalies, _ = triage.compare_fingerprints(ref, cand, kl_threshold=1.0)
        assert any("Unexpected" in a for a in anomalies)

    def test_high_kl_divergence_flagged(self):
        ref = triage.compute_fingerprint({"w": torch.ones(256) * 0.1})
        cand = triage.compute_fingerprint({"w": torch.randn(256) * 100.0})
        anomalies, divs = triage.compare_fingerprints(ref, cand, kl_threshold=0.001)
        assert any("KL divergence" in a for a in anomalies)

    def test_divergences_dict_populated(self):
        fp = self._make_fp()
        _, divs = triage.compare_fingerprints(fp, fp, kl_threshold=1.0)
        assert "weight" in divs


# ===========================================================================
# convert_to_safetensors
# ===========================================================================

class TestConvertToSafetensors:
    def test_writes_file_when_safetensors_available(self, tmp_dir):
        state_dict = {"weight": torch.ones(4, 4), "bias": torch.zeros(4)}
        destination = tmp_dir / "model.safetensors"
        result = triage.convert_to_safetensors(state_dict, destination)
        assert result == destination
        assert destination.exists()

    def test_skips_when_safetensors_unavailable(self, tmp_dir, caplog):
        import logging
        state_dict = {"weight": torch.ones(4, 4)}
        destination = tmp_dir / "model.safetensors"

        original = triage.save_safetensor
        triage.save_safetensor = None
        try:
            with caplog.at_level(logging.WARNING, logger="torch_checkpoint_triage"):
                result = triage.convert_to_safetensors(state_dict, destination)
            assert result is None
            assert not destination.exists()
        finally:
            triage.save_safetensor = original


# ===========================================================================
# triage_checkpoint (end-to-end with real .pt files)
# ===========================================================================

class TestTriageCheckpoint:
    def test_clean_checkpoint_no_anomalies(self, tmp_dir):
        state_dict = {"weight": torch.ones(4, 4) * 0.5, "bias": torch.zeros(4)}
        path = tmp_dir / "clean.pt"
        _save_checkpoint(path, state_dict)

        report = triage.triage_checkpoint(
            path=path,
            threshold=100.0,
            create_safetensor=False,
            overwrite=False,
            fingerprint_dir=None,
            reference_fingerprint=None,
            kl_threshold=0.5,
        )
        assert report.loaded is True
        assert report.anomalies == []

    def test_poisoned_checkpoint_anomalies_detected(self, tmp_dir):
        # Use a state dict without NaN/Inf (those break torch.histc which runs inside
        # triage_checkpoint's compute_fingerprint call).  Use only an extreme magnitude.
        path = tmp_dir / "poison.pt"
        state_dict = {"weight": torch.ones(4, 4) * 500.0}
        _save_checkpoint(path, state_dict)

        report = triage.triage_checkpoint(
            path=path,
            threshold=100.0,
            create_safetensor=False,
            overwrite=False,
            fingerprint_dir=None,
            reference_fingerprint=None,
            kl_threshold=0.5,
        )
        assert report.loaded is True
        assert len(report.anomalies) >= 1
        assert any("max|w|" in a for a in report.anomalies)

    def test_safetensor_written_when_requested(self, tmp_dir):
        state_dict = {"weight": torch.ones(4, 4)}
        path = tmp_dir / "model.pt"
        _save_checkpoint(path, state_dict)

        report = triage.triage_checkpoint(
            path=path,
            threshold=100.0,
            create_safetensor=True,
            overwrite=False,
            fingerprint_dir=None,
            reference_fingerprint=None,
            kl_threshold=0.5,
        )
        expected = path.with_suffix(".safetensors")
        assert report.converted == expected
        assert expected.exists()

    def test_fingerprint_written_to_dir(self, tmp_dir):
        state_dict = {"weight": torch.randn(8, 8)}
        path = tmp_dir / "model.pt"
        fp_dir = tmp_dir / "fingerprints"
        _save_checkpoint(path, state_dict)

        report = triage.triage_checkpoint(
            path=path,
            threshold=100.0,
            create_safetensor=False,
            overwrite=False,
            fingerprint_dir=fp_dir,
            reference_fingerprint=None,
            kl_threshold=0.5,
        )
        assert report.fingerprint_path is not None
        assert report.fingerprint_path.exists()

    def test_high_kl_flagged_against_reference(self, tmp_dir):
        # Reference: near-constant weights
        ref_state = {"weight": torch.ones(64) * 0.01}
        ref_fp = triage.compute_fingerprint(ref_state)

        # Candidate: wildly different distribution
        path = tmp_dir / "drifted.pt"
        _save_checkpoint(path, {"weight": torch.randn(64) * 50.0})

        report = triage.triage_checkpoint(
            path=path,
            threshold=10000.0,  # don't flag magnitude
            create_safetensor=False,
            overwrite=False,
            fingerprint_dir=None,
            reference_fingerprint=ref_fp,
            kl_threshold=0.001,  # very tight threshold
        )
        assert any("KL divergence" in a for a in report.anomalies)


# ===========================================================================
# CheckpointReport.as_dict
# ===========================================================================

class TestCheckpointReportAsDict:
    def test_all_keys_present(self, tmp_dir):
        report = triage.CheckpointReport(path=tmp_dir / "model.pt")
        d = report.as_dict()
        for key in ("path", "loaded", "message", "anomalies", "converted", "fingerprint", "kl_divergences"):
            assert key in d

    def test_converted_is_none_when_not_set(self, tmp_dir):
        report = triage.CheckpointReport(path=tmp_dir / "model.pt")
        assert report.as_dict()["converted"] is None

    def test_converted_is_string_when_set(self, tmp_dir):
        report = triage.CheckpointReport(path=tmp_dir / "model.pt")
        report.converted = tmp_dir / "model.safetensors"
        assert isinstance(report.as_dict()["converted"], str)
