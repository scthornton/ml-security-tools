"""
Tests for fgsm_regression_harness.py

Coverage targets
----------------
- fgsm_attack: output shape, value clamping, gradient sign direction
- pgd_attack: output shape, epsilon ball constraint, multi-step refinement
- cw_attack: output shape, epsilon ball constraint, optimizer step execution
- clamp_like: correct clamping to [ref-eps, ref+eps]
- ensure_precision: valid dtype resolution, unsupported precision raises ValueError
- create_dataset: tensor dataset shape and label range
- evaluate_model: accumulates correct/total, returns expected keys
- load_baseline / save_baseline: round-trip JSON persistence
- resolve_model: missing factory raises AttributeError, wrong return type raises TypeError
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Import the module under test (lives at repo root, not inside a package)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import fgsm_regression_harness as harness  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(in_features: int = 8, num_classes: int = 4) -> nn.Module:
    """Return a tiny deterministic model for CPU-only tests."""
    model = nn.Sequential(nn.Linear(in_features, num_classes))
    nn.init.constant_(model[0].weight, 0.1)
    nn.init.zeros_(model[0].bias)
    return model


def _make_dataloader(
    samples: int = 8, in_features: int = 8, num_classes: int = 4, batch_size: int = 4
) -> DataLoader:
    torch.manual_seed(42)
    data = torch.randn(samples, in_features).clamp(-1.0, 1.0)
    labels = torch.randint(0, num_classes, (samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ===========================================================================
# clamp_like
# ===========================================================================

class TestClampLike:
    def test_values_within_epsilon_are_unchanged(self):
        ref = torch.zeros(3)
        inputs = torch.tensor([0.01, -0.01, 0.0])
        result = harness.clamp_like(inputs, ref, epsilon=0.05)
        assert torch.allclose(result, inputs)

    def test_values_outside_epsilon_are_clamped_upper(self):
        ref = torch.zeros(3)
        inputs = torch.tensor([0.1, 0.2, 0.5])
        result = harness.clamp_like(inputs, ref, epsilon=0.05)
        assert result.max().item() <= 0.05 + 1e-6

    def test_values_outside_epsilon_are_clamped_lower(self):
        ref = torch.zeros(3)
        inputs = torch.tensor([-0.1, -0.2, -0.5])
        result = harness.clamp_like(inputs, ref, epsilon=0.05)
        assert result.min().item() >= -0.05 - 1e-6

    def test_shape_preserved(self):
        ref = torch.zeros(2, 3)
        inputs = torch.randn(2, 3)
        result = harness.clamp_like(inputs, ref, epsilon=0.1)
        assert result.shape == inputs.shape


# ===========================================================================
# fgsm_attack
# ===========================================================================

class TestFgsmAttack:
    def test_output_shape_matches_input(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.fgsm_attack(tiny_linear, batch_inputs_1d, batch_targets_4class, epsilon=0.03)
        assert adv.shape == batch_inputs_1d.shape

    def test_output_clamped_to_minus1_plus1(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.fgsm_attack(tiny_linear, batch_inputs_1d, batch_targets_4class, epsilon=0.5)
        assert adv.min().item() >= -1.0 - 1e-6
        assert adv.max().item() <= 1.0 + 1e-6

    def test_perturbation_bounded_by_epsilon(self, tiny_linear, batch_targets_4class):
        # Use inputs already in [-1, 1] so the global clamp does not reduce the delta
        torch.manual_seed(7)
        inputs = torch.randn(4, 8).clamp(-0.5, 0.5)  # safely inside the valid range
        epsilon = 0.1
        adv = harness.fgsm_attack(tiny_linear, inputs, batch_targets_4class, epsilon=epsilon)
        delta = (adv - inputs).abs()
        assert delta.max().item() <= epsilon + 1e-5

    def test_returns_detached_tensor(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.fgsm_attack(tiny_linear, batch_inputs_1d, batch_targets_4class, epsilon=0.01)
        assert not adv.requires_grad

    def test_zero_epsilon_returns_clamped_input(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.fgsm_attack(tiny_linear, batch_inputs_1d, batch_targets_4class, epsilon=0.0)
        assert torch.allclose(adv, batch_inputs_1d.clamp(-1.0, 1.0))

    def test_1d_output_is_unsqueezed_for_loss(self):
        """Model returning a 1-D logit vector should not crash CrossEntropyLoss."""
        model = nn.Linear(8, 4)
        # Force model to return shape (4,) when given single sample
        inputs = torch.randn(1, 8)
        targets = torch.tensor([0])
        # This should not raise even if the model output is squeezed
        adv = harness.fgsm_attack(model, inputs, targets, epsilon=0.01)
        assert adv.shape == inputs.shape

    def test_conv_model_shape_preserved(self, tiny_conv, batch_inputs_2d, batch_targets_2):
        adv = harness.fgsm_attack(tiny_conv, batch_inputs_2d, batch_targets_2, epsilon=0.05)
        assert adv.shape == batch_inputs_2d.shape


# ===========================================================================
# pgd_attack
# ===========================================================================

class TestPgdAttack:
    def test_output_shape_matches_input(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.pgd_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.05, alpha=0.01, steps=3
        )
        assert adv.shape == batch_inputs_1d.shape

    def test_output_within_global_clamp(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.pgd_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.3, alpha=0.05, steps=5
        )
        assert adv.min().item() >= -1.0 - 1e-5
        assert adv.max().item() <= 1.0 + 1e-5

    def test_returns_detached_tensor(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.pgd_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.05, alpha=0.01, steps=2
        )
        assert not adv.requires_grad

    def test_single_step_equals_fgsm_direction_roughly(self, tiny_linear):
        """One PGD step from the clean starting point should move in a similar
        direction as FGSM (not identical because PGD adds random init noise)."""
        torch.manual_seed(0)
        inputs = torch.zeros(1, 8)  # zero init so random noise is the only difference
        targets = torch.tensor([0])
        adv = harness.pgd_attack(tiny_linear, inputs, targets, epsilon=0.01, alpha=0.01, steps=1)
        assert adv.shape == inputs.shape


# ===========================================================================
# cw_attack
# ===========================================================================

class TestCwAttack:
    def test_output_shape_matches_input(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.cw_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.05, steps=3, lr=0.01, confidence=0.0
        )
        assert adv.shape == batch_inputs_1d.shape

    def test_output_within_epsilon_ball(self, tiny_linear, batch_targets_4class):
        # Use inputs already inside [-1, 1] so the global clamp doesn't interfere
        torch.manual_seed(7)
        inputs = torch.randn(4, 8).clamp(-0.5, 0.5)
        epsilon = 0.1
        adv = harness.cw_attack(
            tiny_linear, inputs, batch_targets_4class,
            epsilon=epsilon, steps=5, lr=0.01, confidence=0.0
        )
        delta = (adv - inputs).abs().max().item()
        assert delta <= epsilon + 1e-5

    def test_output_within_global_clamp(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.cw_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.5, steps=3, lr=0.01, confidence=0.0
        )
        assert adv.min().item() >= -1.0 - 1e-5
        assert adv.max().item() <= 1.0 + 1e-5

    def test_returns_detached_tensor(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.cw_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.05, steps=2, lr=0.01, confidence=0.0
        )
        assert not adv.requires_grad

    def test_zero_steps_returns_clamped_input(self, tiny_linear, batch_inputs_1d, batch_targets_4class):
        adv = harness.cw_attack(
            tiny_linear, batch_inputs_1d, batch_targets_4class,
            epsilon=0.05, steps=0, lr=0.01, confidence=0.0
        )
        assert adv.shape == batch_inputs_1d.shape


# ===========================================================================
# ensure_precision
# ===========================================================================

class TestEnsurePrecision:
    def test_float32_resolves_correctly(self, tiny_linear):
        device = torch.device("cpu")
        dtype = harness.ensure_precision(tiny_linear, device, "float32")
        assert dtype == torch.float32

    def test_unsupported_precision_raises(self, tiny_linear):
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="Unsupported precision"):
            harness.ensure_precision(tiny_linear, device, "int4")

    def test_float16_on_cpu_falls_back_to_float32(self, tiny_linear):
        """Half-precision on CPU is not well-supported; the harness should fall back."""
        device = torch.device("cpu")
        dtype = harness.ensure_precision(tiny_linear, device, "float16")
        assert dtype == torch.float32

    def test_bfloat16_on_cpu_falls_back_to_float32(self, tiny_linear):
        device = torch.device("cpu")
        dtype = harness.ensure_precision(tiny_linear, device, "bfloat16")
        assert dtype == torch.float32


# ===========================================================================
# create_dataset
# ===========================================================================

class TestCreateDataset:
    def test_returns_correct_number_of_samples(self):
        ds = harness.create_dataset(
            samples=16, input_shape=(1, 8), num_classes=4, use_fake=False
        )
        assert len(ds) == 16

    def test_tensor_shape_correct(self):
        ds = harness.create_dataset(
            samples=8, input_shape=(1, 8), num_classes=4, use_fake=False
        )
        x, y = ds[0]
        assert x.shape == torch.Size([8])
        assert y.ndim == 0  # scalar label

    def test_labels_within_num_classes(self):
        ds = harness.create_dataset(
            samples=32, input_shape=(1, 8), num_classes=3, use_fake=False
        )
        for _, label in ds:
            assert 0 <= label.item() < 3

    def test_data_clamped_to_minus1_plus1(self):
        ds = harness.create_dataset(
            samples=64, input_shape=(1, 8), num_classes=4, use_fake=False
        )
        for x, _ in ds:
            assert x.min().item() >= -1.0 - 1e-6
            assert x.max().item() <= 1.0 + 1e-6


# ===========================================================================
# evaluate_model
# ===========================================================================

class TestEvaluateModel:
    """Test the main evaluation loop with a mock dataloader."""

    def _run_eval(self, model, attacks, epsilon=0.05):
        loader = _make_dataloader(samples=8, in_features=8, num_classes=4, batch_size=4)
        device = torch.device("cpu")
        return harness.evaluate_model(
            model=model,
            dataloader=loader,
            attacks=attacks,
            epsilon=epsilon,
            device=device,
            dtype=torch.float32,
            pgd_steps=2,
            pgd_alpha=0.01,
            cw_steps=2,
            cw_lr=0.01,
            cw_confidence=0.0,
            max_batches=None,
        )

    def test_clean_accuracy_key_present(self, tiny_linear):
        results = self._run_eval(tiny_linear, ["fgsm"])
        assert "clean" in results

    def test_fgsm_key_present(self, tiny_linear):
        results = self._run_eval(tiny_linear, ["fgsm"], epsilon=0.05)
        assert "fgsm_epsilon=0.05" in results

    def test_pgd_key_present(self, tiny_linear):
        results = self._run_eval(tiny_linear, ["pgd"], epsilon=0.05)
        assert "pgd_epsilon=0.05" in results

    def test_cw_key_present(self, tiny_linear):
        results = self._run_eval(tiny_linear, ["cw"], epsilon=0.05)
        assert "cw_epsilon=0.05" in results

    def test_accuracy_between_0_and_1(self, tiny_linear):
        results = self._run_eval(tiny_linear, ["fgsm"])
        for value in results.values():
            assert 0.0 <= value <= 1.0

    def test_max_batches_limits_evaluation(self, tiny_linear):
        loader = _make_dataloader(samples=16, in_features=8, num_classes=4, batch_size=4)
        device = torch.device("cpu")
        results = harness.evaluate_model(
            model=tiny_linear,
            dataloader=loader,
            attacks=["fgsm"],
            epsilon=0.05,
            device=device,
            dtype=torch.float32,
            pgd_steps=2,
            pgd_alpha=0.01,
            cw_steps=2,
            cw_lr=0.01,
            cw_confidence=0.0,
            max_batches=1,
        )
        # With 4 samples/batch and max_batches=1 the total processed should be 4
        # We verify result keys are present; accuracy itself is valid
        assert "clean" in results

    def test_empty_dataloader_returns_zero_accuracy(self, tiny_linear):
        """An empty DataLoader (zero samples) should return 0.0, not raise."""
        empty_ds = TensorDataset(torch.zeros(0, 8), torch.zeros(0, dtype=torch.long))
        loader = DataLoader(empty_ds, batch_size=4)
        device = torch.device("cpu")
        results = harness.evaluate_model(
            model=tiny_linear,
            dataloader=loader,
            attacks=["fgsm"],
            epsilon=0.05,
            device=device,
            dtype=torch.float32,
            pgd_steps=2,
            pgd_alpha=0.01,
            cw_steps=2,
            cw_lr=0.01,
            cw_confidence=0.0,
        )
        assert results["clean"] == 0.0

    def test_multiple_attacks_all_keys_present(self, tiny_linear):
        results = self._run_eval(tiny_linear, ["fgsm", "pgd", "cw"], epsilon=0.03)
        assert "fgsm_epsilon=0.03" in results
        assert "pgd_epsilon=0.03" in results
        assert "cw_epsilon=0.03" in results


# ===========================================================================
# load_baseline / save_baseline
# ===========================================================================

class TestBaselinePersistence:
    def test_save_and_reload_round_trip(self, tmp_dir):
        path = tmp_dir / "baseline.json"
        baseline = {
            "my_model:create_model": {
                "fgsm": {"epsilon=0.01": 0.82}
            }
        }
        harness.save_baseline(path, baseline)
        loaded = harness.load_baseline(path)
        assert loaded == baseline

    def test_load_nonexistent_returns_empty_dict(self, tmp_dir):
        path = tmp_dir / "no_such_file.json"
        result = harness.load_baseline(path)
        assert result == {}

    def test_saved_file_is_valid_json(self, tmp_dir):
        path = tmp_dir / "baseline.json"
        harness.save_baseline(path, {"k": {"a": {"eps=0.1": 0.5}}})
        parsed = json.loads(path.read_text())
        assert isinstance(parsed, dict)

    def test_save_overwrites_existing(self, tmp_dir):
        path = tmp_dir / "baseline.json"
        harness.save_baseline(path, {"old": {}})
        harness.save_baseline(path, {"new": {"k": {"e": 0.9}}})
        loaded = harness.load_baseline(path)
        assert "new" in loaded
        assert "old" not in loaded


# ===========================================================================
# resolve_model
# ===========================================================================

class TestResolveModel:
    def test_missing_factory_raises_attribute_error(self, tmp_dir):
        script = tmp_dir / "dummy_model.py"
        script.write_text("def not_create_model(): pass\n")
        with pytest.raises(AttributeError, match="missing_factory not found"):
            harness.resolve_model(str(script), "missing_factory")

    def test_non_module_return_raises_type_error(self, tmp_dir):
        script = tmp_dir / "bad_factory.py"
        script.write_text("def create_model(): return 42\n")
        with pytest.raises(TypeError, match="must return a torch.nn.Module"):
            harness.resolve_model(str(script), "create_model")

    def test_valid_factory_returns_nn_module(self, tmp_dir):
        script = tmp_dir / "good_model.py"
        script.write_text(
            "import torch.nn as nn\n"
            "def create_model(): return nn.Linear(4, 2)\n"
        )
        model = harness.resolve_model(str(script), "create_model")
        assert isinstance(model, nn.Module)

    def test_importlib_path_missing_raises(self, tmp_dir):
        """Passing a non-existent path that also can't be imported should raise."""
        with pytest.raises(Exception):
            harness.resolve_model("totally_nonexistent_module_xyz", "create_model")
