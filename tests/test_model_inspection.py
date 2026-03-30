"""
Tests for model-inspection.py

Coverage targets
----------------
- first_tensor: unwraps Tensor, .logits, .last_hidden_state, nested list/tuple, returns None
- check_suspicious_weights: logs warning for NaN, extreme values; logs info for clean model
- register_activation_watchdog: hook fires, high std triggers warning, NaN/Inf triggers warning
- fgsm_attack (model_inspection version): output shape, clamping, RuntimeError when no logits
- predict_class: correct argmax, shape (batch,), RuntimeError when no logits
- inspect_model: dispatches correctly, logs ERROR for unknown modality
- inspect_text_model / inspect_vision_model: short-circuit on OSError (no downloads)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The file is named with a hyphen which prevents normal import; use importlib.
# The module must be registered in sys.modules BEFORE exec_module so that
# Python's dataclass decorator can resolve the module's __dict__ by name.
_spec = importlib.util.spec_from_file_location(
    "model_inspection", REPO_ROOT / "model-inspection.py"
)
model_inspection = importlib.util.module_from_spec(_spec)
sys.modules["model_inspection"] = model_inspection
_spec.loader.exec_module(model_inspection)

ModelSpec = model_inspection.ModelSpec
first_tensor = model_inspection.first_tensor
check_suspicious_weights = model_inspection.check_suspicious_weights
register_activation_watchdog = model_inspection.register_activation_watchdog
fgsm_attack = model_inspection.fgsm_attack
predict_class = model_inspection.predict_class
inspect_model = model_inspection.inspect_model
inspect_text_model = model_inspection.inspect_text_model
inspect_vision_model = model_inspection.inspect_vision_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyLinear(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.constant_(self.fc.weight, 0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ===========================================================================
# first_tensor
# ===========================================================================

class TestFirstTensor:
    def test_tensor_returns_self(self):
        t = torch.tensor([1.0, 2.0])
        assert first_tensor(t) is t

    def test_object_with_logits_attribute(self):
        obj = SimpleNamespace(logits=torch.ones(3))
        result = first_tensor(obj)
        assert torch.equal(result, torch.ones(3))

    def test_object_with_last_hidden_state_attribute(self):
        obj = SimpleNamespace(last_hidden_state=torch.zeros(2, 5))
        result = first_tensor(obj)
        assert result.shape == (2, 5)

    def test_nested_list_returns_first_tensor(self):
        inner = torch.tensor([7.0])
        result = first_tensor(["not a tensor", [inner]])
        assert result is inner

    def test_nested_tuple_returns_first_tensor(self):
        inner = torch.zeros(1)
        result = first_tensor((None, (inner,)))
        assert result is inner

    def test_none_input_returns_none(self):
        assert first_tensor(None) is None

    def test_plain_string_returns_none(self):
        assert first_tensor("hello") is None

    def test_empty_list_returns_none(self):
        assert first_tensor([]) is None


# ===========================================================================
# check_suspicious_weights
# ===========================================================================

class TestCheckSuspiciousWeights:
    def test_clean_model_does_not_warn(self, caplog):
        model = TinyLinear()
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            check_suspicious_weights(model, threshold=100.0)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0

    def test_extreme_weight_triggers_warning(self, caplog):
        model = TinyLinear()
        with torch.no_grad():
            model.fc.weight.fill_(500.0)
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            check_suspicious_weights(model, threshold=100.0)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert any("Suspicious" in r.message for r in warnings)

    def test_nan_weight_triggers_warning(self, caplog):
        model = TinyLinear()
        with torch.no_grad():
            model.fc.weight[0, 0] = float("nan")
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            check_suspicious_weights(model, threshold=100.0)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1

    def test_integer_parameters_skipped(self, caplog):
        """Non-floating-point parameters (e.g. embedding indices) must not crash."""
        class ModelWithIntParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("ids", torch.tensor([0, 1, 2], dtype=torch.long))
                self.linear = nn.Linear(4, 2)
        model = ModelWithIntParam()
        # Should not raise; integer buffers are not parameters, but verify robustness
        check_suspicious_weights(model, threshold=100.0)

    def test_threshold_boundary_exactly_at_threshold_does_not_warn(self, caplog):
        model = TinyLinear()
        with torch.no_grad():
            model.fc.weight.fill_(100.0)
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            check_suspicious_weights(model, threshold=100.0)
        # max_abs == threshold, not strictly greater; no warning expected
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0


# ===========================================================================
# register_activation_watchdog
# ===========================================================================

class TestActivationWatchdog:
    def test_hook_is_removable(self):
        model = TinyLinear()
        handle = register_activation_watchdog(model, std_threshold=10.0)
        # Removing should not raise
        handle.remove()

    def test_normal_activation_does_not_warn(self, caplog):
        model = TinyLinear()
        handle = register_activation_watchdog(model, std_threshold=10.0)
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            with torch.no_grad():
                model(torch.randn(2, 4))
        handle.remove()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0

    def test_high_variance_activation_warns(self, caplog):
        """Force activations with std >> threshold."""
        model = TinyLinear()
        with torch.no_grad():
            # Set weights large enough that outputs have high variance
            model.fc.weight.fill_(1e4)
            model.fc.bias.fill_(0.0)
        handle = register_activation_watchdog(model, std_threshold=1.0)
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            with torch.no_grad():
                model(torch.randn(4, 4))
        handle.remove()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1

    def test_hook_on_module_with_nan_output_warns(self, caplog):
        """A module that outputs NaN should trigger the watchdog."""
        class NanModule(nn.Module):
            def forward(self, x):
                return torch.full_like(x, float("nan"))

        model = NanModule()
        handle = register_activation_watchdog(model, std_threshold=10.0)
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            with torch.no_grad():
                model(torch.randn(2, 4))
        handle.remove()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1


# ===========================================================================
# fgsm_attack (model_inspection version — untargeted, uses argmax as target)
# ===========================================================================

class TestFgsmAttackInspection:
    def test_output_shape_matches_input(self):
        model = TinyLinear()
        inputs = torch.randn(1, 4)
        adv = fgsm_attack(model, inputs, epsilon=0.01)
        assert adv.shape == inputs.shape

    def test_output_clamped_minus1_plus1(self):
        model = TinyLinear()
        inputs = torch.randn(2, 4) * 10  # deliberately out of [-1,1]
        adv = fgsm_attack(model, inputs, epsilon=0.5)
        assert adv.min().item() >= -1.0 - 1e-6
        assert adv.max().item() <= 1.0 + 1e-6

    def test_returns_detached_tensor(self):
        model = TinyLinear()
        inputs = torch.randn(1, 4)
        adv = fgsm_attack(model, inputs, epsilon=0.01)
        assert not adv.requires_grad

    def test_no_logits_raises_runtime_error(self):
        """A model that returns None for logits should raise RuntimeError."""
        class NoLogitsModel(nn.Module):
            def forward(self, x):
                return None

        model = NoLogitsModel()
        # first_tensor(None) returns None -> RuntimeError
        with pytest.raises(RuntimeError, match="Could not extract logits"):
            fgsm_attack(model, torch.randn(1, 4), epsilon=0.01)


# ===========================================================================
# predict_class
# ===========================================================================

class TestPredictClass:
    def test_output_shape_is_batch_size(self):
        model = TinyLinear()
        inputs = torch.randn(4, 4)
        preds = predict_class(model, inputs)
        assert preds.shape == (4,)

    def test_predictions_are_valid_class_indices(self):
        model = TinyLinear(num_classes=3)
        inputs = torch.randn(8, 4)
        preds = predict_class(model, inputs)
        assert preds.min().item() >= 0
        assert preds.max().item() < 3

    def test_no_logits_raises_runtime_error(self):
        class NoLogitsModel(nn.Module):
            def forward(self, x):
                return []  # empty list -> first_tensor returns None

        model = NoLogitsModel()
        with pytest.raises(RuntimeError, match="Unable to extract logits"):
            predict_class(model, torch.randn(1, 4))


# ===========================================================================
# inspect_model dispatch
# ===========================================================================

class TestInspectModelDispatch:
    def test_unknown_modality_logs_error(self, caplog):
        spec = ModelSpec(name="any", modality="audio")
        with caplog.at_level(logging.ERROR, logger="model_inspection"):
            inspect_model(spec, allow_downloads=False)
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("Unknown modality" in r.message for r in errors)

    def test_text_modality_calls_inspect_text(self):
        spec = ModelSpec(name="distilgpt2", modality="text")
        with patch.object(model_inspection, "inspect_text_model") as mock_fn:
            inspect_model(spec, allow_downloads=False)
            mock_fn.assert_called_once_with(spec, False)

    def test_vision_modality_calls_inspect_vision(self):
        spec = ModelSpec(name="google/vit-base-patch16-224", modality="vision")
        with patch.object(model_inspection, "inspect_vision_model") as mock_fn:
            inspect_model(spec, allow_downloads=False)
            mock_fn.assert_called_once_with(spec, False)


# ===========================================================================
# inspect_text_model — short-circuit on load failure (no downloads)
# ===========================================================================

class TestInspectTextModel:
    def test_oserror_on_load_logs_error_and_returns(self, caplog):
        """When local_files_only=True and model is not cached, OSError is caught."""
        spec = ModelSpec(name="non_existent_model_xyz_123", modality="text")
        with caplog.at_level(logging.ERROR, logger="model_inspection"):
            inspect_text_model(spec, allow_downloads=False)
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("Failed to load text model" in r.message for r in errors)

    def test_successful_load_runs_weight_check(self):
        """With a mocked AutoModelForCausalLM, weight checks and hook run without error."""
        spec = ModelSpec(name="tiny-mock-gpt", modality="text")
        mock_model = TinyLinear()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.ones(1, 4, dtype=torch.long)}

        fake_output = SimpleNamespace(logits=torch.randn(1, 4, 3))

        with patch.object(model_inspection, "AutoModelForCausalLM") as mock_causal, \
             patch.object(model_inspection, "AutoTokenizer") as mock_tok:
            mock_causal.from_pretrained.return_value = mock_model
            mock_tok.from_pretrained.return_value = mock_tokenizer
            # Make model(**inputs) work by patching forward
            mock_model.forward = MagicMock(return_value=fake_output)
            # Should not raise
            inspect_text_model(spec, allow_downloads=False)


# ===========================================================================
# inspect_vision_model — short-circuit on load failure
# ===========================================================================

class TestInspectVisionModel:
    def test_oserror_on_load_logs_error_and_returns(self, caplog):
        spec = ModelSpec(name="non_existent_vision_xyz_123", modality="vision")
        with caplog.at_level(logging.ERROR, logger="model_inspection"):
            inspect_vision_model(spec, allow_downloads=False)
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("Failed to load vision model" in r.message for r in errors)

    def test_poisoned_model_placeholder_is_skipped_when_path_missing(self, caplog):
        """The special 'your-poisoned-model' name must be skipped gracefully."""
        spec = ModelSpec(name="your-poisoned-model", modality="vision")
        with caplog.at_level(logging.WARNING, logger="model_inspection"):
            inspect_vision_model(spec, allow_downloads=False)
        # Must not raise; should log a warning about skipping
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Skipping" in r.message for r in warnings)

    def test_fgsm_runtime_error_is_caught_and_logged(self, caplog):
        """If fgsm_attack raises (e.g. no logits), the error is logged, not propagated.

        The inspect_vision_model function calls predict_class first (which needs logits)
        and then fgsm_attack.  We use a model whose first call returns valid logits so
        predict_class succeeds, but whose second call returns None so fgsm_attack raises.
        """
        spec = ModelSpec(name="mock-vision", modality="vision", epsilon=0.01, image_size=8)

        call_count = {"n": 0}

        class FailOnSecondCallModel(nn.Module):
            def forward(self, x):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First call: predict_class — return valid logits
                    return torch.randn(x.shape[0], 4)
                # Second call: fgsm_attack needs grad — return None to trigger RuntimeError
                return None

        mock_model = FailOnSecondCallModel()

        with patch.object(model_inspection, "AutoModelForImageClassification") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_model
            with caplog.at_level(logging.ERROR, logger="model_inspection"):
                inspect_vision_model(spec, allow_downloads=False)
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("FGSM attack failed" in r.message for r in errors)
