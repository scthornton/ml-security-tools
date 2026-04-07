#!/usr/bin/env python3
"""
Model inspection utilities for quick security triage of language and vision models.

The script performs three lightweight checks:
  1. Flag anomalous weight magnitudes.
  2. Register a forward hook to watch activations for suspicious variance.
  3. Evaluate a simple FGSM adversarial example (vision models only).

The defaults use small, openly available models so the script can run without
special credentials. Replace or extend `MODEL_SPECS` as needed for your setup.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoTokenizer,
    )
except ImportError as exc:  # pragma: no cover - communicated to user at runtime
    raise SystemExit(
        "The `transformers` package is required. Install it with `pip install transformers` and retry."
    ) from exc


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("model_inspection")


@dataclass(frozen=True)
class ModelSpec:
    """Minimal configuration for a model under inspection."""

    name: str
    modality: str  # "text" or "vision"
    epsilon: float = 0.01
    image_size: int = 224  # only used for vision models


MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec(name="distilgpt2", modality="text"),
    ModelSpec(name="google/vit-base-patch16-224", modality="vision"),
    # Replace with a local path to an intentionally corrupted artifact, if available.
    ModelSpec(name="your-poisoned-model", modality="vision"),
)


def first_tensor(obj: object) -> torch.Tensor | None:
    """Return the first tensor contained in a nested structure."""
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "logits"):
        return obj.logits
    if hasattr(obj, "last_hidden_state"):
        return obj.last_hidden_state
    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def check_suspicious_weights(model: nn.Module, threshold: float = 100.0) -> None:
    """Alert if any parameter exceeds the absolute threshold or contains NaNs."""
    flagged = False
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        with torch.no_grad():
            max_abs = torch.max(param.abs()).item()
            has_nan = torch.isnan(param).any().item()
        if has_nan or max_abs > threshold:
            flagged = True
            LOGGER.warning(
                "Suspicious weights detected in %s (max_abs=%.2f, contains_nan=%s)",
                name,
                max_abs,
                bool(has_nan),
            )
    if not flagged:
        LOGGER.info("No suspicious weights detected.")


def register_activation_watchdog(model: nn.Module, std_threshold: float = 10.0) -> torch.utils.hooks.RemovableHandle:
    """Monitor activations for extreme variance during inference."""

    def _watchdog(module: nn.Module, _: Iterable[torch.Tensor], output: object) -> None:
        tensor = first_tensor(output)
        if tensor is None:
            return
        with torch.no_grad():
            std_val = tensor.detach().float().std().item()
        if torch.isnan(torch.tensor(std_val)) or torch.isinf(torch.tensor(std_val)):
            LOGGER.warning("Activation contains NaN/Inf in %s", module.__class__.__name__)
        elif std_val > std_threshold:
            LOGGER.warning(
                "High activation variance detected in %s (std=%.2f)",
                module.__class__.__name__,
                std_val,
            )

    return model.register_forward_hook(_watchdog)


def fgsm_attack(model: nn.Module, inputs: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Single-step FGSM attack for vision models.

    The inputs are assumed to be in [-1, 1] with shape [batch, channels, height, width].
    """
    model.eval()
    attack_inputs = inputs.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    outputs = model(attack_inputs)
    logits = first_tensor(outputs)
    if logits is None:
        raise RuntimeError("Could not extract logits for FGSM attack.")
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    target = logits.argmax(dim=1)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    perturbation = epsilon * attack_inputs.grad.sign()
    adv_example = attack_inputs + perturbation
    return torch.clamp(adv_example.detach(), -1.0, 1.0)


def predict_class(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return class indices from the model's logits."""
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    logits = first_tensor(outputs)
    if logits is None:
        raise RuntimeError("Unable to extract logits for classification.")
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits.argmax(dim=1)


def inspect_text_model(spec: ModelSpec, allow_downloads: bool) -> None:
    """Run basic inspection for language models."""
    LOGGER.info("Inspecting text model: %s", spec.name)
    local_only = not allow_downloads
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.name, local_files_only=local_only)
        tokenizer = AutoTokenizer.from_pretrained(spec.name, local_files_only=local_only)
    except OSError as exc:
        LOGGER.error("Failed to load text model '%s': %s", spec.name, exc)
        return

    model.eval()
    check_suspicious_weights(model)
    hook = register_activation_watchdog(model)

    prompt = "Security scanning prompt."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    logits = first_tensor(outputs)
    if logits is None:
        LOGGER.warning("Model output did not expose logits.")
    else:
        LOGGER.info("Sampled last token id: %s", logits[:, -1, :].argmax(dim=-1).tolist())

    hook.remove()


def inspect_vision_model(spec: ModelSpec, allow_downloads: bool) -> None:
    """Run basic inspection for vision models."""
    LOGGER.info("Inspecting vision model: %s", spec.name)
    if spec.name == "your-poisoned-model":
        local_path = Path(spec.name)
        if not local_path.exists():
            LOGGER.warning(
                "Skipping '%s': provide a valid local path for your poisoned model demo.",
                spec.name,
            )
            return
    local_only = not allow_downloads
    try:
        model = AutoModelForImageClassification.from_pretrained(spec.name, local_files_only=local_only)
    except OSError as exc:
        LOGGER.error("Failed to load vision model '%s': %s", spec.name, exc)
        return

    model.eval()
    check_suspicious_weights(model)
    hook = register_activation_watchdog(model)

    clean_input = torch.randn(1, 3, spec.image_size, spec.image_size)
    clean_pred = predict_class(model, clean_input)

    try:
        adv_input = fgsm_attack(model, clean_input, spec.epsilon)
    except RuntimeError as exc:
        hook.remove()
        LOGGER.error("FGSM attack failed: %s", exc)
        return

    adv_pred = predict_class(model, adv_input)
    if not torch.equal(clean_pred, adv_pred):
        LOGGER.warning(
            "Model vulnerable to FGSM attack (epsilon=%.3f). clean=%s adv=%s",
            spec.epsilon,
            clean_pred.tolist(),
            adv_pred.tolist(),
        )
    else:
        LOGGER.info("Model prediction stable under FGSM attack (epsilon=%.3f).", spec.epsilon)

    hook.remove()


def inspect_model(spec: ModelSpec, allow_downloads: bool) -> None:
    """Dispatch inspection routine based on modality."""
    if spec.modality == "text":
        inspect_text_model(spec, allow_downloads)
    elif spec.modality == "vision":
        inspect_vision_model(spec, allow_downloads)
    else:
        LOGGER.error("Unknown modality '%s' for %s", spec.modality, spec.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect models for basic security issues.")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Permit huggingface to download weights that are not cached locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in MODEL_SPECS:
        inspect_model(spec, allow_downloads=args.allow_downloads)


if __name__ == "__main__":
    main()
