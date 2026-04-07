#!/usr/bin/env python3
"""
Adversarial regression harness for tracking robustness drift.

Supports multiple attack families (FGSM, PGD, CW) and mixed precision to
surface vulnerabilities introduced by quantization or compilation tweaks.
Maintains a JSON baseline keyed by model name, attack type, and epsilon so
regressions are easy to spot across versions.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

try:
    from torchvision import transforms  # type: ignore
    from torchvision.datasets import FakeData  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FakeData = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("fgsm_regression_harness")

ATTACK_CHOICES = ("fgsm", "pgd", "cw")
PRECISION_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_module(script_path: str):
    path = Path(script_path)
    if path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(script_path)


def resolve_model(script: str, factory: str) -> nn.Module:
    module = load_module(script)
    builder = getattr(module, factory, None)
    if builder is None:
        raise AttributeError(f"{factory} not found in {script}")
    model = builder()
    if not isinstance(model, nn.Module):
        raise TypeError(f"{factory} must return a torch.nn.Module")
    return model


def create_dataset(
    samples: int,
    input_shape: Sequence[int],
    num_classes: int,
    use_fake: bool,
) -> Dataset:
    if use_fake and FakeData is not None and transforms is not None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
        return FakeData(
            size=samples,
            image_size=tuple(input_shape[1:]),
            num_classes=num_classes,
            transform=transform,
        )
    # Fallback synthetic tensor dataset in [-1, 1]
    data = torch.randn(samples, *input_shape[1:]).clamp_(-1.0, 1.0)
    labels = torch.randint(0, num_classes, (samples,))
    return TensorDataset(data, labels)


def clamp_like(inputs: torch.Tensor, reference: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.max(torch.min(inputs, reference + epsilon), reference - epsilon)


def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    model.eval()
    inputs_adv = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs_adv)
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(0)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()
    adv = inputs_adv + epsilon * inputs_adv.grad.sign()
    return adv.detach().clamp_(-1.0, 1.0)


def pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    model.eval()
    adv = inputs.clone().detach() + torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    adv = adv.clamp(-1.0, 1.0)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        outputs = model(adv)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = clamp_like(adv, inputs, epsilon)
            adv = adv.clamp(-1.0, 1.0)
    return adv.detach()


def cw_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    steps: int,
    lr: float,
    confidence: float,
) -> torch.Tensor:
    """
    Simplified L∞-bounded Carlini & Wagner-style attack.
    Minimises margin between target logit and highest non-target logit.
    """
    model.eval()
    adv = inputs.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([adv], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(adv)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        target_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, targets.unsqueeze(1), float("-inf"))
        max_other = masked.max(dim=1)[0]
        loss = torch.relu(target_logit - max_other + confidence).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            adv.data = clamp_like(adv, inputs, epsilon)
            adv.data.clamp_(-1.0, 1.0)
    return adv.detach()


def ensure_precision(model: nn.Module, device: torch.device, precision: str) -> torch.dtype:
    dtype = PRECISION_MAP.get(precision.lower())
    if dtype is None:
        raise ValueError(f"Unsupported precision: {precision}")
    if dtype != torch.float32 and device.type == "cpu":
        LOGGER.warning("Requested %s precision on CPU; falling back to float32.", precision)
        dtype = torch.float32
    model.to(device=device, dtype=dtype)
    return dtype


def prepare_inputs(inputs: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    data = inputs.to(device=device, dtype=dtype)
    return data.clamp_(-1.0, 1.0)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    attacks: Iterable[str],
    epsilon: float,
    device: torch.device,
    dtype: torch.dtype,
    pgd_steps: int,
    pgd_alpha: float,
    cw_steps: int,
    cw_lr: float,
    cw_confidence: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    clean_correct = 0
    total = 0
    attack_correct = {attack: 0 for attack in attacks}

    model.to(device)
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = prepare_inputs(inputs, device, dtype)
        targets = targets.to(device=device)

        with torch.no_grad():
            outputs = model(inputs)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        clean_pred = outputs.argmax(dim=1)
        clean_correct += (clean_pred == targets).sum().item()
        total += targets.size(0)

        for attack in attacks:
            if attack == "fgsm":
                adv_inputs = fgsm_attack(model, inputs, targets, epsilon)
            elif attack == "pgd":
                adv_inputs = pgd_attack(model, inputs, targets, epsilon, pgd_alpha, pgd_steps)
            elif attack == "cw":
                adv_inputs = cw_attack(model, inputs, targets, epsilon, cw_steps, cw_lr, cw_confidence)
            else:  # pragma: no cover - guarded by argparse choices
                raise ValueError(f"Unknown attack {attack}")

            adv_inputs = adv_inputs.to(device=device, dtype=dtype)
            with torch.no_grad():
                adv_outputs = model(adv_inputs)
            if adv_outputs.ndim == 1:
                adv_outputs = adv_outputs.unsqueeze(0)
            adv_pred = adv_outputs.argmax(dim=1)
            attack_correct[attack] += (adv_pred == targets).sum().item()

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    results = {"clean": clean_correct / total if total else 0.0}
    for attack, correct in attack_correct.items():
        results[f"{attack}_epsilon={epsilon}"] = correct / total if total else 0.0
    return results


def load_baseline(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(k): dict(v) for k, v in data.items()}


def save_baseline(path: Path, baseline: dict[str, dict[str, dict[str, float]]]) -> None:
    path.write_text(json.dumps(baseline, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness regression.")
    parser.add_argument("--model-script", required=True, help="Python module or path exposing the model factory.")
    parser.add_argument("--factory", default="create_model", help="Factory function name (default: create_model).")
    parser.add_argument("--input-shape", required=True, help="Input shape including batch, e.g., 1,3,224,224.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of output classes.")
    parser.add_argument("--samples", type=int, default=128, help="Number of samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Perturbation bound for FGSM/PGD/CW.")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["fgsm"],
        choices=ATTACK_CHOICES,
        help="Attack types to run (default: fgsm).",
    )
    parser.add_argument(
        "--pgd-steps", type=int, default=20, help="Number of PGD steps (default: 20, per Madry et al.)."
    )
    parser.add_argument("--pgd-alpha", type=float, default=0.005, help="PGD step size.")
    parser.add_argument("--cw-steps", type=int, default=10, help="Number of CW optimization steps.")
    parser.add_argument("--cw-lr", type=float, default=0.01, help="Learning rate for CW optimizer.")
    parser.add_argument("--cw-confidence", type=float, default=0.0, help="CW attack confidence margin.")
    parser.add_argument(
        "--baseline-file", type=Path, default=Path("fgsm_baselines.json"), help="Path to baseline JSON file."
    )
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline with current metrics.")
    parser.add_argument("--use-fake-data", action="store_true", help="Use torchvision FakeData if available.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit the number of batches processed.")
    parser.add_argument("--device", default="cpu", help="Torch device string (default: cpu).")
    parser.add_argument(
        "--precision",
        default="float32",
        choices=tuple(PRECISION_MAP.keys()),
        help="Computation precision for evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attacks = [attack.lower() for attack in args.attacks]

    input_shape = tuple(int(dim) for dim in args.input_shape.split(","))
    if len(input_shape) < 2:
        raise ValueError("Input shape must include batch dimension and feature dims.")

    model = resolve_model(args.model_script, args.factory)
    device = torch.device(args.device)
    dtype = ensure_precision(model, device, args.precision)

    dataset = create_dataset(
        samples=args.samples,
        input_shape=input_shape,
        num_classes=args.num_classes,
        use_fake=args.use_fake_data,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        attacks=attacks,
        epsilon=args.epsilon,
        device=device,
        dtype=dtype,
        pgd_steps=args.pgd_steps,
        pgd_alpha=args.pgd_alpha,
        cw_steps=args.cw_steps,
        cw_lr=args.cw_lr,
        cw_confidence=args.cw_confidence,
        max_batches=args.max_batches,
    )

    LOGGER.info("Clean accuracy: %.4f", results["clean"])
    for attack in attacks:
        key = f"{attack}_epsilon={args.epsilon}"
        LOGGER.info("%s accuracy: %.4f", attack.upper(), results[key])

    baseline = load_baseline(args.baseline_file)
    model_key = f"{args.model_script}:{args.factory}"
    eps_key = f"epsilon={args.epsilon}"

    for attack in attacks:
        attack_key = attack
        prev = baseline.get(model_key, {}).get(attack_key, {}).get(eps_key)
        current = results[f"{attack}_epsilon={args.epsilon}"]
        if prev is not None:
            drift = current - prev
            LOGGER.info("Baseline %s accuracy: %.4f (drift = %.4f)", attack_key.upper(), prev, drift)
            if abs(drift) > 0.05:
                LOGGER.warning(
                    "%s accuracy drift exceeds 5%% threshold. Investigate robustness regression.",
                    attack_key.upper(),
                )
        else:
            LOGGER.info("No baseline recorded for %s @ %s.", attack_key.upper(), eps_key)

    if args.update_baseline:
        baseline.setdefault(model_key, {})
        for attack in attacks:
            baseline[model_key].setdefault(attack, {})
            baseline[model_key][attack][eps_key] = results[f"{attack}_epsilon={args.epsilon}"]
        save_baseline(args.baseline_file, baseline)
        LOGGER.info("Baseline updated at %s", args.baseline_file)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
