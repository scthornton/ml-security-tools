#!/usr/bin/env python3
"""
Triage PyTorch checkpoints for common security issues and distribution drift.

Capabilities
------------
1. Recursively locate checkpoint files (.pt/.pth/.bin/.ckpt).
2. Load using `torch.load(weights_only=True)` when supported to mitigate pickle exploits.
3. Flag anomalous weight magnitudes, NaNs/Infs, and unexpected parameter insertions.
4. Generate histogram fingerprints for tensors and compare against a reference using KL divergence.
5. Optionally emit `safetensors` alongside original checkpoints and persist fingerprints for reuse.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

try:
    from safetensors.torch import save_file as save_safetensor
except ImportError:  # pragma: no cover -- optional dependency
    save_safetensor = None  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("torch_checkpoint_triage")


CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".bin", ".ckpt"}


def supports_weights_only() -> bool:
    """Return True if torch.load accepts the weights_only kwarg."""
    signature = inspect.signature(torch.load)
    return "weights_only" in signature.parameters


@dataclass
class CheckpointReport:
    path: Path
    loaded: bool = False
    message: Optional[str] = None
    anomalies: List[str] = field(default_factory=list)
    converted: Optional[Path] = None
    fingerprint_path: Optional[Path] = None
    kl_divergences: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "loaded": self.loaded,
            "message": self.message,
            "anomalies": list(self.anomalies),
            "converted": str(self.converted) if self.converted else None,
            "fingerprint": str(self.fingerprint_path) if self.fingerprint_path else None,
            "kl_divergences": self.kl_divergences,
        }


def find_checkpoints(root: Path) -> Iterable[Path]:
    """Yield candidate checkpoint files under the root directory."""
    if root.is_file():
        if root.suffix.lower() in CHECKPOINT_EXTENSIONS:
            yield root
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in CHECKPOINT_EXTENSIONS:
            yield path


def extract_state_dict(obj: object) -> Optional[Dict[str, torch.Tensor]]:
    """Best-effort extraction of a state dict from an arbitrary checkpoint object."""
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
        if isinstance(state_dict, dict):
            return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    return None


def inspect_state_dict(state_dict: Dict[str, torch.Tensor], threshold: float) -> List[str]:
    """Return anomaly descriptions for the provided state dict."""
    anomalies: List[str] = []
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        with torch.no_grad():
            max_abs = tensor.abs().max().item()
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
        if max_abs > threshold:
            anomalies.append(f"{name}: max|w|={max_abs:.2f} > {threshold}")
        if nan_count:
            anomalies.append(f"{name}: contains {int(nan_count)} NaNs")
        if inf_count:
            anomalies.append(f"{name}: contains {int(inf_count)} Infs")
    return anomalies


def tensor_histogram(tensor: torch.Tensor, bins: int = 64) -> Tuple[List[float], Tuple[float, float]]:
    """Compute a normalized histogram for the tensor."""
    flat = tensor.detach().float().cpu().flatten()
    if flat.numel() == 0:
        return [0.0] * bins, (0.0, 0.0)
    min_val = flat.min().item()
    max_val = flat.max().item()
    if math.isclose(min_val, max_val, rel_tol=1e-6, abs_tol=1e-6):
        min_val -= 1e-3
        max_val += 1e-3
    hist = torch.histc(flat, bins=bins, min=min_val, max=max_val)
    hist = hist + 1e-8  # avoid zeros for KL
    hist = hist / hist.sum()
    return hist.tolist(), (min_val, max_val)


def compute_fingerprint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, object]]:
    """Generate histogram-based fingerprints for each parameter."""
    fingerprint: Dict[str, Dict[str, object]] = {}
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        hist, bounds = tensor_histogram(tensor)
        fingerprint[name] = {
            "shape": list(tensor.shape),
            "histogram": hist,
            "range": {"min": bounds[0], "max": bounds[1]},
        }
    return fingerprint


def kl_divergence(p: List[float], q: List[float]) -> float:
    """Compute KL divergence between two discrete distributions."""
    if len(p) != len(q):
        raise ValueError("Histogram length mismatch for KL divergence.")
    total = 0.0
    for pi, qi in zip(p, q):
        total += pi * math.log(pi / qi)
    return total


def compare_fingerprints(
    reference: Dict[str, Dict[str, object]],
    candidate: Dict[str, Dict[str, object]],
    kl_threshold: float,
) -> Tuple[List[str], Dict[str, float]]:
    """Compare candidate fingerprint against reference and emit anomalies."""
    anomalies: List[str] = []
    divergences: Dict[str, float] = {}

    ref_params = set(reference.keys())
    cand_params = set(candidate.keys())

    missing = ref_params - cand_params
    unexpected = cand_params - ref_params
    if missing:
        anomalies.append("Missing parameters: " + ", ".join(sorted(missing)))
    if unexpected:
        anomalies.append(
            "Unexpected parameters detected (possible inserted layers): "
            + ", ".join(sorted(unexpected))
        )

    common = ref_params & cand_params
    for name in common:
        ref_hist = reference[name].get("histogram")
        cand_hist = candidate[name].get("histogram")
        if not isinstance(ref_hist, list) or not isinstance(cand_hist, list):
            continue
        divergence = kl_divergence(cand_hist, ref_hist)
        divergences[name] = divergence
        if divergence > kl_threshold:
            anomalies.append(
                f"{name}: KL divergence {divergence:.3f} exceeds threshold {kl_threshold}"
            )

    return anomalies, divergences


def convert_to_safetensors(state_dict: Dict[str, torch.Tensor], destination: Path) -> Optional[Path]:
    """Write a safetensors file adjacent to the source checkpoint."""
    if save_safetensor is None:
        LOGGER.warning("Skipping safetensors conversion for %s (missing dependency).", destination)
        return None
    tensors = {k: v.detach().cpu() for k, v in state_dict.items()}
    save_safetensor(tensors, str(destination))
    return destination


def triage_checkpoint(
    path: Path,
    threshold: float,
    create_safetensor: bool,
    overwrite: bool,
    fingerprint_dir: Optional[Path],
    reference_fingerprint: Optional[Dict[str, Dict[str, object]]],
    kl_threshold: float,
) -> CheckpointReport:
    """Perform inspection and optional conversion for a single checkpoint file."""
    report = CheckpointReport(path=path)
    load_kwargs = {"map_location": "cpu"}
    if supports_weights_only():
        load_kwargs["weights_only"] = True

    try:
        obj = torch.load(path, **load_kwargs)
        report.loaded = True
    except Exception as exc:  # pragma: no cover - depends on external files
        report.message = f"Failed to load: {exc}"
        return report

    state_dict = extract_state_dict(obj)
    if state_dict is None:
        report.message = "No state dict found; manual inspection recommended."
        return report

    report.anomalies.extend(inspect_state_dict(state_dict, threshold))

    fingerprint = compute_fingerprint(state_dict)
    if reference_fingerprint:
        fingerprint_anomalies, divergences = compare_fingerprints(
            reference_fingerprint, fingerprint, kl_threshold
        )
        report.anomalies.extend(fingerprint_anomalies)
        report.kl_divergences = divergences

    if create_safetensor:
        destination = path.with_suffix(".safetensors")
        if destination.exists() and not overwrite:
            LOGGER.info("Safetensors already exists for %s; skipping.", path)
        else:
            converted = convert_to_safetensors(state_dict, destination)
            report.converted = converted

    if fingerprint_dir:
        fingerprint_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_path = fingerprint_dir / f"{path.stem}.fingerprint.json"
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
        report.fingerprint_path = fingerprint_path

    if not report.anomalies and report.message is None:
        report.message = "No anomalies detected."

    return report


def load_reference_fingerprint(path: Optional[Path]) -> Optional[Dict[str, Dict[str, object]]]:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Reference fingerprint must be a JSON object.")
        return data
    except Exception as exc:  # pragma: no cover - depends on user file
        LOGGER.error("Failed to load reference fingerprint: %s", exc)
        return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoints for anomalies.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more checkpoint files or directories to inspect.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Flag weights whose absolute value exceeds this threshold (default: 100).",
    )
    parser.add_argument(
        "--convert-safetensors",
        action="store_true",
        help="Convert clean state dicts to safetensors alongside original files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing safetensors files if present.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human logs.",
    )
    parser.add_argument(
        "--write-fingerprint",
        type=Path,
        help="Directory to store histogram fingerprints for inspected checkpoints.",
    )
    parser.add_argument(
        "--reference-fingerprint",
        type=Path,
        help="Path to a reference fingerprint JSON for KL comparison.",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=0.5,
        help="KL divergence threshold for flagging distribution drift (default: 0.5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    reports: List[CheckpointReport] = []

    reference_fingerprint = load_reference_fingerprint(args.reference_fingerprint)

    for target in args.paths:
        root = Path(target).expanduser().resolve()
        if not root.exists():
            LOGGER.error("Path not found: %s", root)
            continue
        for checkpoint in find_checkpoints(root):
            LOGGER.info("Inspecting %s", checkpoint)
            report = triage_checkpoint(
                checkpoint,
                threshold=args.threshold,
                create_safetensor=args.convert_safetensors,
                overwrite=args.overwrite,
                fingerprint_dir=args.write_fingerprint,
                reference_fingerprint=reference_fingerprint,
                kl_threshold=args.kl_threshold,
            )
            reports.append(report)
            if not args.json:
                LOGGER.info("→ %s", report.message or "Inspection completed.")
                for anomaly in report.anomalies:
                    LOGGER.warning("   - %s", anomaly)
                if report.converted:
                    LOGGER.info("   - Safetensors written to %s", report.converted)
                if report.fingerprint_path:
                    LOGGER.info("   - Fingerprint stored at %s", report.fingerprint_path)
                for name, divergence in report.kl_divergences.items():
                    LOGGER.info("   - KL[%s] = %.3f", name, divergence)

    if args.json:
        json_data = [report.as_dict() for report in reports]
        json.dump(json_data, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
