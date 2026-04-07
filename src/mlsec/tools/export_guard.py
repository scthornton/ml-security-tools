#!/usr/bin/env python3
"""
Guardrail utility for PyTorch → ONNX → TensorRT conversions with security linting.

Enhancements over a plain export script:
  * Captures PyTorch reference outputs and hashes the state_dict for provenance.
  * Validates ONNX graphs, linting for custom domains, control-flow ops, and large constants.
  * Optionally evaluates ONNX Runtime drift and invokes `trtexec` to build engines.
  * Emits SHA-256 digests for each stage to support attestation/integrity workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

try:
    import onnx  # type: ignore
    from onnx import numpy_helper  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    onnx = None  # type: ignore[assignment]
    numpy_helper = None  # type: ignore[assignment]

try:
    import onnxruntime  # type: ignore
except ImportError:  # pragma: no cover
    onnxruntime = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("tensorrt_export_guard")

CONTROL_FLOW_OPS = {"Loop", "If", "Scan", "While"}
DEFAULT_ALLOWED_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


def load_module_from_path(script_path: Path) -> Any:
    """Import a Python module from an explicit file path."""
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_factory(model_script: str, factory_name: str) -> nn.Module:
    """Instantiate a model from the provided script/module."""
    script_path = Path(model_script)
    if script_path.exists():
        module = load_module_from_path(script_path.resolve())
    else:
        module = importlib.import_module(model_script)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise AttributeError(f"{factory_name} not found in {model_script}")
    model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError(f"{factory_name} must return an nn.Module instance.")
    return model


def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse a comma-separated tensor shape specification."""
    try:
        return tuple(int(dim.strip()) for dim in shape_str.split(",") if dim.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid shape specification: {shape_str}") from exc


def describe_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Return numeric diagnostics for a tensor."""
    with torch.no_grad():
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std().item()),
        }


def state_dict_hash(model: nn.Module) -> str:
    """Deterministically hash model parameters for provenance tracking."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        data = tensor.detach().cpu().numpy().tobytes()
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    onnx_path: Path,
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    LOGGER.info("Exporting ONNX graph to %s (opset %d)", onnx_path, opset)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample_input,),
        onnx_path,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def compare_with_onnxruntime(
    onnx_path: Path, input_tensor: torch.Tensor, reference: torch.Tensor, atol: float, rtol: float
) -> float | None:
    """Run ONNX Runtime for the exported model and compute the max absolute diff."""
    if onnxruntime is None:
        LOGGER.warning("onnxruntime not available; skipping numerical comparison.")
        return None

    LOGGER.info("Running ONNX Runtime accuracy check.")
    session = onnxruntime.InferenceSession(str(onnx_path))
    ort_inputs = {session.get_inputs()[0].name: input_tensor.detach().cpu().numpy()}
    ort_outputs = session.run(None, ort_inputs)
    ort_tensor = torch.from_numpy(ort_outputs[0])

    diff = torch.max(torch.abs(reference.cpu() - ort_tensor))
    if diff > atol + rtol * torch.max(torch.abs(reference.cpu())):
        LOGGER.warning("ONNX Runtime drift detected: max abs diff=%.6f", diff.item())
    else:
        LOGGER.info("ONNX Runtime outputs within tolerance (max abs diff=%.6f).", diff.item())
    return float(diff.item())


def lint_onnx_graph(onnx_model_path: Path, allowed_domains: Iterable[str], constant_threshold: float) -> list[str]:
    """Inspect the ONNX graph for risky patterns."""
    findings: list[str] = []
    if onnx is None or numpy_helper is None:
        LOGGER.warning("onnx package not available; skipping structural lint.")
        return findings

    model = onnx.load(str(onnx_model_path))
    graph = model.graph

    allowed_domains_set = set(allowed_domains)

    for node in graph.node:
        domain = node.domain or ""
        if domain not in allowed_domains_set:
            findings.append(f"Custom domain '{domain}' detected for op {node.op_type}.")
        if node.op_type in CONTROL_FLOW_OPS:
            findings.append(f"Control-flow op '{node.op_type}' present. Review conversion.")
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    array = numpy_helper.to_array(attr.t)
                    max_abs = float(abs(array).max())
                    if max_abs > constant_threshold:
                        findings.append(
                            f"Constant node carries large magnitude data (max|x|={max_abs:.2f} > {constant_threshold})."
                        )
        # Detect subgraphs (possible hidden logic)
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                findings.append(
                    f"Node {node.op_type} contains embedded subgraph '{attr.name}'. Inspect for hidden control flow."
                )

    return findings


def run_trtexec(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace: int,
    extra_args: Sequence[str] | None,
) -> Path | None:
    """Invoke NVIDIA trtexec to build a TensorRT engine, if available."""
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        LOGGER.warning("trtexec executable not found; skipping TensorRT build.")
        return None

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
    ]
    precision_flag = f"--{precision.lower()}"
    if precision.lower() in {"fp32", "fp16", "int8", "fp8"}:
        cmd.append(precision_flag)
    if extra_args:
        cmd.extend(extra_args)

    LOGGER.info("Running trtexec: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on environment
        LOGGER.error("trtexec failed: %s", exc)
        return None

    if not engine_path.exists():
        LOGGER.error("Expected engine file %s not found after trtexec run.", engine_path)
        return None

    LOGGER.info("TensorRT engine written to %s", engine_path)
    return engine_path


def validate_onnx(onnx_path: Path) -> None:
    """Check ONNX graph validity."""
    if onnx is None:
        LOGGER.warning("onnx package not available; skipping graph validation.")
        return
    LOGGER.info("Validating ONNX graph with onnx.checker.")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX/TensorRT with guardrails.")
    parser.add_argument("--model-script", required=True, help="Python module or path that provides the model factory.")
    parser.add_argument(
        "--factory", default="create_model", help="Factory function to instantiate the model (default: create_model)."
    )
    parser.add_argument(
        "--input-shape", type=parse_shape, required=True, help="Comma-separated input tensor shape, e.g. 1,3,224,224."
    )
    parser.add_argument("--dtype", default="float32", help="Input tensor dtype (default: float32).")
    parser.add_argument("--export-dir", default="export_guard", help="Directory to place generated artifacts.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version to use.")
    parser.add_argument(
        "--dynamic-axis",
        action="append",
        default=[],
        help="Dynamic axis specification in the form dim:name (e.g. 0:batch).",
    )
    parser.add_argument("--enable-onnxruntime", action="store_true", help="Run ONNX Runtime to compare outputs.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for numerical comparisons.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for numerical comparisons.")
    parser.add_argument("--build-engine", action="store_true", help="Invoke trtexec to produce a TensorRT engine.")
    parser.add_argument("--precision", default="fp16", help="Precision flag to pass to trtexec (fp32|fp16|int8|fp8).")
    parser.add_argument("--workspace", type=int, default=1024, help="TensorRT workspace size in MB.")
    parser.add_argument("--trtexec-extra", nargs="*", default=None, help="Additional arguments to pass to trtexec.")
    parser.add_argument(
        "--hash-record",
        type=Path,
        help="Path to write SHA-256 digests for PyTorch/ONNX/TensorRT artifacts.",
    )
    parser.add_argument(
        "--allowed-domain",
        action="append",
        default=[],
        help="Additional ONNX op domains to allow during linting.",
    )
    parser.add_argument(
        "--constant-threshold",
        type=float,
        default=1e3,
        help="Maximum allowed absolute value for ONNX Constant nodes before flagging.",
    )
    return parser.parse_args(argv)


def create_sample_input(shape: tuple[int, ...], dtype: str) -> torch.Tensor:
    """Create a random sample tensor with the requested shape/dtype."""
    torch_dtype = getattr(torch, dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return torch.randn(shape, dtype=torch_dtype)


def build_dynamic_axes(specs: Sequence[str]) -> dict[str, dict[int, str]] | None:
    """Parse dynamic axis specifications like '0:batch'."""
    if not specs:
        return None
    axes: dict[int, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise argparse.ArgumentTypeError(f"Invalid dynamic axis spec: {spec}")
        dim_str, name = spec.split(":", 1)
        axes[int(dim_str)] = name
    return {"input": axes, "output": axes}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model using %s.%s", args.model_script, args.factory)
    model = resolve_factory(args.model_script, args.factory)
    model.eval()

    provenance_hash = state_dict_hash(model)
    LOGGER.info("PyTorch state_dict SHA-256: %s", provenance_hash)

    sample_input = create_sample_input(args.input_shape, args.dtype)
    with torch.no_grad():
        reference_output = model(sample_input)
    if not isinstance(reference_output, torch.Tensor):
        raise TypeError("Model output must be a tensor for comparison.")

    torch_report = {
        "input": describe_tensor(sample_input),
        "output": describe_tensor(reference_output),
    }
    LOGGER.info("PyTorch reference statistics: %s", json.dumps(torch_report, indent=2))

    onnx_path = export_dir / "model.onnx"
    dynamic_axes = build_dynamic_axes(args.dynamic_axis)
    export_to_onnx(model, sample_input, onnx_path, args.opset, dynamic_axes)
    validate_onnx(onnx_path)

    allowed_domains = DEFAULT_ALLOWED_DOMAINS | set(args.allowed_domain)
    lint_findings = lint_onnx_graph(onnx_path, allowed_domains, args.constant_threshold)
    for finding in lint_findings:
        LOGGER.warning("ONNX lint: %s", finding)

    if args.enable_onnxruntime:
        compare_with_onnxruntime(onnx_path, sample_input, reference_output, args.atol, args.rtol)

    engine_path: Path | None = None
    if args.build_engine:
        engine_path = export_dir / "model.engine"
        workspace_bytes = args.workspace * (1 << 20)
        engine_path = run_trtexec(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=args.precision,
            workspace=workspace_bytes,
            extra_args=args.trtexec_extra,
        )

    if args.hash_record:
        hashes: dict[str, Any] = {
            "pytorch_state_dict": provenance_hash,
            "onnx_sha256": file_sha256(onnx_path),
            "torch_report": torch_report,
            "onnx_findings": lint_findings,
        }
        if engine_path and engine_path.exists():
            hashes["tensorrt_engine_sha256"] = file_sha256(engine_path)
        args.hash_record.parent.mkdir(parents=True, exist_ok=True)
        args.hash_record.write_text(json.dumps(hashes, indent=2) + "\n")
        LOGGER.info("Hash record written to %s", args.hash_record)

    LOGGER.info("Export guard run complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
