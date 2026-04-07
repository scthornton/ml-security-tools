"""Unified CLI entry point for all mlsec tools."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlsec",
        description="ML Security Toolkit — analyze, audit, and harden machine learning systems.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_get_version()}")

    subparsers = parser.add_subparsers(dest="tool", help="Security tool to run")

    # Each tool registers its own subparser
    _register_adversarial(subparsers)
    _register_inspect(subparsers)
    _register_poison(subparsers)
    _register_export_guard(subparsers)
    _register_checkpoint(subparsers)
    _register_triton(subparsers)

    args = parser.parse_args(argv)
    if args.tool is None:
        parser.print_help()
        return 0

    return int(args.func(args))


def _get_version() -> str:
    try:
        from mlsec import __version__

        return __version__
    except ImportError:
        return "unknown"


def _register_adversarial(subparsers: argparse._SubParsersAction) -> None:
    sub = subparsers.add_parser(
        "adversarial",
        aliases=["adv"],
        help="Test adversarial robustness (FGSM, PGD, CW attacks)",
    )
    sub.set_defaults(func=lambda args: _run_tool("mlsec.tools.adversarial", args))


def _register_inspect(subparsers: argparse._SubParsersAction) -> None:
    sub = subparsers.add_parser(
        "inspect",
        help="Inspect model weights and activations for anomalies",
    )
    sub.set_defaults(func=lambda args: _run_tool("mlsec.tools.model_inspect", args))


def _register_poison(subparsers: argparse._SubParsersAction) -> None:
    sub = subparsers.add_parser(
        "poison",
        help="Monitor distributed training for gradient poisoning",
    )
    sub.set_defaults(func=lambda args: _run_tool("mlsec.tools.poison_monitor", args))


def _register_export_guard(subparsers: argparse._SubParsersAction) -> None:
    sub = subparsers.add_parser(
        "export-guard",
        aliases=["export"],
        help="Validate PyTorch → ONNX → TensorRT export pipeline",
    )
    sub.set_defaults(func=lambda args: _run_tool("mlsec.tools.export_guard", args))


def _register_checkpoint(subparsers: argparse._SubParsersAction) -> None:
    sub = subparsers.add_parser(
        "checkpoint",
        aliases=["ckpt"],
        help="Triage PyTorch checkpoint files for security issues",
    )
    sub.set_defaults(func=lambda args: _run_tool("mlsec.tools.checkpoint_triage", args))


def _register_triton(subparsers: argparse._SubParsersAction) -> None:
    sub = subparsers.add_parser(
        "triton",
        help="Audit Triton Inference Server configurations",
    )
    sub.set_defaults(func=lambda args: _run_tool("mlsec.tools.triton_auditor", args))


def _run_tool(module_path: str, _args: argparse.Namespace) -> int:
    """Import and run a tool's main(), forwarding remaining CLI args."""
    import importlib

    module = importlib.import_module(module_path)
    # Strip "mlsec <tool>" from argv so each tool sees only its own args
    tool_argv = sys.argv[2:] if len(sys.argv) > 2 else []
    result = module.main(tool_argv)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
