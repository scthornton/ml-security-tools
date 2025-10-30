#!/usr/bin/env python3
"""
Audit NVIDIA Triton `config.pbtxt` files for common security hardening gaps.

The parser is intentionally lightweight and does not require protobuf tooling.
It performs heuristic checks for:
  * Missing `max_batch_size`.
  * `instance_group` blocks without explicit `kind`, `count`, or `memory_limit_mb`.
  * Absent or weak dynamic batching settings.
  * Inputs lacking dimensional constraints (e.g., max sequence length).
  * Missing `model_transaction_policy` entries.

Usage:
    python triton_config_auditor.py models/**/config.pbtxt
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("triton_config_auditor")


@dataclass
class ConfigFinding:
    severity: str  # INFO | WARN | ERROR
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


@dataclass
class ConfigReport:
    path: Path
    findings: List[ConfigFinding] = field(default_factory=list)

    def add(self, severity: str, message: str) -> None:
        self.findings.append(ConfigFinding(severity=severity, message=message))

    def summarize(self) -> None:
        LOGGER.info("Audit report for %s", self.path)
        if not self.findings:
            LOGGER.info("  No issues detected.")
        for finding in self.findings:
            LOGGER.info("  %s", finding)


def preprocess(text: str) -> str:
    """Normalize pbtxt text for the simple parser."""
    text = re.sub(r"\[\s*\{", "{", text)
    text = re.sub(r"\}\s*\]", "}", text)
    return text


def parse_block(lines: List[str], start: int = 0) -> Dict[str, List[object]]:
    """
    Recursively parse a pbtxt-like structure into nested dictionaries.

    Values are stored as lists to preserve multiple entries per key.
    Primitive values remain as strings; numeric conversion is deferred to checks.
    """
    data: Dict[str, List[object]] = {}
    idx = start
    while idx < len(lines):
        raw_line = lines[idx]
        line = raw_line.strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        if line == "}":
            break
        if line.endswith("{"):
            key = line[:-1].strip()
            nested, consumed = parse_block(lines, idx)
            idx = consumed
            data.setdefault(key, []).append(nested)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            data.setdefault(key, []).append(value)
        else:
            # Fallback for keys without values (rare in pbtxt)
            data.setdefault(line, []).append("")
    return data, idx


def load_config(path: Path) -> Dict[str, List[object]]:
    text = preprocess(path.read_text())
    lines = text.splitlines()
    parsed, _ = parse_block(lines)
    return parsed


def get_single_value(block: Dict[str, List[object]], key: str) -> Optional[str]:
    values = block.get(key)
    if not values:
        return None
    return values[0]  # pbtxt stores repeated values explicitly; we only need the first occurrence here


def to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(re.sub(r"[^\d\-]", "", value))
    except ValueError:
        return None


def analyze_config(path: Path) -> ConfigReport:
    report = ConfigReport(path=path)
    try:
        config = load_config(path)
    except Exception as exc:  # pragma: no cover - depends on file contents
        report.add("ERROR", f"Failed to parse config: {exc}")
        return report

    max_batch = to_int(get_single_value(config, "max_batch_size"))
    if max_batch is None:
        report.add("WARN", "max_batch_size missing. Set an explicit upper bound.")
    elif max_batch <= 0:
        report.add("WARN", f"max_batch_size is non-positive ({max_batch}). Check batching limits.")

    instance_groups = config.get("instance_group", [])
    if not instance_groups:
        report.add("WARN", "No instance_group defined. Define GPU instances with explicit limits.")
    for idx, group in enumerate(instance_groups):
        if not isinstance(group, dict):
            continue
        kind = get_single_value(group, "kind")
        if kind is None:
            report.add("WARN", f"instance_group[{idx}] missing kind (e.g., KIND_GPU).")
        count = to_int(get_single_value(group, "count"))
        if count is None:
            report.add("WARN", f"instance_group[{idx}] missing count.")
        memory_limit = to_int(get_single_value(group, "memory_limit_mb"))
        if memory_limit is None:
            report.add("WARN", f"instance_group[{idx}] missing memory_limit_mb (GPU memory quota).")
        gpus = group.get("gpus")
        if kind == "KIND_GPU" and not gpus:
            report.add("WARN", f"instance_group[{idx}] kind=KIND_GPU but no GPU IDs listed.")

    dyn_batch_blocks = config.get("dynamic_batching", [])
    if not dyn_batch_blocks:
        report.add("WARN", "dynamic_batching not configured. Define queue delays to mitigate DoS.")
    else:
        for idx, block in enumerate(dyn_batch_blocks):
            if not isinstance(block, dict):
                continue
            delay = to_int(get_single_value(block, "max_queue_delay_microseconds"))
            if delay is None or delay <= 0:
                report.add(
                    "WARN",
                    "dynamic_batching block missing max_queue_delay_microseconds; set a timeout to prevent slowloris attacks.",
                )

    rate_limiters = config.get("rate_limiter", [])
    if not rate_limiters:
        report.add("WARN", "rate_limiter block absent. Configure request budgets to mitigate abuse.")
    else:
        for idx, limiter in enumerate(rate_limiters):
            if not isinstance(limiter, dict):
                continue
            resources = limiter.get("resources")
            if not resources:
                report.add("WARN", f"rate_limiter[{idx}] missing resources quota definitions.")

    inputs = config.get("input", [])
    if not inputs:
        report.add("WARN", "No input schema defined. Specify inputs to enable validation.")
    for idx, inp in enumerate(inputs):
        if not isinstance(inp, dict):
            continue
        name = get_single_value(inp, "name") or f"input[{idx}]"
        dims = get_single_value(inp, "dims")
        reshape = get_single_value(inp, "reshape")
        if dims is None and reshape is None:
            report.add(
                "WARN",
                f"{name} missing dimension constraints (dims/reshape). Enforce sequence length or tensor bounds.",
            )

    policy_blocks = config.get("model_transaction_policy", [])
    if not policy_blocks:
        report.add("INFO", "model_transaction_policy absent. Consider setting decoupled=false for auditability.")
    else:
        for block in policy_blocks:
            if not isinstance(block, dict):
                continue
            decoupled = get_single_value(block, "decoupled")
            if decoupled is None:
                report.add("INFO", "model_transaction_policy missing decoupled flag; default varies by backend.")

    parameter_blocks = config.get("parameters", [])
    parameter_keys = {
        (get_single_value(block, "key") or "").strip().lower()
        for block in parameter_blocks
        if isinstance(block, dict)
    }
    if parameter_blocks:
        if not any("guard" in key for key in parameter_keys):
            report.add("WARN", "parameters missing NeMo Guardrails/LLM guard configuration.")
        if not any(key in parameter_keys for key in ["auth", "token", "api_key", "authorization"]):
            report.add(
                "WARN",
                "parameters missing explicit auth controls (expected keys like auth/token/api_key).",
            )
        if not any(any(term in key for term in ["redact", "mask", "scrub"]) for key in parameter_keys):
            report.add(
                "WARN",
                "parameters missing logging redaction hints (expect redact/mask configuration).",
            )
    else:
        report.add(
            "WARN",
            "parameters block missing. Configure guardrails, auth, and logging policies explicitly.",
        )

    return report


def iter_targets(patterns: Iterable[str]) -> List[Path]:
    """Expand glob patterns and return config paths."""
    targets: List[Path] = []
    for pattern in patterns:
        base = Path(pattern)
        if base.exists():
            if base.is_file():
                targets.append(base.resolve())
            else:
                targets.extend(
                    path.resolve()
                    for path in base.rglob("config.pbtxt")
                    if path.is_file()
                )
            continue
        # Treat as glob
        for path in Path().glob(pattern):
            if path.is_file():
                targets.append(path.resolve())
    return sorted(set(targets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Triton config.pbtxt files for security gaps.")
    parser.add_argument(
        "targets",
        nargs="+",
        help="Files, directories, or glob patterns pointing to config.pbtxt files.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print WARN/ERROR counts per file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = iter_targets(args.targets)
    if not paths:
        LOGGER.error("No config.pbtxt files found for the provided targets.")
        return 1

    exit_code = 0
    for path in paths:
        report = analyze_config(path)
        if args.summary:
            warn_count = sum(1 for finding in report.findings if finding.severity != "INFO")
            LOGGER.info("%s: %d issues", path, warn_count)
        else:
            report.summarize()
        if any(f.severity == "ERROR" for f in report.findings):
            exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
