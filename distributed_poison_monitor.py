#!/usr/bin/env python3
"""
Gradient divergence monitor for distributed (DDP) training with live streaming support.

Features
--------
1. `GradientSnapshotter` class to embed in training loops. Captures per-step gradient norms/statistics and optionally broadcasts them via UDP to an aggregator.
2. Offline analyzer (`monitor`) that reads JSONL logs, computes divergence ratios, and applies simple change-point detection (CUSUM) to surface slow-burn poisoning.
3. Live listener (`listen`) that ingests broadcast snapshots, aggregates per-step metrics, and raises alerts in real time once all expected workers report.
4. `simulate` helper for quickly generating demonstration data.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("distributed_poison_monitor")


@dataclass
class GradientSnapshot:
    rank: int
    step: int
    global_l2: float
    global_linf: float
    mean: float
    std: float
    parameter_count: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class GradientSnapshotter:
    """
    Capture gradient statistics after backward passes.

    Call `record(step)` after computing gradients, and `flush()` optionally to
    force-write the buffer. If `broadcast_endpoint` is supplied, each snapshot
    is also sent via UDP to a listening aggregator.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        log_dir: str,
        buffer_size: int = 32,
        broadcast_endpoint: str | None = None,
    ):
        self.model = model
        self.rank = rank
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.buffer: list[GradientSnapshot] = []
        self.log_path = self.log_dir / f"worker_{rank}.jsonl"

        self._sock: socket.socket | None = None
        self._broadcast_addr: tuple[str, int] | None = None
        if broadcast_endpoint:
            host, port_str = broadcast_endpoint.split(":")
            self._broadcast_addr = (host, int(port_str))
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _collect_stats(self) -> GradientSnapshot | None:
        total_l2 = 0.0
        total_linf = 0.0
        means: list[float] = []
        stds: list[float] = []
        param_count = 0

        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                param_count += grad.numel()
                l2 = grad.norm(2).item()
                linf = grad.abs().max().item()
                total_l2 += l2
                total_linf = max(total_linf, linf)
                means.append(grad.mean().item())
                stds.append(grad.std().item())

        if param_count == 0:
            return None

        mean_val = float(sum(means) / len(means)) if means else 0.0
        std_val = float(sum(stds) / len(stds)) if stds else 0.0
        return GradientSnapshot(
            rank=self.rank,
            step=-1,
            global_l2=float(total_l2),
            global_linf=float(total_linf),
            mean=mean_val,
            std=std_val,
            parameter_count=param_count,
        )

    def record(self, step: int) -> None:
        snapshot = self._collect_stats()
        if snapshot is None:
            LOGGER.warning("No gradients found for rank %d at step %d.", self.rank, step)
            return
        snapshot.step = step
        self.buffer.append(snapshot)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
        if self._sock and self._broadcast_addr:
            payload = snapshot.to_json().encode("utf-8")
            self._sock.sendto(payload, self._broadcast_addr)

    def flush(self) -> None:
        if not self.buffer:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            for snapshot in self.buffer:
                handle.write(snapshot.to_json() + "\n")
        LOGGER.debug("Wrote %d gradient snapshots to %s", len(self.buffer), self.log_path)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._sock:
            self._sock.close()
            self._sock = None


def load_logs(log_dir: Path) -> list[GradientSnapshot]:
    snapshots: list[GradientSnapshot] = []
    for log_path in sorted(log_dir.glob("worker_*.jsonl")):
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            snapshots.append(GradientSnapshot(**data))
    return snapshots


def compute_step_metrics(snapshots: Iterable[GradientSnapshot]) -> dict[str, float]:
    l2_vals = [entry.global_l2 for entry in snapshots]
    linf_vals = [entry.global_linf for entry in snapshots]
    mean_vals = [entry.mean for entry in snapshots]
    if not l2_vals:
        return {}
    return {
        "l2_ratio": max(l2_vals) / (min(l2_vals) + 1e-6),
        "linf_ratio": max(linf_vals) / (min(linf_vals) + 1e-6),
        "mean_spread": max(mean_vals) - min(mean_vals),
        "workers": len(l2_vals),
    }


def compute_divergence(snapshots: Iterable[GradientSnapshot]) -> dict[int, dict[str, float]]:
    per_step: dict[int, list[GradientSnapshot]] = {}
    for snap in snapshots:
        per_step.setdefault(snap.step, []).append(snap)

    divergences: dict[int, dict[str, float]] = {}
    for step, entries in per_step.items():
        if len(entries) < 2:
            continue
        divergences[step] = compute_step_metrics(entries)
    return divergences


class CUSUMDetector:
    """Simple two-sided cumulative sum detector."""

    def __init__(self, threshold: float, drift: float):
        self.threshold = threshold
        self.drift = drift
        self.pos = 0.0
        self.neg = 0.0

    def update(self, value: float) -> bool:
        self.pos = max(0.0, self.pos + value - self.drift)
        self.neg = min(0.0, self.neg + value + self.drift)
        if self.pos > self.threshold or abs(self.neg) > self.threshold:
            self.pos = 0.0
            self.neg = 0.0
            return True
        return False


def detect_changepoints(divergences: dict[int, dict[str, float]], threshold: float, drift: float) -> list[int]:
    detector = CUSUMDetector(threshold=threshold, drift=drift)
    flagged_steps: list[int] = []
    for step in sorted(divergences.keys()):
        ratio = divergences[step]["l2_ratio"]
        if detector.update(ratio):
            flagged_steps.append(step)
    return flagged_steps


def monitor_logs(
    log_dir: Path,
    threshold: float,
    cusum_threshold: float,
    cusum_drift: float,
) -> int:
    snapshots = load_logs(log_dir)
    if not snapshots:
        LOGGER.error("No gradient logs found in %s", log_dir)
        return 1

    divergences = compute_divergence(snapshots)
    alert_count = 0
    for step, metrics in sorted(divergences.items()):
        l2_ratio = metrics["l2_ratio"]
        linf_ratio = metrics["linf_ratio"]
        mean_spread = metrics["mean_spread"]
        if l2_ratio > threshold or linf_ratio > threshold:
            LOGGER.warning(
                "Step %d: gradient divergence detected (l2_ratio=%.2f, linf_ratio=%.2f, mean_spread=%.4f)",
                step,
                l2_ratio,
                linf_ratio,
                mean_spread,
            )
            alert_count += 1
        else:
            LOGGER.info(
                "Step %d: gradients consistent (l2_ratio=%.2f, linf_ratio=%.2f)",
                step,
                l2_ratio,
                linf_ratio,
            )

    flagged = detect_changepoints(divergences, cusum_threshold, cusum_drift)
    for step in flagged:
        LOGGER.warning("CUSUM change-point detected at step %d (slow drift suspected).", step)
        alert_count += 1

    if alert_count == 0:
        LOGGER.info("No divergence exceeding threshold %.2f detected.", threshold)
    else:
        LOGGER.warning("Detected %d suspicious events. Review logs for potential poisoning.", alert_count)
    return 0 if alert_count == 0 else 2


def simulate_logs(log_dir: Path, steps: int, workers: int) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(workers):
        path = log_dir / f"worker_{rank}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for step in range(steps):
                drift_factor = 5.0 if (rank == 0 and step == steps // 2) else 1.0
                snapshot = GradientSnapshot(
                    rank=rank,
                    step=step,
                    global_l2=1.0 * drift_factor,
                    global_linf=0.5 * drift_factor,
                    mean=0.0,
                    std=1.0,
                    parameter_count=1000,
                )
                handle.write(snapshot.to_json() + "\n")
    LOGGER.info("Synthetic logs written to %s", log_dir)


def listen_broadcast(
    host: str,
    port: int,
    expected_workers: int | None,
    threshold: float,
    cusum_threshold: float,
    cusum_drift: float,
    inactivity_timeout: float,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(inactivity_timeout)
    LOGGER.info("Listening for gradient snapshots on %s:%d", host, port)

    per_step: dict[int, dict[int, GradientSnapshot]] = {}
    detector = CUSUMDetector(cusum_threshold, cusum_drift)

    while True:
        try:
            payload, _ = sock.recvfrom(65535)
        except TimeoutError:
            LOGGER.info("No data received for %.1fs; continuing to listen.", inactivity_timeout)
            continue
        data = json.loads(payload.decode("utf-8"))
        snapshot = GradientSnapshot(**data)
        per_step.setdefault(snapshot.step, {})[snapshot.rank] = snapshot
        ready = per_step[snapshot.step]
        if expected_workers and len(ready) < expected_workers:
            continue

        metrics = compute_step_metrics(ready.values())
        if not metrics:
            continue
        l2_ratio = metrics["l2_ratio"]
        linf_ratio = metrics["linf_ratio"]
        if l2_ratio > threshold or linf_ratio > threshold:
            LOGGER.warning(
                "LIVE ALERT step %d: l2_ratio=%.2f linf_ratio=%.2f (workers=%d)",
                snapshot.step,
                l2_ratio,
                linf_ratio,
                metrics["workers"],
            )
        elif detector.update(l2_ratio):
            LOGGER.warning("LIVE ALERT step %d: CUSUM drift signal triggered.", snapshot.step)
        else:
            LOGGER.info(
                "Step %d live metrics OK (l2_ratio=%.2f, linf_ratio=%.2f, workers=%d)",
                snapshot.step,
                l2_ratio,
                linf_ratio,
                metrics["workers"],
            )
        # Cleanup to avoid unbounded memory.
        per_step.pop(snapshot.step, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed gradient poison monitor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor_parser = subparsers.add_parser("monitor", help="Analyze gradient logs for divergence.")
    monitor_parser.add_argument("--log-dir", required=True, type=Path, help="Directory containing worker_*.jsonl logs.")
    monitor_parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Ratio threshold for raising warnings (default: 3.0).",
    )
    monitor_parser.add_argument(
        "--cusum-threshold",
        type=float,
        default=1.0,
        help="CUSUM alarm threshold for slow drift detection (default: 1.0).",
    )
    monitor_parser.add_argument(
        "--cusum-drift",
        type=float,
        default=0.05,
        help="CUSUM drift parameter (default: 0.05).",
    )

    listen_parser = subparsers.add_parser("listen", help="Listen for live broadcast gradient snapshots.")
    listen_parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind (default: 0.0.0.0).")
    listen_parser.add_argument("--port", type=int, default=5454, help="Port to bind (default: 5454).")
    listen_parser.add_argument(
        "--expected-workers",
        type=int,
        help="Number of workers expected per step. Alerts emitted once all report.",
    )
    listen_parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Ratio threshold for live alerts (default: 3.0).",
    )
    listen_parser.add_argument(
        "--cusum-threshold",
        type=float,
        default=1.0,
        help="CUSUM alarm threshold for live monitoring (default: 1.0).",
    )
    listen_parser.add_argument(
        "--cusum-drift",
        type=float,
        default=0.05,
        help="CUSUM drift parameter for live monitoring (default: 0.05).",
    )
    listen_parser.add_argument(
        "--inactivity-timeout",
        type=float,
        default=30.0,
        help="Seconds before logging inactivity heartbeat (default: 30).",
    )

    simulate_parser = subparsers.add_parser("simulate", help="Generate synthetic logs for demonstration.")
    simulate_parser.add_argument("--log-dir", required=True, type=Path)
    simulate_parser.add_argument("--steps", type=int, default=5)
    simulate_parser.add_argument("--workers", type=int, default=4)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "monitor":
        return monitor_logs(args.log_dir, args.threshold, args.cusum_threshold, args.cusum_drift)
    if args.command == "listen":
        listen_broadcast(
            host=args.host,
            port=args.port,
            expected_workers=args.expected_workers,
            threshold=args.threshold,
            cusum_threshold=args.cusum_threshold,
            cusum_drift=args.cusum_drift,
            inactivity_timeout=args.inactivity_timeout,
        )
        return 0
    if args.command == "simulate":
        simulate_logs(args.log_dir, args.steps, args.workers)
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
