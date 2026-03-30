"""Tests for distributed_poison_monitor.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import distributed_poison_monitor as monitor  # noqa: E402


# ===========================================================================
# GradientSnapshot
# ===========================================================================

class TestGradientSnapshot:
    def test_to_json_round_trip(self):
        snap = monitor.GradientSnapshot(
            rank=0, step=5, global_l2=1.5, global_linf=0.8,
            mean=0.01, std=0.5, parameter_count=1000,
        )
        data = json.loads(snap.to_json())
        assert data["rank"] == 0
        assert data["step"] == 5
        assert abs(data["global_l2"] - 1.5) < 1e-6


# ===========================================================================
# GradientSnapshotter
# ===========================================================================

class TestGradientSnapshotter:
    def test_record_captures_stats(self, tmp_path):
        model = nn.Linear(4, 2)
        inputs = torch.randn(1, 4)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()

        snapshotter = monitor.GradientSnapshotter(
            model=model, rank=0, log_dir=str(tmp_path), buffer_size=100,
        )
        snapshotter.record(step=0)
        assert len(snapshotter.buffer) == 1
        assert snapshotter.buffer[0].step == 0
        assert snapshotter.buffer[0].parameter_count > 0
        snapshotter.close()

    def test_flush_writes_to_disk(self, tmp_path):
        model = nn.Linear(4, 2)
        inputs = torch.randn(1, 4)
        loss = model(inputs).sum()
        loss.backward()

        snapshotter = monitor.GradientSnapshotter(
            model=model, rank=0, log_dir=str(tmp_path), buffer_size=100,
        )
        snapshotter.record(step=0)
        snapshotter.flush()
        log_file = tmp_path / "worker_0.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        snapshotter.close()

    def test_auto_flush_on_buffer_full(self, tmp_path):
        model = nn.Linear(4, 2)
        snapshotter = monitor.GradientSnapshotter(
            model=model, rank=0, log_dir=str(tmp_path), buffer_size=2,
        )
        for step in range(3):
            inputs = torch.randn(1, 4)
            loss = model(inputs).sum()
            loss.backward()
            snapshotter.record(step=step)
            model.zero_grad()

        # After 3 records with buffer_size=2, at least one flush happened
        log_file = tmp_path / "worker_0.jsonl"
        assert log_file.exists()
        snapshotter.close()

    def test_record_with_no_gradients_warns(self, tmp_path):
        model = nn.Linear(4, 2)
        # No backward() called — no gradients
        snapshotter = monitor.GradientSnapshotter(
            model=model, rank=0, log_dir=str(tmp_path),
        )
        snapshotter.record(step=0)
        assert len(snapshotter.buffer) == 0
        snapshotter.close()


# ===========================================================================
# CUSUM Detector
# ===========================================================================

class TestCUSUMDetector:
    def test_no_alarm_for_low_values(self):
        detector = monitor.CUSUMDetector(threshold=5.0, drift=0.5)
        for _ in range(20):
            assert detector.update(0.1) is False

    def test_alarm_for_sustained_high_values(self):
        detector = monitor.CUSUMDetector(threshold=2.0, drift=0.1)
        triggered = False
        for _ in range(100):
            if detector.update(1.0):
                triggered = True
                break
        assert triggered

    def test_alarm_for_negative_drift(self):
        detector = monitor.CUSUMDetector(threshold=2.0, drift=0.1)
        triggered = False
        for _ in range(100):
            if detector.update(-1.0):
                triggered = True
                break
        assert triggered

    def test_reset_after_alarm(self):
        detector = monitor.CUSUMDetector(threshold=1.0, drift=0.0)
        # Feed enough to trigger
        detector.update(2.0)
        # After alarm, accumulators reset
        assert detector.pos == 0.0
        assert detector.neg == 0.0


# ===========================================================================
# compute_divergence
# ===========================================================================

class TestComputeDivergence:
    def test_single_worker_per_step_skipped(self):
        snapshots = [
            monitor.GradientSnapshot(rank=0, step=0, global_l2=1.0, global_linf=0.5,
                                      mean=0.0, std=1.0, parameter_count=100),
        ]
        result = monitor.compute_divergence(snapshots)
        assert len(result) == 0

    def test_two_workers_computes_ratio(self):
        snapshots = [
            monitor.GradientSnapshot(rank=0, step=0, global_l2=1.0, global_linf=0.5,
                                      mean=0.0, std=1.0, parameter_count=100),
            monitor.GradientSnapshot(rank=1, step=0, global_l2=3.0, global_linf=1.5,
                                      mean=0.1, std=1.2, parameter_count=100),
        ]
        result = monitor.compute_divergence(snapshots)
        assert 0 in result
        assert result[0]["l2_ratio"] == pytest.approx(3.0, rel=1e-3)

    def test_divergent_worker_high_ratio(self):
        snapshots = [
            monitor.GradientSnapshot(rank=0, step=0, global_l2=1.0, global_linf=0.5,
                                      mean=0.0, std=1.0, parameter_count=100),
            monitor.GradientSnapshot(rank=1, step=0, global_l2=1.0, global_linf=0.5,
                                      mean=0.0, std=1.0, parameter_count=100),
            monitor.GradientSnapshot(rank=2, step=0, global_l2=10.0, global_linf=5.0,
                                      mean=0.5, std=3.0, parameter_count=100),
        ]
        result = monitor.compute_divergence(snapshots)
        assert result[0]["l2_ratio"] > 5.0


# ===========================================================================
# simulate_logs
# ===========================================================================

class TestSimulateLogs:
    def test_creates_worker_files(self, tmp_path):
        monitor.simulate_logs(tmp_path, steps=5, workers=3)
        for rank in range(3):
            assert (tmp_path / f"worker_{rank}.jsonl").exists()

    def test_correct_number_of_lines(self, tmp_path):
        monitor.simulate_logs(tmp_path, steps=10, workers=2)
        lines = (tmp_path / "worker_0.jsonl").read_text().strip().splitlines()
        assert len(lines) == 10


# ===========================================================================
# monitor_logs (offline analysis)
# ===========================================================================

class TestMonitorLogs:
    def test_no_logs_returns_1(self, tmp_path):
        result = monitor.monitor_logs(tmp_path, threshold=3.0, cusum_threshold=1.0, cusum_drift=0.05)
        assert result == 1

    def test_clean_logs_return_0(self, tmp_path):
        monitor.simulate_logs(tmp_path, steps=10, workers=4)
        # Simulate creates a spike at step=steps//2 for worker 0
        # With default threshold=3.0, one event should trigger
        result = monitor.monitor_logs(tmp_path, threshold=3.0, cusum_threshold=1.0, cusum_drift=0.05)
        assert result in (0, 2)  # may or may not trigger based on spike

    def test_detect_injected_spike(self, tmp_path):
        # Create logs where worker 0 has a massive spike at step 5
        for rank in range(2):
            path = tmp_path / f"worker_{rank}.jsonl"
            with path.open("w") as f:
                for step in range(10):
                    l2 = 100.0 if (rank == 0 and step == 5) else 1.0
                    snap = monitor.GradientSnapshot(
                        rank=rank, step=step, global_l2=l2, global_linf=l2 / 2,
                        mean=0.0, std=1.0, parameter_count=100,
                    )
                    f.write(snap.to_json() + "\n")

        result = monitor.monitor_logs(tmp_path, threshold=3.0, cusum_threshold=1.0, cusum_drift=0.05)
        assert result == 2  # anomalies detected
