"""
Tests for distributed_poison_monitor.py

Coverage targets
----------------
- GradientSnapshot: serialisation (to_json / round-trip)
- GradientSnapshotter.record: stats collected correctly after backward pass
- GradientSnapshotter.flush: writes JSONL file and clears buffer
- GradientSnapshotter.record (buffer overflow): auto-flush at buffer_size
- GradientSnapshotter: UDP broadcast (socket.sendto mocked)
- GradientSnapshotter._collect_stats: returns None when no gradients exist
- load_logs: parses JSONL files correctly, handles empty/missing files
- compute_step_metrics: correct ratios and spread
- compute_divergence: groups by step, skips steps with < 2 workers
- CUSUMDetector.update: returns True on change-point, resets accumulators
- detect_changepoints: identifies flagged steps from divergence data
- simulate_logs: creates the right number of files with correct format
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from torch import nn

import sys
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import distributed_poison_monitor as dpm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model_with_grads() -> nn.Module:
    """Return a tiny model that has accumulated gradients."""
    model = nn.Linear(4, 2, bias=False)
    nn.init.constant_(model.weight, 0.1)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    return model


def _write_snapshot_lines(path: Path, snapshots: list) -> None:
    with path.open("w") as fh:
        for s in snapshots:
            fh.write(s.to_json() + "\n")


# ===========================================================================
# GradientSnapshot serialisation
# ===========================================================================

class TestGradientSnapshotSerialisation:
    def test_to_json_produces_valid_json(self):
        snap = dpm.GradientSnapshot(
            rank=0, step=1, global_l2=1.5, global_linf=0.8,
            mean=0.01, std=0.5, parameter_count=100
        )
        data = json.loads(snap.to_json())
        assert data["rank"] == 0
        assert data["step"] == 1

    def test_round_trip_through_json(self):
        snap = dpm.GradientSnapshot(
            rank=2, step=5, global_l2=2.3, global_linf=1.1,
            mean=-0.05, std=0.9, parameter_count=512
        )
        restored = dpm.GradientSnapshot(**json.loads(snap.to_json()))
        assert restored.rank == snap.rank
        assert restored.step == snap.step
        assert abs(restored.global_l2 - snap.global_l2) < 1e-9


# ===========================================================================
# GradientSnapshotter.record
# ===========================================================================

class TestGradientSnapshotterRecord:
    def test_record_appends_to_buffer(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.record(step=1)
        assert len(snapshotter.buffer) == 1

    def test_recorded_snapshot_has_correct_step(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.record(step=42)
        assert snapshotter.buffer[0].step == 42

    def test_record_without_gradients_does_not_append(self, tmp_dir, caplog):
        """A model that has no gradients should result in no snapshot."""
        import logging
        model = nn.Linear(4, 2, bias=False)  # no backward called
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        with caplog.at_level(logging.WARNING, logger="distributed_poison_monitor"):
            snapshotter.record(step=0)
        assert len(snapshotter.buffer) == 0

    def test_l2_norm_is_positive(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.record(step=0)
        assert snapshotter.buffer[0].global_l2 > 0


# ===========================================================================
# GradientSnapshotter.flush
# ===========================================================================

class TestGradientSnapshotterFlush:
    def test_flush_writes_jsonl_file(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.record(step=1)
        snapshotter.record(step=2)
        snapshotter.flush()
        log_file = tmp_dir / "worker_0.jsonl"
        assert log_file.exists()
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_flush_clears_buffer(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.record(step=1)
        snapshotter.flush()
        assert len(snapshotter.buffer) == 0

    def test_flush_on_empty_buffer_does_not_create_file(self, tmp_dir):
        model = nn.Linear(4, 2)
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.flush()
        log_file = tmp_dir / "worker_0.jsonl"
        assert not log_file.exists()

    def test_auto_flush_at_buffer_size(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir), buffer_size=2)
        snapshotter.record(step=1)
        assert len(snapshotter.buffer) == 1
        snapshotter.record(step=2)  # triggers auto-flush
        assert len(snapshotter.buffer) == 0
        log_file = tmp_dir / "worker_0.jsonl"
        assert log_file.exists()

    def test_close_flushes_remaining_buffer(self, tmp_dir):
        model = _simple_model_with_grads()
        snapshotter = dpm.GradientSnapshotter(model, rank=0, log_dir=str(tmp_dir))
        snapshotter.record(step=5)
        snapshotter.close()
        log_file = tmp_dir / "worker_0.jsonl"
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1


# ===========================================================================
# GradientSnapshotter — UDP broadcast (mocked)
# ===========================================================================

class TestGradientSnapshotterBroadcast:
    def test_sendto_called_on_record(self, tmp_dir):
        model = _simple_model_with_grads()
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            snapshotter = dpm.GradientSnapshotter(
                model, rank=0, log_dir=str(tmp_dir),
                broadcast_endpoint="127.0.0.1:5454"
            )
            snapshotter.record(step=1)
            assert mock_sock.sendto.called

    def test_payload_is_valid_json(self, tmp_dir):
        model = _simple_model_with_grads()
        sent_payloads = []
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.sendto.side_effect = lambda data, addr: sent_payloads.append(data)
            mock_socket_cls.return_value = mock_sock
            snapshotter = dpm.GradientSnapshotter(
                model, rank=0, log_dir=str(tmp_dir),
                broadcast_endpoint="127.0.0.1:5454"
            )
            snapshotter.record(step=1)
        assert len(sent_payloads) == 1
        parsed = json.loads(sent_payloads[0].decode("utf-8"))
        assert "rank" in parsed and "step" in parsed


# ===========================================================================
# load_logs
# ===========================================================================

class TestLoadLogs:
    def test_loads_single_worker_file(self, tmp_dir):
        snaps = [
            dpm.GradientSnapshot(rank=0, step=i, global_l2=1.0, global_linf=0.5,
                                 mean=0.0, std=1.0, parameter_count=100)
            for i in range(3)
        ]
        _write_snapshot_lines(tmp_dir / "worker_0.jsonl", snaps)
        loaded = dpm.load_logs(tmp_dir)
        assert len(loaded) == 3

    def test_loads_multiple_worker_files(self, tmp_dir):
        for rank in range(3):
            snaps = [
                dpm.GradientSnapshot(rank=rank, step=0, global_l2=1.0, global_linf=0.5,
                                     mean=0.0, std=1.0, parameter_count=100)
            ]
            _write_snapshot_lines(tmp_dir / f"worker_{rank}.jsonl", snaps)
        loaded = dpm.load_logs(tmp_dir)
        assert len(loaded) == 3

    def test_empty_directory_returns_empty_list(self, tmp_dir):
        loaded = dpm.load_logs(tmp_dir)
        assert loaded == []

    def test_skips_blank_lines(self, tmp_dir):
        path = tmp_dir / "worker_0.jsonl"
        snap = dpm.GradientSnapshot(rank=0, step=1, global_l2=1.0, global_linf=0.5,
                                    mean=0.0, std=1.0, parameter_count=100)
        path.write_text("\n" + snap.to_json() + "\n\n")
        loaded = dpm.load_logs(tmp_dir)
        assert len(loaded) == 1


# ===========================================================================
# compute_step_metrics / compute_divergence
# ===========================================================================

class TestComputeMetrics:
    def test_compute_step_metrics_correct_l2_ratio(self):
        snaps = [
            dpm.GradientSnapshot(rank=0, step=0, global_l2=1.0, global_linf=0.5,
                                 mean=0.0, std=1.0, parameter_count=100),
            dpm.GradientSnapshot(rank=1, step=0, global_l2=4.0, global_linf=2.0,
                                 mean=0.0, std=1.0, parameter_count=100),
        ]
        metrics = dpm.compute_step_metrics(snaps)
        # max/min = 4.0/(1.0+1e-6) ≈ 4.0
        assert abs(metrics["l2_ratio"] - 4.0) < 0.01

    def test_compute_step_metrics_empty_returns_empty(self):
        assert dpm.compute_step_metrics([]) == {}

    def test_compute_divergence_skips_single_worker_step(self):
        snaps = [
            dpm.GradientSnapshot(rank=0, step=1, global_l2=1.0, global_linf=0.5,
                                 mean=0.0, std=1.0, parameter_count=100)
        ]
        result = dpm.compute_divergence(snaps)
        assert 1 not in result

    def test_compute_divergence_includes_multi_worker_step(self):
        snaps = [
            dpm.GradientSnapshot(rank=0, step=2, global_l2=1.0, global_linf=0.5,
                                 mean=0.0, std=1.0, parameter_count=100),
            dpm.GradientSnapshot(rank=1, step=2, global_l2=2.0, global_linf=1.0,
                                 mean=0.0, std=1.0, parameter_count=100),
        ]
        result = dpm.compute_divergence(snaps)
        assert 2 in result

    def test_mean_spread_computed_correctly(self):
        snaps = [
            dpm.GradientSnapshot(rank=0, step=0, global_l2=1.0, global_linf=0.5,
                                 mean=0.1, std=1.0, parameter_count=100),
            dpm.GradientSnapshot(rank=1, step=0, global_l2=1.0, global_linf=0.5,
                                 mean=0.5, std=1.0, parameter_count=100),
        ]
        metrics = dpm.compute_step_metrics(snaps)
        assert abs(metrics["mean_spread"] - 0.4) < 1e-6


# ===========================================================================
# CUSUMDetector
# ===========================================================================

class TestCUSUMDetector:
    def test_no_changepoint_when_value_equals_drift(self):
        """When every value equals the drift parameter, pos stays at 0 and never fires."""
        drift = 0.5
        detector = dpm.CUSUMDetector(threshold=5.0, drift=drift)
        # value - drift == 0, so pos stays at 0 indefinitely
        flags = [detector.update(drift) for _ in range(50)]
        assert not any(flags)

    def test_changepoint_detected_on_spike(self):
        detector = dpm.CUSUMDetector(threshold=1.0, drift=0.05)
        flags = [detector.update(10.0) for _ in range(5)]
        assert any(flags), "Expected at least one change-point detection on a sustained spike"

    def test_accumulator_resets_after_detection(self):
        detector = dpm.CUSUMDetector(threshold=1.0, drift=0.05)
        detector.update(100.0)  # triggers reset
        assert detector.pos == 0.0
        assert detector.neg == 0.0

    def test_negative_drift_detected(self):
        """Large negative values should trigger the neg accumulator."""
        detector = dpm.CUSUMDetector(threshold=1.0, drift=0.05)
        flags = [detector.update(-10.0) for _ in range(5)]
        assert any(flags)


# ===========================================================================
# detect_changepoints
# ===========================================================================

class TestDetectChangepoints:
    def test_returns_empty_on_stable_divergences(self):
        # Use value == drift so the CUSUM accumulator stays at exactly 0
        drift = 1.1
        divergences = {i: {"l2_ratio": drift, "linf_ratio": 1.0, "mean_spread": 0.01, "workers": 2}
                      for i in range(20)}
        result = dpm.detect_changepoints(divergences, threshold=100.0, drift=drift)
        assert result == []

    def test_returns_step_number_for_spike(self):
        # Build divergences where step 5 has a huge l2_ratio
        divergences = {}
        for i in range(10):
            ratio = 50.0 if i == 5 else 1.0
            divergences[i] = {"l2_ratio": ratio, "linf_ratio": 1.0, "mean_spread": 0.0, "workers": 2}
        result = dpm.detect_changepoints(divergences, threshold=1.0, drift=0.05)
        assert len(result) > 0

    def test_steps_are_in_sorted_order(self):
        divergences = {3: {"l2_ratio": 1.0, "linf_ratio": 1.0, "mean_spread": 0.0, "workers": 2},
                      1: {"l2_ratio": 1.0, "linf_ratio": 1.0, "mean_spread": 0.0, "workers": 2},
                      2: {"l2_ratio": 1.0, "linf_ratio": 1.0, "mean_spread": 0.0, "workers": 2}}
        # Should not raise; ordering is internal to detect_changepoints
        dpm.detect_changepoints(divergences, threshold=10.0, drift=0.1)


# ===========================================================================
# simulate_logs
# ===========================================================================

class TestSimulateLogs:
    def test_creates_correct_number_of_worker_files(self, tmp_dir):
        dpm.simulate_logs(tmp_dir, steps=4, workers=3)
        files = list(tmp_dir.glob("worker_*.jsonl"))
        assert len(files) == 3

    def test_each_file_has_correct_number_of_lines(self, tmp_dir):
        dpm.simulate_logs(tmp_dir, steps=6, workers=2)
        for rank in range(2):
            path = tmp_dir / f"worker_{rank}.jsonl"
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            assert len(lines) == 6

    def test_simulated_snapshots_are_valid_json(self, tmp_dir):
        dpm.simulate_logs(tmp_dir, steps=3, workers=1)
        path = tmp_dir / "worker_0.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                assert "rank" in data
                assert "global_l2" in data

    def test_drift_injected_at_midpoint(self, tmp_dir):
        """Rank 0 at step N//2 should have l2=5.0, others should have l2=1.0."""
        steps = 6
        dpm.simulate_logs(tmp_dir, steps=steps, workers=2)
        path = tmp_dir / "worker_0.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        midpoint_row = rows[steps // 2]
        assert midpoint_row["global_l2"] == 5.0
        # Non-midpoint row should be 1.0
        normal_row = rows[0]
        assert normal_row["global_l2"] == 1.0
