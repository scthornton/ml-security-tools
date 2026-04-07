"""Tests for triton_config_auditor.py"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import triton_config_auditor as auditor  # noqa: E402

# ===========================================================================
# parse_block
# ===========================================================================


class TestParseBlock:
    def test_simple_key_value(self):
        lines = ['name: "test"', "max_batch_size: 8"]
        result, _ = auditor.parse_block(lines)
        assert result["name"] == ['"test"']
        assert result["max_batch_size"] == ["8"]

    def test_nested_block(self):
        lines = ["input {", '  name: "INPUT0"', "  dims: [3]", "}"]
        result, _ = auditor.parse_block(lines)
        assert "input" in result
        assert len(result["input"]) == 1
        inner = result["input"][0]
        assert inner["name"] == ['"INPUT0"']

    def test_empty_lines_and_comments_skipped(self):
        lines = ["# comment", "", "name: test", "  ", "# another"]
        result, _ = auditor.parse_block(lines)
        assert result["name"] == ["test"]

    def test_multiple_blocks_same_key(self):
        lines = [
            "input {",
            '  name: "A"',
            "}",
            "input {",
            '  name: "B"',
            "}",
        ]
        result, _ = auditor.parse_block(lines)
        assert len(result["input"]) == 2


# ===========================================================================
# analyze_config — security checks
# ===========================================================================


class TestAnalyzeConfig:
    def test_well_configured_has_fewer_findings(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text("""\
name: "good_model"
max_batch_size: 16

input {
  name: "INPUT0"
  dims: [3, 224, 224]
}

instance_group {
  kind: KIND_GPU
  count: 2
  gpus: [0]
  memory_limit_mb: 4096
}

dynamic_batching {
  max_queue_delay_microseconds: 500
}

rate_limiter {
  resources {
    name: "exec_slots"
    count: 4
  }
}

parameters {
  key: "guard_enabled"
}

parameters {
  key: "auth_token"
}

parameters {
  key: "redact_pii"
}
""")
        report = auditor.analyze_config(config)
        warn_count = sum(1 for f in report.findings if f.severity == "WARN")
        assert warn_count == 0

    def test_missing_max_batch_warns(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "model"\n')
        report = auditor.analyze_config(config)
        msgs = [f.message for f in report.findings]
        assert any("max_batch_size" in m for m in msgs)

    def test_missing_dynamic_batching_warns(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "model"\nmax_batch_size: 8\n')
        report = auditor.analyze_config(config)
        msgs = [f.message for f in report.findings]
        assert any("dynamic_batching" in m for m in msgs)

    def test_missing_rate_limiter_warns(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "model"\nmax_batch_size: 8\n')
        report = auditor.analyze_config(config)
        msgs = [f.message for f in report.findings]
        assert any("rate_limiter" in m for m in msgs)

    def test_missing_instance_group_warns(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "model"\nmax_batch_size: 8\n')
        report = auditor.analyze_config(config)
        msgs = [f.message for f in report.findings]
        assert any("instance_group" in m.lower() or "instance" in m.lower() for m in msgs)

    def test_input_without_dims_warns(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text("""\
name: "model"
max_batch_size: 8
input {
  name: "INPUT0"
}
""")
        report = auditor.analyze_config(config)
        msgs = [f.message for f in report.findings]
        assert any("dimension" in m.lower() or "dims" in m.lower() for m in msgs)

    def test_missing_parameters_warns(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "model"\nmax_batch_size: 8\n')
        report = auditor.analyze_config(config)
        msgs = [f.message for f in report.findings]
        assert any("parameters" in m.lower() for m in msgs)


# ===========================================================================
# iter_targets
# ===========================================================================


class TestIterTargets:
    def test_finds_files_by_path(self, tmp_path):
        f = tmp_path / "config.pbtxt"
        f.touch()
        paths = auditor.iter_targets([str(f)])
        assert len(paths) == 1

    def test_empty_directory_returns_empty(self, tmp_path):
        paths = auditor.iter_targets([str(tmp_path)])
        assert len(paths) == 0

    def test_directory_finds_config_files(self, tmp_path):
        sub = tmp_path / "model1"
        sub.mkdir()
        (sub / "config.pbtxt").touch()
        paths = auditor.iter_targets([str(tmp_path)])
        assert len(paths) == 1


# ===========================================================================
# CLI main
# ===========================================================================


class TestTritonMain:
    def test_main_with_valid_config(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "test"\nmax_batch_size: 8\n')
        exit_code = auditor.main([str(config)])
        assert exit_code == 0

    def test_main_summary_mode(self, tmp_path):
        config = tmp_path / "config.pbtxt"
        config.write_text('name: "test"\nmax_batch_size: 8\n')
        exit_code = auditor.main([str(config), "--summary"])
        assert exit_code == 0

    def test_main_no_files_returns_1(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        exit_code = auditor.main([str(empty_dir)])
        assert exit_code == 1
