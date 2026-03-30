"""
Tests for triton_config_auditor.py

Coverage targets
----------------
- preprocess: bracket normalisation for pbtxt list syntax
- parse_block: primitives, nested blocks, comment skipping, block termination
- load_config: round-trip from text through preprocess -> parse_block
- get_single_value: returns first value, returns None for missing key
- to_int: valid int, string with quotes/whitespace, None input, non-numeric
- analyze_config: every finding category triggered independently:
    * missing max_batch_size
    * non-positive max_batch_size
    * no instance_group
    * instance_group missing kind / count / memory_limit_mb / gpus
    * no dynamic_batching
    * dynamic_batching missing max_queue_delay_microseconds
    * no rate_limiter
    * no input schema
    * input missing dims
    * model_transaction_policy absent (INFO)
    * model_transaction_policy missing decoupled
    * parameters block missing
    * parameters missing guard/auth/redact keys
- iter_targets: expands glob, handles directory with config.pbtxt, deduplicates
- summary mode: only WARN/ERROR counted
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import triton_config_auditor as auditor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: Path, text: str) -> Path:
    path.write_text(dedent(text))
    return path


def _findings_of_severity(report, severity: str) -> list:
    return [f for f in report.findings if f.severity == severity]


def _minimal_valid_config() -> str:
    """A fully-specified config that should produce zero WARN/ERROR findings."""
    return """
        max_batch_size: 8

        instance_group {
          kind: KIND_GPU
          count: 1
          memory_limit_mb: 4096
          gpus: [ 0 ]
        }

        dynamic_batching {
          max_queue_delay_microseconds: 5000
        }

        rate_limiter {
          resources {
            name: "inference"
            global_limit: 100
          }
        }

        input {
          name: "INPUT0"
          dims: [ 1, 512 ]
        }

        model_transaction_policy {
          decoupled: false
        }

        parameters {
          key: "guard_config"
          value { string_value: "/opt/guardrails.yaml" }
        }
        parameters {
          key: "auth_token"
          value { string_value: "header:Authorization" }
        }
        parameters {
          key: "redact_patterns"
          value { string_value: "pii" }
        }
    """


# ===========================================================================
# preprocess
# ===========================================================================

class TestPreprocess:
    def test_bracket_open_normalised(self):
        result = auditor.preprocess("dims: [{ name: x }]")
        assert "[{" not in result
        assert "{" in result

    def test_bracket_close_normalised(self):
        result = auditor.preprocess("dims: [{ name: x }]")
        assert "}]" not in result
        assert "}" in result

    def test_plain_text_unchanged(self):
        text = "max_batch_size: 8\n"
        assert auditor.preprocess(text) == text


# ===========================================================================
# parse_block
# ===========================================================================

class TestParseBlock:
    def test_simple_key_value(self):
        lines = ["max_batch_size: 8"]
        data, _ = auditor.parse_block(lines)
        assert data["max_batch_size"] == ["8"]

    def test_multiple_values_same_key(self):
        lines = ["gpus: 0", "gpus: 1"]
        data, _ = auditor.parse_block(lines)
        assert data["gpus"] == ["0", "1"]

    def test_comments_are_skipped(self):
        lines = ["# this is a comment", "key: value"]
        data, _ = auditor.parse_block(lines)
        assert "key" in data
        assert "#" not in data

    def test_empty_lines_are_skipped(self):
        lines = ["", "   ", "key: value"]
        data, _ = auditor.parse_block(lines)
        assert "key" in data

    def test_nested_block_parsed(self):
        lines = [
            "outer {",
            "  inner: 42",
            "}",
        ]
        data, _ = auditor.parse_block(lines)
        assert "outer" in data
        nested = data["outer"][0]
        assert isinstance(nested, dict)
        assert nested["inner"] == ["42"]

    def test_closing_brace_terminates_block(self):
        lines = ["a: 1", "}", "b: 2"]
        data, consumed = auditor.parse_block(lines)
        assert "a" in data
        assert "b" not in data  # outer scope, not inside this block

    def test_block_without_values_uses_empty_string(self):
        lines = ["orphan_key"]
        data, _ = auditor.parse_block(lines)
        assert data["orphan_key"] == [""]


# ===========================================================================
# get_single_value
# ===========================================================================

class TestGetSingleValue:
    def test_returns_first_value(self):
        block = {"key": ["first", "second"]}
        assert auditor.get_single_value(block, "key") == "first"

    def test_returns_none_for_missing_key(self):
        assert auditor.get_single_value({}, "missing") is None

    def test_returns_none_for_empty_list(self):
        assert auditor.get_single_value({"key": []}, "key") is None


# ===========================================================================
# to_int
# ===========================================================================

class TestToInt:
    def test_plain_integer(self):
        assert auditor.to_int("8") == 8

    def test_quoted_integer(self):
        assert auditor.to_int('"16"') == 16

    def test_negative_integer(self):
        assert auditor.to_int("-1") == -1

    def test_none_input_returns_none(self):
        assert auditor.to_int(None) is None

    def test_non_numeric_returns_none(self):
        assert auditor.to_int("abc") is None

    def test_whitespace_stripped(self):
        assert auditor.to_int("  42  ") == 42


# ===========================================================================
# analyze_config — individual finding triggers
# ===========================================================================

class TestAnalyzeConfig:
    def _analyze(self, text: str, tmp_dir: Path) -> auditor.ConfigReport:
        path = _write_config(tmp_dir / "config.pbtxt", text)
        return auditor.analyze_config(path)

    # --- max_batch_size ---

    def test_missing_max_batch_size_warns(self, tmp_dir):
        report = self._analyze("", tmp_dir)
        messages = [f.message for f in report.findings]
        assert any("max_batch_size missing" in m for m in messages)

    def test_zero_max_batch_size_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: 0", tmp_dir)
        messages = [f.message for f in report.findings]
        assert any("non-positive" in m for m in messages)

    def test_negative_max_batch_size_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: -1", tmp_dir)
        messages = [f.message for f in report.findings]
        assert any("non-positive" in m for m in messages)

    def test_positive_max_batch_size_no_batch_warn(self, tmp_dir):
        report = self._analyze("max_batch_size: 8", tmp_dir)
        batch_warns = [f for f in report.findings
                       if "max_batch_size" in f.message and "non-positive" in f.message]
        assert len(batch_warns) == 0

    # --- instance_group ---

    def test_no_instance_group_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: 1", tmp_dir)
        assert any("instance_group" in f.message for f in report.findings)

    def test_instance_group_missing_kind_warns(self, tmp_dir):
        text = "max_batch_size: 1\ninstance_group {\n  count: 1\n  memory_limit_mb: 4096\n}"
        report = self._analyze(text, tmp_dir)
        assert any("missing kind" in f.message for f in report.findings)

    def test_instance_group_missing_count_warns(self, tmp_dir):
        text = "max_batch_size: 1\ninstance_group {\n  kind: KIND_GPU\n  memory_limit_mb: 4096\n}"
        report = self._analyze(text, tmp_dir)
        assert any("missing count" in f.message for f in report.findings)

    def test_instance_group_missing_memory_limit_warns(self, tmp_dir):
        text = "max_batch_size: 1\ninstance_group {\n  kind: KIND_GPU\n  count: 1\n}"
        report = self._analyze(text, tmp_dir)
        assert any("memory_limit_mb" in f.message for f in report.findings)

    def test_kind_gpu_without_gpu_ids_warns(self, tmp_dir):
        text = "max_batch_size: 1\ninstance_group {\n  kind: KIND_GPU\n  count: 1\n  memory_limit_mb: 4096\n}"
        report = self._analyze(text, tmp_dir)
        assert any("no GPU IDs" in f.message for f in report.findings)

    # --- dynamic_batching ---

    def test_no_dynamic_batching_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: 1", tmp_dir)
        assert any("dynamic_batching" in f.message for f in report.findings)

    def test_dynamic_batching_missing_delay_warns(self, tmp_dir):
        text = "max_batch_size: 1\ndynamic_batching {\n}"
        report = self._analyze(text, tmp_dir)
        assert any("max_queue_delay_microseconds" in f.message for f in report.findings)

    def test_zero_delay_warns(self, tmp_dir):
        text = "max_batch_size: 1\ndynamic_batching {\n  max_queue_delay_microseconds: 0\n}"
        report = self._analyze(text, tmp_dir)
        assert any("max_queue_delay_microseconds" in f.message for f in report.findings)

    # --- rate_limiter ---

    def test_no_rate_limiter_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: 1", tmp_dir)
        assert any("rate_limiter" in f.message for f in report.findings)

    def test_rate_limiter_without_resources_warns(self, tmp_dir):
        text = "max_batch_size: 1\nrate_limiter {\n}"
        report = self._analyze(text, tmp_dir)
        assert any("resources" in f.message for f in report.findings)

    # --- input schema ---

    def test_no_input_schema_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: 1", tmp_dir)
        assert any("No input schema" in f.message for f in report.findings)

    def test_input_missing_dims_warns(self, tmp_dir):
        text = "max_batch_size: 1\ninput {\n  name: TOKENS\n}"
        report = self._analyze(text, tmp_dir)
        assert any("missing dimension constraints" in f.message for f in report.findings)

    def test_input_with_dims_no_warn(self, tmp_dir):
        text = "max_batch_size: 1\ninput {\n  name: TOKENS\n  dims: [ 512 ]\n}"
        report = self._analyze(text, tmp_dir)
        assert not any("missing dimension constraints" in f.message for f in report.findings)

    # --- model_transaction_policy ---

    def test_no_policy_emits_info(self, tmp_dir):
        report = self._analyze("max_batch_size: 1", tmp_dir)
        info_messages = [f.message for f in report.findings if f.severity == "INFO"]
        assert any("model_transaction_policy" in m for m in info_messages)

    def test_policy_missing_decoupled_emits_info(self, tmp_dir):
        text = "max_batch_size: 1\nmodel_transaction_policy {\n}"
        report = self._analyze(text, tmp_dir)
        info_messages = [f.message for f in report.findings if f.severity == "INFO"]
        assert any("decoupled" in m for m in info_messages)

    # --- parameters ---

    def test_no_parameters_block_warns(self, tmp_dir):
        report = self._analyze("max_batch_size: 1", tmp_dir)
        assert any("parameters block missing" in f.message for f in report.findings)

    def test_parameters_missing_guard_key_warns(self, tmp_dir):
        text = "max_batch_size: 1\nparameters {\n  key: auth\n}"
        report = self._analyze(text, tmp_dir)
        assert any("guard" in f.message for f in report.findings)

    def test_parameters_missing_auth_key_warns(self, tmp_dir):
        text = "max_batch_size: 1\nparameters {\n  key: guard_config\n}"
        report = self._analyze(text, tmp_dir)
        assert any("auth" in f.message for f in report.findings)

    def test_parameters_missing_redact_key_warns(self, tmp_dir):
        text = "max_batch_size: 1\nparameters {\n  key: guard_config\n}\nparameters {\n  key: auth\n}"
        report = self._analyze(text, tmp_dir)
        assert any("redact" in f.message for f in report.findings)

    # --- minimal valid config ---

    def test_fully_valid_config_has_no_warn_or_error(self, tmp_dir):
        report = self._analyze(_minimal_valid_config(), tmp_dir)
        warn_errors = [f for f in report.findings if f.severity in ("WARN", "ERROR")]
        # Allow zero WARN/ERROR findings for the complete config
        assert len(warn_errors) == 0


# ===========================================================================
# iter_targets
# ===========================================================================

class TestIterTargets:
    def test_direct_file_resolved(self, tmp_dir):
        path = tmp_dir / "config.pbtxt"
        path.write_text("max_batch_size: 1\n")
        results = auditor.iter_targets([str(path)])
        assert path.resolve() in results

    def test_directory_recursively_finds_config_pbtxt(self, tmp_dir):
        sub = tmp_dir / "model"
        sub.mkdir()
        config = sub / "config.pbtxt"
        config.write_text("max_batch_size: 1\n")
        results = auditor.iter_targets([str(tmp_dir)])
        assert config.resolve() in results

    def test_deduplicates_results(self, tmp_dir):
        path = tmp_dir / "config.pbtxt"
        path.write_text("max_batch_size: 1\n")
        results = auditor.iter_targets([str(path), str(path)])
        assert results.count(path.resolve()) == 1

    def test_nonexistent_glob_returns_empty(self, tmp_dir):
        results = auditor.iter_targets(["*.nonexistent_xyz"])
        assert results == []


# ===========================================================================
# ConfigFinding / ConfigReport helpers
# ===========================================================================

class TestConfigReport:
    def test_add_appends_finding(self):
        report = auditor.ConfigReport(path=Path("test.pbtxt"))
        report.add("WARN", "Something is wrong")
        assert len(report.findings) == 1
        assert report.findings[0].severity == "WARN"

    def test_str_representation_contains_severity_and_message(self):
        finding = auditor.ConfigFinding(severity="ERROR", message="critical issue")
        s = str(finding)
        assert "ERROR" in s
        assert "critical issue" in s
