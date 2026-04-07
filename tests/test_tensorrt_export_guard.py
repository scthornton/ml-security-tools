"""
Tests for tensorrt_export_guard.py

Coverage targets
----------------
- parse_shape: valid input, empty string, non-integer values
- build_dynamic_axes: None when empty, correct dict structure, invalid spec raises
- describe_tensor: correct keys and value types
- state_dict_hash: deterministic, different models produce different hashes
- file_sha256: matches hashlib output for known content
- create_sample_input: correct shape and dtype
- export_to_onnx: torch.onnx.export called with correct arguments (onnx mocked)
- validate_onnx: onnx.checker.check_model called (onnx mocked)
- lint_onnx_graph: returns empty list when onnx unavailable; detects custom domain,
                   control-flow ops, and large constants when onnx is available (onnx mocked)
- resolve_factory: delegates correctly (already tested more thoroughly via harness tests;
                   here we verify the TensorRT guard's own version)
- run_trtexec: returns None when trtexec not found
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tensorrt_export_guard as guard  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)
        nn.init.constant_(self.fc.weight, 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ===========================================================================
# parse_shape
# ===========================================================================


class TestParseShape:
    def test_valid_4d_shape(self):
        result = guard.parse_shape("1,3,224,224")
        assert result == (1, 3, 224, 224)

    def test_valid_1d_shape(self):
        result = guard.parse_shape("128")
        assert result == (128,)

    def test_whitespace_is_stripped(self):
        result = guard.parse_shape(" 1 , 3 , 32 , 32 ")
        assert result == (1, 3, 32, 32)

    def test_trailing_comma_ignored(self):
        result = guard.parse_shape("1,4,")
        assert result == (1, 4)

    def test_non_integer_raises(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            guard.parse_shape("1,abc,32")


# ===========================================================================
# build_dynamic_axes
# ===========================================================================


class TestBuildDynamicAxes:
    def test_empty_specs_returns_none(self):
        assert guard.build_dynamic_axes([]) is None

    def test_single_spec_parsed_correctly(self):
        result = guard.build_dynamic_axes(["0:batch"])
        assert result is not None
        assert result["input"][0] == "batch"
        assert result["output"][0] == "batch"

    def test_multiple_specs_merged(self):
        result = guard.build_dynamic_axes(["0:batch", "1:seq"])
        assert result["input"][0] == "batch"
        assert result["input"][1] == "seq"

    def test_missing_colon_raises(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            guard.build_dynamic_axes(["0batch"])

    def test_name_with_colon_uses_first_split(self):
        # "0:some:thing" -> dim=0, name="some:thing"
        result = guard.build_dynamic_axes(["0:some:thing"])
        assert result["input"][0] == "some:thing"


# ===========================================================================
# describe_tensor
# ===========================================================================


class TestDescribeTensor:
    def test_returns_all_required_keys(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        desc = guard.describe_tensor(t)
        for key in ("shape", "dtype", "min", "max", "mean", "std"):
            assert key in desc

    def test_shape_is_list(self):
        t = torch.zeros(3, 4)
        desc = guard.describe_tensor(t)
        assert desc["shape"] == [3, 4]

    def test_min_max_correct(self):
        t = torch.tensor([-1.0, 0.0, 1.0])
        desc = guard.describe_tensor(t)
        assert abs(desc["min"] - (-1.0)) < 1e-6
        assert abs(desc["max"] - 1.0) < 1e-6

    def test_dtype_is_string(self):
        t = torch.ones(2, dtype=torch.float16)
        desc = guard.describe_tensor(t)
        assert isinstance(desc["dtype"], str)
        assert "float16" in desc["dtype"]


# ===========================================================================
# state_dict_hash
# ===========================================================================


class TestStateDictHash:
    def test_hash_is_deterministic(self):
        model = TinyLinear()
        h1 = guard.state_dict_hash(model)
        h2 = guard.state_dict_hash(model)
        assert h1 == h2

    def test_different_weights_produce_different_hashes(self):
        m1 = TinyLinear()
        m2 = TinyLinear()
        with torch.no_grad():
            m2.fc.weight.fill_(0.99)
        assert guard.state_dict_hash(m1) != guard.state_dict_hash(m2)

    def test_hash_is_64_hex_chars(self):
        model = TinyLinear()
        h = guard.state_dict_hash(model)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# file_sha256
# ===========================================================================


class TestFileSha256:
    def test_matches_hashlib_for_known_content(self, tmp_dir):
        path = tmp_dir / "test.bin"
        content = b"hello world"
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert guard.file_sha256(path) == expected

    def test_different_files_produce_different_hashes(self, tmp_dir):
        p1 = tmp_dir / "a.bin"
        p2 = tmp_dir / "b.bin"
        p1.write_bytes(b"aaa")
        p2.write_bytes(b"bbb")
        assert guard.file_sha256(p1) != guard.file_sha256(p2)

    def test_empty_file_produces_known_hash(self, tmp_dir):
        path = tmp_dir / "empty.bin"
        path.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert guard.file_sha256(path) == expected


# ===========================================================================
# create_sample_input
# ===========================================================================


class TestCreateSampleInput:
    def test_shape_matches_request(self):
        t = guard.create_sample_input((1, 3, 32, 32), "float32")
        assert t.shape == (1, 3, 32, 32)

    def test_dtype_float32(self):
        t = guard.create_sample_input((2, 4), "float32")
        assert t.dtype == torch.float32

    def test_dtype_float16(self):
        t = guard.create_sample_input((2, 4), "float16")
        assert t.dtype == torch.float16


# ===========================================================================
# export_to_onnx (torch.onnx.export mocked)
# ===========================================================================


class TestExportToOnnx:
    def test_torch_onnx_export_called(self, tmp_dir):
        model = TinyLinear()
        model.eval()
        onnx_path = tmp_dir / "model.onnx"
        sample = torch.randn(1, 4)

        with patch("torch.onnx.export") as mock_export:
            guard.export_to_onnx(model, sample, onnx_path, opset=17, dynamic_axes=None)
            mock_export.assert_called_once()

    def test_export_passes_correct_opset(self, tmp_dir):
        model = TinyLinear()
        model.eval()
        onnx_path = tmp_dir / "model.onnx"
        sample = torch.randn(1, 4)

        with patch("torch.onnx.export") as mock_export:
            guard.export_to_onnx(model, sample, onnx_path, opset=11, dynamic_axes=None)
            _, kwargs = mock_export.call_args
            assert kwargs.get("opset_version") == 11

    def test_parent_dir_is_created(self, tmp_dir):
        model = TinyLinear()
        model.eval()
        nested_path = tmp_dir / "sub" / "dir" / "model.onnx"
        sample = torch.randn(1, 4)

        with patch("torch.onnx.export"):
            guard.export_to_onnx(model, sample, nested_path, opset=17, dynamic_axes=None)
        assert nested_path.parent.exists()


# ===========================================================================
# validate_onnx (onnx mocked)
# ===========================================================================


class TestValidateOnnx:
    def test_skips_when_onnx_unavailable(self, tmp_dir, caplog):
        import logging

        onnx_path = tmp_dir / "fake.onnx"
        onnx_path.write_bytes(b"fake")

        original_onnx = guard.onnx
        guard.onnx = None  # temporarily remove onnx
        try:
            with caplog.at_level(logging.WARNING, logger="tensorrt_export_guard"):
                guard.validate_onnx(onnx_path)
            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("onnx package not available" in r.message for r in warnings)
        finally:
            guard.onnx = original_onnx

    def test_calls_check_model_when_onnx_available(self, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        onnx_path.write_bytes(b"fake")

        mock_onnx = MagicMock()
        original_onnx = guard.onnx
        guard.onnx = mock_onnx
        try:
            guard.validate_onnx(onnx_path)
            mock_onnx.checker.check_model.assert_called_once()
        finally:
            guard.onnx = original_onnx


# ===========================================================================
# lint_onnx_graph (onnx mocked)
# ===========================================================================


class TestLintOnnxGraph:
    def test_returns_empty_when_onnx_unavailable(self, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        onnx_path.write_bytes(b"fake")

        original_onnx = guard.onnx
        guard.onnx = None
        try:
            findings = guard.lint_onnx_graph(onnx_path, {""}, 1000.0)
            assert findings == []
        finally:
            guard.onnx = original_onnx

    def _mock_onnx_context(self, nodes: list):
        """Return a context manager that injects a mock onnx + numpy_helper into guard."""
        import contextlib

        mock_onnx = MagicMock()
        mock_numpy_helper = MagicMock()

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_graph.node = nodes
        mock_model.graph = mock_graph
        mock_onnx.load.return_value = mock_model
        mock_onnx.AttributeProto.GRAPH = 5
        mock_onnx.AttributeProto.GRAPHS = 10

        @contextlib.contextmanager
        def _ctx():
            orig_onnx = guard.onnx
            orig_nh = guard.numpy_helper
            guard.onnx = mock_onnx
            guard.numpy_helper = mock_numpy_helper
            try:
                yield mock_onnx
            finally:
                guard.onnx = orig_onnx
                guard.numpy_helper = orig_nh

        return _ctx()

    def test_custom_domain_flagged(self, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        onnx_path.write_bytes(b"fake")

        node = MagicMock()
        node.domain = "com.custom"
        node.op_type = "CustomOp"
        node.attribute = []

        with self._mock_onnx_context([node]):
            findings = guard.lint_onnx_graph(onnx_path, {""}, 1000.0)
        assert any("Custom domain" in f for f in findings)

    def test_control_flow_op_flagged(self, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        onnx_path.write_bytes(b"fake")

        node = MagicMock()
        node.domain = ""
        node.op_type = "Loop"
        node.attribute = []

        with self._mock_onnx_context([node]):
            findings = guard.lint_onnx_graph(onnx_path, {""}, 1000.0)
        assert any("Loop" in f for f in findings)

    def test_clean_graph_returns_no_findings(self, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        onnx_path.write_bytes(b"fake")

        node = MagicMock()
        node.domain = ""
        node.op_type = "MatMul"
        node.attribute = []

        with self._mock_onnx_context([node]):
            findings = guard.lint_onnx_graph(onnx_path, {""}, 1000.0)
        assert findings == []


# ===========================================================================
# run_trtexec
# ===========================================================================


class TestRunTrtexec:
    def test_returns_none_when_trtexec_not_found(self, tmp_dir):
        with patch("shutil.which", return_value=None):
            result = guard.run_trtexec(
                onnx_path=tmp_dir / "model.onnx",
                engine_path=tmp_dir / "model.engine",
                precision="fp16",
                workspace=1024,
                extra_args=None,
            )
        assert result is None

    def test_subprocess_called_with_correct_args(self, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        engine_path = tmp_dir / "model.engine"
        # Fake a successful engine creation
        engine_path.touch()

        with patch("shutil.which", return_value="/usr/bin/trtexec"), patch("subprocess.run") as mock_run:
            guard.run_trtexec(
                onnx_path=onnx_path,
                engine_path=engine_path,
                precision="fp16",
                workspace=1024,
                extra_args=None,
            )
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "--fp16" in cmd


# ===========================================================================
# resolve_factory
# ===========================================================================


class TestResolveFactory:
    def test_missing_factory_attribute_raises(self, tmp_dir):
        script = tmp_dir / "model_script.py"
        script.write_text("def wrong_name(): pass\n")
        with pytest.raises(AttributeError, match="create_model not found"):
            guard.resolve_factory(str(script), "create_model")

    def test_non_module_return_raises_type_error(self, tmp_dir):
        script = tmp_dir / "model_script.py"
        script.write_text("def create_model(): return 'not a module'\n")
        with pytest.raises(TypeError, match=r"nn\.Module"):
            guard.resolve_factory(str(script), "create_model")

    def test_valid_factory_returns_nn_module(self, tmp_dir):
        script = tmp_dir / "model_script.py"
        script.write_text("import torch.nn as nn\ndef create_model(): return nn.Linear(4, 2)\n")
        model = guard.resolve_factory(str(script), "create_model")
        assert isinstance(model, nn.Module)
