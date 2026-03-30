"""Tests for tensorrt_export_guard.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tensorrt_export_guard as guard  # noqa: E402


# ===========================================================================
# parse_shape
# ===========================================================================

class TestParseShape:
    def test_valid_shape(self):
        assert guard.parse_shape("1,3,224,224") == (1, 3, 224, 224)

    def test_single_dim(self):
        assert guard.parse_shape("32") == (32,)

    def test_with_spaces(self):
        assert guard.parse_shape("1, 3, 16, 16") == (1, 3, 16, 16)

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            guard.parse_shape("a,b,c")


# ===========================================================================
# describe_tensor
# ===========================================================================

class TestDescribeTensor:
    def test_output_keys(self):
        t = torch.randn(2, 3)
        desc = guard.describe_tensor(t)
        assert "shape" in desc
        assert "dtype" in desc
        assert "min" in desc
        assert "max" in desc
        assert "mean" in desc
        assert "std" in desc

    def test_shape_matches(self):
        t = torch.randn(4, 8)
        desc = guard.describe_tensor(t)
        assert desc["shape"] == [4, 8]


# ===========================================================================
# state_dict_hash
# ===========================================================================

class TestStateDictHash:
    def test_deterministic(self):
        model = nn.Linear(4, 2)
        h1 = guard.state_dict_hash(model)
        h2 = guard.state_dict_hash(model)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_weights_different_hash(self):
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        nn.init.constant_(m1.weight, 0.0)
        nn.init.constant_(m2.weight, 1.0)
        assert guard.state_dict_hash(m1) != guard.state_dict_hash(m2)


# ===========================================================================
# file_sha256
# ===========================================================================

class TestFileSha256:
    def test_known_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = guard.file_sha256(f)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00" * 100)
        assert guard.file_sha256(f) == guard.file_sha256(f)


# ===========================================================================
# build_dynamic_axes
# ===========================================================================

class TestBuildDynamicAxes:
    def test_empty_returns_none(self):
        assert guard.build_dynamic_axes([]) is None

    def test_single_axis(self):
        result = guard.build_dynamic_axes(["0:batch"])
        assert result == {"input": {0: "batch"}, "output": {0: "batch"}}

    def test_multiple_axes(self):
        result = guard.build_dynamic_axes(["0:batch", "1:seq_len"])
        assert 0 in result["input"]
        assert 1 in result["input"]

    def test_invalid_spec_raises(self):
        with pytest.raises(Exception):
            guard.build_dynamic_axes(["bad_format"])


# ===========================================================================
# create_sample_input
# ===========================================================================

class TestCreateSampleInput:
    def test_correct_shape(self):
        t = guard.create_sample_input((1, 3, 8, 8), "float32")
        assert t.shape == (1, 3, 8, 8)
        assert t.dtype == torch.float32

    def test_float16(self):
        t = guard.create_sample_input((2, 4), "float16")
        assert t.dtype == torch.float16


# ===========================================================================
# ONNX export (requires onnx)
# ===========================================================================

@pytest.mark.skipif(guard.onnx is None, reason="onnx not installed")
class TestONNXExport:
    def test_export_creates_file(self, tmp_path):
        model = nn.Linear(4, 2)
        model.eval()
        sample = torch.randn(1, 4)
        onnx_path = tmp_path / "model.onnx"
        guard.export_to_onnx(model, sample, onnx_path, opset=17, dynamic_axes=None)
        assert onnx_path.exists()

    def test_validate_onnx_passes(self, tmp_path):
        model = nn.Linear(4, 2)
        model.eval()
        sample = torch.randn(1, 4)
        onnx_path = tmp_path / "model.onnx"
        guard.export_to_onnx(model, sample, onnx_path, opset=17, dynamic_axes=None)
        guard.validate_onnx(onnx_path)  # should not raise

    def test_lint_clean_model(self, tmp_path):
        model = nn.Linear(4, 2)
        model.eval()
        sample = torch.randn(1, 4)
        onnx_path = tmp_path / "model.onnx"
        guard.export_to_onnx(model, sample, onnx_path, opset=17, dynamic_axes=None)
        findings = guard.lint_onnx_graph(onnx_path, guard.DEFAULT_ALLOWED_DOMAINS, 1e3)
        # A simple linear model should have no lint findings
        assert len(findings) == 0


# ===========================================================================
# resolve_factory
# ===========================================================================

class TestResolveFactory:
    def test_valid_factory(self, tmp_path):
        script = tmp_path / "my_model.py"
        script.write_text(
            "import torch.nn as nn\n"
            "def create_model(): return nn.Linear(4, 2)\n"
        )
        model = guard.resolve_factory(str(script), "create_model")
        assert isinstance(model, nn.Module)

    def test_missing_factory_raises(self, tmp_path):
        script = tmp_path / "empty.py"
        script.write_text("# empty\n")
        with pytest.raises(AttributeError):
            guard.resolve_factory(str(script), "create_model")

    def test_non_module_raises(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("def create_model(): return 42\n")
        with pytest.raises(TypeError):
            guard.resolve_factory(str(script), "create_model")
