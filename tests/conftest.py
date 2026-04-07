"""
Shared fixtures and helpers for the ml-security-tools test suite.

Design contract
---------------
- No GPU required: all tensors live on CPU.
- No model downloads: every model is constructed from nn.Module primitives.
- No external services: sockets, filesystem writes, and subprocess calls
  are mocked at the boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Ensure the project root is importable without an editable install.
# The six tool scripts live at the repo root, not inside src/.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Minimal in-process models that need no downloads
# ---------------------------------------------------------------------------


class TinyLinear(nn.Module):
    """Bare-bones two-layer classifier.  Deterministically initialised."""

    def __init__(self, in_features: int = 8, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, num_classes)
        # Fixed weights so tests are reproducible
        nn.init.constant_(self.fc1.weight, 0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.constant_(self.fc2.weight, 0.05)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TinyConvNet(nn.Module):
    """Minimal vision-style conv classifier (no pre-trained weights)."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x).flatten(1)
        return self.classifier(feats)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_linear() -> TinyLinear:
    model = TinyLinear()
    model.eval()
    return model


@pytest.fixture()
def tiny_conv() -> TinyConvNet:
    model = TinyConvNet()
    model.eval()
    return model


@pytest.fixture()
def batch_inputs_1d() -> torch.Tensor:
    """Shape (4, 8) — compatible with TinyLinear."""
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.fixture()
def batch_inputs_2d() -> torch.Tensor:
    """Shape (2, 3, 16, 16) — compatible with TinyConvNet."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 16, 16)


@pytest.fixture()
def batch_targets_4class() -> torch.Tensor:
    return torch.tensor([0, 1, 2, 3])


@pytest.fixture()
def batch_targets_2() -> torch.Tensor:
    return torch.tensor([0, 1])


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Alias so tests can request a named temp directory."""
    return tmp_path


@pytest.fixture()
def simple_state_dict() -> dict[str, torch.Tensor]:
    """A minimal state dict with known, clean values."""
    return {
        "weight": torch.ones(4, 4) * 0.5,
        "bias": torch.zeros(4),
    }


@pytest.fixture()
def poisoned_state_dict() -> dict[str, torch.Tensor]:
    """State dict with NaN, Inf, and extreme values for anomaly tests."""
    d: dict[str, torch.Tensor] = {
        "weight": torch.ones(4, 4) * 0.5,
        "bias": torch.zeros(4),
    }
    d["weight"][0, 0] = float("nan")
    d["weight"][1, 1] = float("inf")
    d["weight"][2, 2] = 999.0
    return d
