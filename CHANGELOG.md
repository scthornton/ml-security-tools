# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-30

### Added
- Proper Python package structure (`pip install mlsec`)
- Unified CLI entry point (`mlsec <tool>`) with subcommands and aliases
- `pyproject.toml` with optional dependency groups (`vision`, `transformers`, `onnx`, `safetensors`, `all`, `dev`)
- Comprehensive test suite (289 tests, 79% coverage)
- GitHub Actions CI pipeline (lint, test matrix, type checking)
- All tools now accept `argv` parameter for programmatic use

### Fixed
- **PGD attack gradient crash** — `adv.grad.zero_()` on `None` after tensor reassignment in projection step
- **Checkpoint NaN masking** — `tensor.abs().max()` returned NaN when tensor contained NaN, masking extreme finite values. Now uses finite-only filtering.
- **Checkpoint histogram crash** — `torch.histc` crashed on tensors containing NaN/Inf values. Now filters non-finite values before computing histograms.
- **Dead code in `extract_state_dict`** — `"state_dict"` key check was unreachable because generic dict check matched first. Reordered to check `"state_dict"` key before generic dict extraction.
- **Triton config quote stripping** — Parameter keys from pbtxt retained surrounding quotes, causing auth/guard/redact pattern matching to fail.
- **Triton auth check** — Changed from exact match to substring match for auth parameter detection (consistent with guard and redact checks).

### Changed
- Restructured from 6 loose scripts to `src/mlsec/` package layout
- Original standalone scripts preserved at root for backward compatibility

## [1.0.0] - 2025-10-30

### Added
- Initial release of ML security tools
- Six specialized security analysis scripts
- Support for PyTorch, TensorRT, and Triton models
- Model inspection and auditing capabilities

[2.0.0]: https://github.com/scthornton/ml-security-tools/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/scthornton/ml-security-tools/releases/tag/v1.0.0
