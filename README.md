# mlsec — ML Security Toolkit

[![CI](https://github.com/scthornton/ml-security-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/scthornton/ml-security-tools/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Security analysis toolkit for machine learning models and infrastructure. Covers the full ML lifecycle — from adversarial robustness testing through training-time poisoning detection to deployment hardening.

```
pip install mlsec          # core (torch only)
pip install mlsec[all]     # all optional dependencies
```

## Why mlsec?

Most ML security tools focus on a single attack surface. **mlsec covers six critical areas in one toolkit**, designed to integrate into existing MLOps pipelines:

| Tool | What it catches | When to use it |
|------|----------------|----------------|
| `mlsec adversarial` | Robustness regressions from FGSM, PGD, CW attacks | Before every model release — CI gate |
| `mlsec inspect` | Suspicious weights, activation anomalies, FGSM vulnerability | Quick triage of new/third-party models |
| `mlsec poison` | Gradient divergence in distributed training (DDP) | During training — live or post-hoc |
| `mlsec export-guard` | ONNX graph tampering, supply-chain integrity | Before deploying optimized models |
| `mlsec checkpoint` | NaN/Inf injection, weight magnitude anomalies, backdoor fingerprints | Before loading any checkpoint |
| `mlsec triton` | Missing rate limits, auth gaps, DoS vectors | Before deploying inference servers |

## Quick Start

### Adversarial Robustness Testing

Track robustness drift across model versions with baseline regression:

```bash
mlsec adversarial \
  --model-script models.py \
  --factory create_resnet \
  --input-shape 1,3,224,224 \
  --num-classes 1000 \
  --attacks fgsm pgd cw \
  --epsilon 0.01 \
  --baseline-file baselines.json \
  --update-baseline
```

### Checkpoint Security Triage

Scan checkpoints for anomalies, generate fingerprints, convert to safe formats:

```bash
mlsec checkpoint /path/to/checkpoints/ \
  --convert-safetensors \
  --write-fingerprint fingerprints/ \
  --reference-fingerprint reference.json \
  --json
```

### Distributed Poisoning Detection

Monitor gradient divergence during distributed training:

```bash
# Generate demo data
mlsec poison simulate --log-dir logs/ --workers 8 --steps 500

# Analyze for poisoning signals
mlsec poison monitor --log-dir logs/ --threshold 3.0

# Live monitoring via UDP broadcast
mlsec poison listen --port 5454 --expected-workers 8
```

### ONNX/TensorRT Export Validation

Validate the full export pipeline with provenance tracking:

```bash
mlsec export-guard \
  --model-script models.py \
  --input-shape 1,3,224,224 \
  --enable-onnxruntime \
  --build-engine \
  --hash-record attestation.json
```

### Triton Server Hardening

Audit inference server configs for security gaps:

```bash
mlsec triton models/**/config.pbtxt --summary
```

### Model Inspection

Quick security triage of HuggingFace models:

```bash
mlsec inspect --allow-downloads
```

## Architecture

```
mlsec/
├── adversarial    — FGSM, PGD, CW attacks with baseline regression tracking
├── inspect        — Weight anomaly detection + activation monitoring hooks
├── poison         — Gradient snapshotter + CUSUM change-point detection
├── export-guard   — ONNX lint + SHA-256 provenance chain + trtexec integration
├── checkpoint     — KL-divergence fingerprinting + safetensors conversion
└── triton         — Heuristic config.pbtxt parser + security rule engine
```

Each tool is **standalone** (no cross-dependencies) and works both as a CLI command and as a Python library:

```python
from mlsec.tools.checkpoint_triage import inspect_state_dict, compute_fingerprint

anomalies = inspect_state_dict(model.state_dict(), threshold=100.0)
fingerprint = compute_fingerprint(model.state_dict())
```

## Installation

```bash
# Minimal (PyTorch only)
pip install mlsec

# With specific extras
pip install mlsec[vision]        # + torchvision
pip install mlsec[transformers]  # + HuggingFace transformers
pip install mlsec[onnx]          # + onnx + onnxruntime
pip install mlsec[safetensors]   # + safetensors format support
pip install mlsec[all]           # everything

# Development
pip install -e ".[dev,all]"
pytest tests/ -v
```

## CI/CD Integration

mlsec is designed to run in automated pipelines. Every tool returns structured exit codes:

| Exit Code | Meaning |
|-----------|---------|
| `0` | Clean — no anomalies detected |
| `1` | Error — invalid input or missing files |
| `2` | Alert — anomalies detected (investigate) |

Example GitHub Actions step:

```yaml
- name: Adversarial regression gate
  run: |
    mlsec adversarial \
      --model-script models.py \
      --factory create_model \
      --input-shape 1,3,224,224 \
      --num-classes 1000 \
      --attacks fgsm pgd \
      --baseline-file baselines.json
```

JSON output is available for checkpoint triage (`--json` flag) for machine-readable results.

## Key Detection Capabilities

### Adversarial Attacks
- **FGSM** — Fast Gradient Sign Method (single-step)
- **PGD** — Projected Gradient Descent (multi-step with random restarts)
- **CW** — Carlini-Wagner L-inf (optimization-based)
- Mixed-precision testing (float32, float16, bfloat16)
- Baseline tracking with configurable regression thresholds

### Poisoning Detection
- Per-worker gradient L2/L-inf norm monitoring
- Cross-worker divergence ratio computation
- **CUSUM** change-point detection for slow-burn poisoning
- Live UDP broadcast aggregation for real-time alerts

### Supply Chain Security
- SHA-256 provenance hashing at every pipeline stage
- ONNX graph linting (custom domains, control-flow ops, large constants, embedded subgraphs)
- KL-divergence fingerprinting for checkpoint drift detection
- Automatic safetensors conversion for pickle-free distribution

### Deployment Hardening
- Triton config auditing without protobuf dependency
- Checks for: rate limiting, dynamic batching, auth controls, input bounds, logging redaction
- ONNX Runtime numerical drift comparison

## Responsible Use

These tools are for **authorized security testing only**. See [SECURITY.md](SECURITY.md) for the full responsible use policy.

## Contributing

```bash
git clone https://github.com/scthornton/ml-security-tools.git
cd ml-security-tools
pip install -e ".[dev,all]"
pytest tests/ -v
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

- **Email:** scott@perfecxion.ai
- **Security issues:** See [SECURITY.md](SECURITY.md)
