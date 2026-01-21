# ML Security Tools

Collection of Python security analysis tools for machine learning models and infrastructure.

## 🛠️ Tools

### FGSM Regression Harness (`fgsm_regression_harness.py`)
Tests adversarial robustness of ML models using Fast Gradient Sign Method (FGSM) attacks. Validates model resilience against adversarial examples.

### Model Inspection (`model-inspection.py`)
Analyzes ML model architecture, parameters, and configuration. Provides detailed insights into model structure and potential security concerns.

### Distributed Poison Monitor (`distributed_poison_monitor.py`)
Detects potential poisoning attacks in distributed training environments. Monitors for malicious data injection and model corruption.

### TensorRT Export Guard (`tensorrt_export_guard.py`)
Validates security of TensorRT model exports. Ensures safe conversion and deployment of optimized models.

### Torch Checkpoint Triage (`torch_checkpoint_triage.py`)
Analyzes PyTorch checkpoint files for security issues. Detects potential malicious code or suspicious patterns in saved models.

### Triton Config Auditor (`triton_config_auditor.py`)
Audits Triton Inference Server configurations for security best practices. Identifies misconfigurations and potential vulnerabilities.

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/scthornton/ml-security-tools.git
cd ml-security-tools

# Install dependencies (each tool specifies its own requirements)
pip install torch torchvision tensorrt tritonclient
```

## 🚀 Usage

Each tool is standalone and can be run independently:

```bash
# Example: Run FGSM regression harness
python fgsm_regression_harness.py --model path/to/model.pt

# Example: Inspect model
python model-inspection.py --model path/to/model.pt

# Example: Monitor for poisoning
python distributed_poison_monitor.py --dataset path/to/data
```

Refer to each tool's `--help` flag for specific usage instructions.

## ⚠️ Responsible Use

These tools are designed for authorized security testing only:

✅ **Authorized Use:**
- Security testing of your own ML models
- Authorized penetration testing
- Security research in controlled environments
- Educational purposes

❌ **Prohibited:**
- Unauthorized testing of production systems
- Attacking third-party ML services
- Malicious exploitation

See [SECURITY.md](SECURITY.md) for full responsible use guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Email:** scott@perfecxion.ai
- **Alternative:** scthornton@gmail.com
- **Security Issues:** See [SECURITY.md](SECURITY.md)

## 🔗 Related Projects

- **MetaLLM** - AI/ML security testing framework
- **model-extraction-research** - Network-based model extraction research
