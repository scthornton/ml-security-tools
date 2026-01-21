# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in these tools, please report it privately.

### How to Report

**Email:** scott@perfecxion.ai

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fixes

### Response Timeline

- **Initial Response:** Within 48 hours
- **Assessment:** Within 7 days
- **Resolution:** Based on severity

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Responsible Use

These tools are designed for authorized security testing and analysis of ML models you own or have permission to test.

### Authorized Use

✅ **Permitted:**
- Security testing of your own ML models
- Authorized penetration testing with permission
- Security research in controlled environments
- Educational purposes
- Defensive security analysis

### Prohibited Use

❌ **Not Permitted:**
- Unauthorized testing of production ML systems
- Attacking third-party ML services without permission
- Any activity violating terms of service
- Malicious use or exploitation

## Tool-Specific Security Notes

### FGSM Regression Harness
- Tests adversarial robustness of models
- Use only on models you own or have permission to test

### Model Inspection
- Analyzes model architecture and parameters
- May reveal sensitive model details - handle output securely

### Distributed Poison Monitor
- Detects potential poisoning in training data
- Run only on datasets you're authorized to analyze

### TensorRT Export Guard
- Validates TensorRT export security
- Ensures safe model deployment

### Torch Checkpoint Triage
- Analyzes PyTorch checkpoint files
- May contain sensitive information - handle securely

### Triton Config Auditor
- Audits Triton Inference Server configurations
- Use only on infrastructure you own or manage

## Contact

- **Email:** scott@perfecxion.ai
- **Alternative:** scthornton@gmail.com
