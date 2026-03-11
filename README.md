# Quantum-Based-Enhanced-Transformer-QBET
Major revision submitted at Evolving Systems (Springer), 2025
(Code to be soon uploaded)

Official implementation of **QBET (Quantum-Based Enhanced Transformer)**, a hybrid quantum-classical transformer architecture for natural language processing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

QBET integrates trainable quantum components into transformer architectures, achieving **93.3 perplexity** on Penn Treebank—outperforming both classical Transformer baselines (97.0 PPL) and quantum baselines like Quixer (122.0 PPL).

**Key Features:**
- **Position-Aware Quantum Mixing**: Variational quantum circuits (RY, RZ, RX gates + CNOT) process token embeddings
- **Sparse Quantum Attention**: Selective quantum enhancement on important tokens
- **Surrogate Gradient Training**: 10-15× faster than parameter-shift methods
- **Modular Design**: Configurable quantum components (qubits, circuit depth, attention tokens)

## Architecture
```
Input Tokens → Embedding → [Quantum Mixing → Sparse Attention → FFN] × L → Output
```

Each layer contains:
1. **Quantum Mixing Layer**: Projects embeddings to quantum amplitudes, applies parameterized quantum circuits
2. **Sparse Attention**: Classical attention + quantum enhancement on top-K tokens
3. **Feed-Forward Network**: Standard FFN (supports MoE with `num_experts>1`)

## Installation

### Requirements
- Python 3.11+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/sVinit108/QBET.git
cd QBET

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
torch>=2.0.0
torchtext>=0.15.0
torchquantum>=0.2.0
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Quick Start

### Training on Penn Treebank
```bash
python run.py -m QBET -d cuda
```

### Configuration

Edit hyperparameters in `run.py`:
```python
QBET_hparams = {
    "dimension": 128,          # Embedding dimension
    "num_heads": 2,            # Attention heads
    "num_layers": 2,           # Transformer layers
    "n_qubits": 3,             # Qubits for quantum mixing
    "attn_qubits": 3,          # Qubits for quantum attention
    "quantum_tokens": 4,       # Top-K tokens for quantum enhancement
    "batch_size": 32,
    "lr": 0.0015,
    "epochs": 30,
}
```

### Custom Dataset

Modify `setup_training.py` to load your dataset:
```python
# Replace Penn Treebank loading
raw_dset = load_dataset("your_dataset_name")
```

## Results

| Model | Dimension | Layers | Val PPL |
|-------|-----------|--------|---------|
| LSTM | 128 | 2 | 127.1 |
| FNet | 128 | 2 | 117.7 |
| Transformer | 128 | 1 | 97.0 |
| Quixer | 128 | cubic | 122.0 |
| **QBET (Ours)** | **128** | **2** | **93.3** |

### Ablation Study

| Configuration | Val PPL | Δ PPL |
|---------------|---------|-------|
| QBET (Full) | 93.3 | - |
| w/o Quantum Mixing | 121.6 | +28.3 |

## Project Structure
```
QBET/
├── QBET/
│   ├── QBET.py              # Main model architecture
│   ├── setup_training.py    # Training loop and data loading
│   ├── baseline_models.py   # LSTM, Transformer, FNet baselines
│   └── quixer_model.py      # Quixer baseline implementation
├── run.py                   # Entry point for training
├── requirements.txt         # Python dependencies
└── README.md
```

## Key Components

### Quantum Mixing Layer
```python
class PositionAwareQuantumMixing(nn.Module):
    # Projects embeddings → quantum amplitudes
    # Applies variational quantum circuit (RY, RZ, RX + CNOT)
    # Measures expectation values → projects back to embedding space
```

### Sparse Attention with Quantum Enhancement
```python
class SparseAttention(nn.Module):
    # Standard Q, K, V projections
    # Sparse pattern (local window + global tokens)
    # Quantum circuit on top-K important tokens
    # Surrogate gradient for efficient training
```

### Surrogate Gradient Training
Classical MLP approximates quantum gradients during backpropagation:
```python
# Forward: quantum measurements
quantum_output = quantum_circuit(q_device, input)

# Backward: surrogate gradients
if training:
    surrogate_grad = surrogate_network(input)
    output = quantum_output.detach() + (surrogate_grad - surrogate_grad.detach())
```

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (NVIDIA RTX 2080 or equivalent)
- RAM: 16GB
- Training time: ~14 sec/iteration on RTX A6000

**Recommended:**
- GPU: 24GB VRAM (NVIDIA RTX 3090/4090, A6000)
- RAM: 32GB+

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Built with [TorchQuantum](https://github.com/mit-han-lab/torchquantum) for quantum circuit simulation
- Baseline implementations inspired by [Quixer](https://arxiv.org/abs/2406.04305)
- Trained on Penn Treebank dataset


For questions or issues, please open a GitHub issue or contact [vdsharma_m24@ce.vjti.ac.in]

---

**Note:** This implementation uses classical simulation of quantum circuits. For deployment on actual quantum hardware, circuit compilation and noise mitigation strategies will be required.
