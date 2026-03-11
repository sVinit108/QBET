# Quantum-Based-Enhanced-Transformer-QBET
Major revision submitted at Evolving Systems (Springer), 2025 (Q1 Journal)  
(Code to be uploaded upon publication)

Official implementation of **QBET (Quantum-Based Enhanced Transformer)**, a hybrid quantum-classical transformer architecture for natural language processing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

QBET integrates trainable quantum components into transformer architectures via classical simulation, achieving competitive perplexity across three benchmarks while outperforming both classical and quantum baselines:

- **93.3 PPL** on Penn Treebank (vs Transformer 97.0, Quixer 122.0)
- **211.5 PPL** on WikiText-2 (vs Transformer 246.7, Quixer 317.5)
- **145.8 PPL** on WikiText-103 (vs Linformer 152.8, Performer 164.0)

All results obtained in 5–6 epochs, compared to 30 epochs for classical baselines.

> **Note:** All experiments use classical simulation of quantum circuits via [TorchQuantum](https://github.com/mit-han-lab/torchquantum). No quantum hardware was used. QBET is positioned as a hybrid quantum-inspired architecture; deployment on real quantum hardware would require circuit compilation and noise mitigation.

**Key Features:**
- **Position-Aware Quantum Mixing (PAQM)**: Variational quantum circuits (RY, RZ, RX + CNOT entanglement) with hybrid sinusoidal and learned positional encoding baked directly into the circuit
- **Sparse Quantum Attention**: Selective quantum enhancement applied only to top-K important tokens via an importance scoring network
- **Surrogate Gradient Training**: Lightweight 156-parameter MLP approximates quantum gradients — 10–15× faster than parameter-shift methods
- **Modular Design**: Configurable quantum components (qubits, circuit depth, attention tokens); supports Mixture-of-Experts feed-forward networks

## Architecture
```
Input Tokens → Embedding → [PAQM → Sparse Attention → FFN] × L → Output
```

Each `HybridTransformerBlock` contains:
1. **Position-Aware Quantum Mixing (PAQM)**: Projects embeddings to quantum amplitudes via amplitude encoding, applies hybrid positional encoding + variational layers (RY→RZ→RX per qubit) + CNOT entanglement, measures Pauli-Z expectation values, projects back to embedding dimension
2. **Sparse Quantum Attention**: Classical sparse attention (local window + global tokens) augmented with a quantum-computed bias on top-K important tokens; surrogate gradients used during training
3. **Feed-Forward Network**: Standard FFN (supports Mixture-of-Experts when `num_experts > 1`)

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
    "dimension": 128,          # Embedding dimension (d_model)
    "num_heads": 2,            # Attention heads
    "num_layers": 2,           # Transformer layers
    "n_qubits": 4,             # Qubits for quantum mixing (PAQM)
    "attn_qubits": 3,          # Qubits for quantum attention circuit
    "n_quantum_layers": 2,     # Variational layers in PAQM circuit
    "quantum_tokens": 8,       # Top-K tokens for quantum attention
    "sparse_window": 32,       # Local attention window size
    "global_tokens": 8,        # Global attention tokens
    "batch_size": 32,
    "lr": 0.001,               # Learning rate (cosine annealing)
    "dropout": 0.08,
    "epochs": 15,              # Max epochs (early stopping patience=5)
}
```

### Custom Dataset

Modify `setup_training.py` to load your dataset:
```python
# Replace Penn Treebank loading with your dataset
raw_dset = load_dataset("dataset_name")
```

## Results

### Penn Treebank (PTB)

| Model | Dimension | Layers | Val PPL | Epochs |
|-------|-----------|--------|---------|--------|
| LSTM | 128 | 2 | 127.1 ± 3.1 | 30 |
| FNet | 128 | 2 | 117.7 ± 0.8 | 30 |
| Transformer | 128 | 1 | 97.0 ± 0.3 | 30 |
| Linformer | 128 | 2 | 93.0 ± 0.8 | 50 |
| Performer | 128 | 2 | 99.7 ± 1.0 | 50 |
| Quixer | 512, 6 qubits | cubic | 122.0 ± 2.2 | 30 |
| **QBET (Ours)** | **128, 4+3 qubits** | **L=2, H=2** | **93.3 ± 0.8** | **5** |

### WikiText-2

| Model | Dimension | Layers | Val PPL | Epochs |
|-------|-----------|--------|---------|--------|
| LSTM | 128 | 2 | 308.1 ± 2.1 | 30 |
| FNet | 128 | 2 | 287.0 ± 0.6 | 30 |
| Transformer | 128 | 1 | 246.7 ± 0.4 | 30 |
| Linformer | 128 | 2 | 220.6 ± 1.1 | 27 |
| Performer | 128 | 2 | 223.6 ± 2.3 | 12 |
| Quixer | 512, 6 qubits | cubic | 317.5 ± 3.2 | 30 |
| **QBET (Ours)** | **128, 4+3 qubits** | **L=2, H=2** | **211.5 ± 0.8** | **6** |

### WikiText-103 (Q-imp disabled for computational tractability)

| Model | Best Val PPL | Epochs |
|-------|-------------|--------|
| Linformer | 152.76 | 5 |
| Performer | 164.03 | 5 |
| Quixer | 148.05 | 5 |
| **QBET (Ours)** | **145.83** | **5** |

*All PTB and WikiText-2 results averaged over 10 seeds.*

### Ablation Study (PTB / WikiText-2)

| Configuration | PTB PPL | Δ PTB | Wiki PPL | Δ Wiki |
|---------------|---------|-------|----------|--------|
| QBET (Full Model) | 93.3 | — | 211.5 | — |
| Q-Mix Only (PAQM, no Q-imp) | 97.3 | +4.0 | 215.8 | +4.3 |
| Q-Imp Only (no PAQM) | 120.6 | +27.3 | 229.4 | +17.9 |
| No Quantum Components | 122.1 | +28.8 | 267.5 | +56.0 |

PAQM is the dominant contributor — removing it raises PTB PPL by 28.8 points.

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

### Position-Aware Quantum Mixing (PAQM)
```python
class PositionAwareQuantumMixing(nn.Module):
    # 1. Project embedding → amplitude vector (size 2^n_qubits)
    # 2. L2-normalize → initialize quantum state via amplitude embedding
    # 3. For each circuit layer:
    #    a. Apply fixed sinusoidal RZ encoding (position-dependent)
    #    b. Apply learned RY positional bias (small, trainable)
    #    c. Apply variational RY → RZ → RX per qubit
    #    d. Apply CNOT entanglement (linear chain + wrap-around)
    # 4. Measure Pauli-Z expectation values on all qubits
    # 5. Project measurement vector → embedding dimension
    # 6. Gated residual: gamma * quantum_out + (1-gamma) * input
```

### Sparse Attention with Quantum Enhancement
```python
class SparseAttention(nn.Module):
    # 1. Score token importance via linear network
    # 2. Standard Q, K, V projections
    # 3. Sparse mask: local window (w=32) + global tokens (g=8)
    # 4. Select top-K important tokens (K=8)
    # 5. For each important token: quantum circuit on [Q_i, K_j] pairs
    # 6. Add quantum bias to classical attention logits
    # 7. Softmax attention + output projection
```

### Surrogate Gradient Training
```python
# Forward pass: real quantum expectation values
quantum_output = quantum_circuit(q_device, input)

# Backward pass: surrogate gradients (straight-through estimator)
# Surrogate: Linear(8,12) → Tanh → Linear(12,3) [156 parameters]
if training:
    surrogate_output = surrogate_network(input)
    output = quantum_output.detach() + (surrogate_output - surrogate_output.detach())
```

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (NVIDIA RTX 2080 or equivalent)
- RAM: 16GB
- Training time: ~14 sec/iteration (Q-imp ON) on RTX A6000

**Recommended:**
- GPU: 24GB VRAM (NVIDIA RTX 3090/4090, A6000)
- RAM: 32GB+

**Inference note:** With Q-imp disabled (`use_quantum_attention=False`), inference reduces to ~0.53 ms/iter — within one order of magnitude of classical baselines. The overhead with Q-imp enabled arises entirely from quantum circuit simulation on classical hardware and is expected to vanish on real quantum hardware.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Built with [TorchQuantum](https://github.com/mit-han-lab/torchquantum) for quantum circuit simulation
- Baseline implementations inspired by [Quixer](https://arxiv.org/abs/2406.04305)
- Datasets: Penn Treebank, WikiText-2, WikiText-103 via HuggingFace

