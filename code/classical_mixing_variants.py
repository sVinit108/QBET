import torch
import torch.nn as nn


class _BaseMixing(nn.Module):
    def __init__(self, d_model: int, n_qubits: int):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.learned_pos_embed = nn.Parameter(torch.randn(512, n_qubits) * 0.02)
        self.pos_proj = nn.Linear(n_qubits, d_model, bias=False)
        self.residual_gate = nn.Parameter(torch.tensor(0.7))

    def _pos_bias(self, positions: torch.Tensor) -> torch.Tensor:
        idx = positions.long().clamp(0, self.learned_pos_embed.shape[0] - 1)
        return self.pos_proj(self.learned_pos_embed[idx])

    def _residual(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = self.residual_gate
        return g * out + (1.0 - g) * x


class LinearMixing(_BaseMixing):
    """
    Linear bottleneck: d_model -> n_qubits -> d_model (no activation).
    """
    def __init__(self, d_model: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__(d_model, n_qubits)
        self.down = nn.Linear(d_model, n_qubits)
        self.up = nn.Linear(n_qubits, d_model)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x + self._pos_bias(positions)))
        return self._residual(out, x)


class MLP1xMixing(_BaseMixing):
    """
    Two-layer MLP parameter-matched to PAQM.
    """
    _H = 8

    def __init__(self, d_model: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__(d_model, n_qubits)
        h = self._H
        self.fc1 = nn.Linear(d_model, h)
        self.fc2 = nn.Linear(h, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.fc1(x + self._pos_bias(positions))))
        out = self.fc2(h)
        return self._residual(out, x)


class DeepMLP1xMixing(_BaseMixing):
    """
    Three-layer MLP parameter-matched to PAQM.
    """
    _H = 8

    def __init__(self, d_model: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__(d_model, n_qubits)
        h = self._H
        self.fc1 = nn.Linear(d_model, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x + self._pos_bias(positions)))
        h2 = self.dropout(self.act(self.fc2(h1))) + h1
        out = self.fc3(h2)
        return self._residual(out, x)


class MLP2xMixing(_BaseMixing):
    """
    Two-layer MLP with approximately 2x PAQM parameters.
    """
    _H = 26

    def __init__(self, d_model: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__(d_model, n_qubits)
        h = self._H
        self.fc1 = nn.Linear(d_model, h)
        self.fc2 = nn.Linear(h, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.fc1(x + self._pos_bias(positions))))
        out = self.fc2(h)
        return self._residual(out, x)


if __name__ == "__main__":
    d, nq, nl = 128, 4, 2
    PAQM_TOTAL = 4777

    rows = [
        ("PAQM (reference)", None, PAQM_TOTAL),
        ("LinearMixing", LinearMixing, None),
        ("MLP1xMixing", MLP1xMixing, None),
        ("DeepMLP1xMixing", DeepMLP1xMixing, None),
        ("MLP2xMixing (~2x)", MLP2xMixing, None),
    ]

    print(f"\n{'Variant':<20} {'Params':>8}  {'Diff':>8}  {'%':>7}")
    print("-" * 50)
    for name, cls, fixed in rows:
        p = fixed if cls is None else sum(v.numel() for v in cls(d, nq, nl).parameters())
        diff = p - PAQM_TOTAL
        print(f"{name:<20} {p:>8}  {diff:>+8}  {100*diff/PAQM_TOTAL:>+6.2f}%")

    print("\nRunning forward pass smoke tests (batch=4, seq=32, d=128)...")
    x = torch.randn(4, 32, d)
    pos = torch.arange(32).unsqueeze(0).expand(4, -1)
    
    for name, cls, _ in rows:
        if cls is None:
            continue
        m = cls(d, nq, nl)
        out = m(x, pos)
        assert out.shape == (4, 32, d), f"Shape error {name}: {out.shape}"
        assert not torch.allclose(out, x), f"Output == input for {name}"
        g = m.residual_gate.item()
        print(f"  {name:<20}  {tuple(out.shape)}  gate={g:.2f}  [PASS]")