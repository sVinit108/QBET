import math
import torch


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hid_dim: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(
            emb_dim, n_heads, hid_dim, dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, n_layers)

        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device
        )

        x = self.transformer_encoder(x, mask=causal_mask)

        # Only return output of the final word
        return self.project(x[:, -1, :]), None


class FNet(torch.nn.Module):
    def __init__(
        self, emb_dim: int, hid_dim: int, n_layers: int, vocab_size: int, dropout: float
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        self.dropout = torch.nn.Dropout(dropout)

        self.ffs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    self.dropout,
                    torch.nn.Linear(emb_dim, hid_dim),
                    torch.nn.ReLU(),
                    self.dropout,
                    torch.nn.Linear(hid_dim, emb_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.ln1s = torch.nn.ModuleList(
            [torch.nn.LayerNorm([emb_dim]) for _ in range(n_layers)]
        )
        self.ln2s = torch.nn.ModuleList(
            [torch.nn.LayerNorm([emb_dim]) for _ in range(n_layers)]
        )

        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))

        for ff, ln1, ln2 in zip(self.ffs, self.ln1s, self.ln2s):
            x = ln1(torch.fft.fft2(x).real + x)
            x = ln2(ff(x) + x)

        # Only return output of the final word
        return self.project(x[:, -1, :]), None


class LSTM(torch.nn.Module):
    def __init__(
        self, emb_dim: int, hid_dim: int, n_layers: int, vocab_size: int, dropout: float
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)

        self.rnn = torch.nn.LSTM(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )

        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        rnn_op, _ = self.rnn(x)

        # Only return output of the final word
        return self.project(rnn_op[:, -1, :]), None


# LINFORMER

class LinformerAttention(torch.nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, seq_len: int,
                 k: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.k = k
        self.scale = math.sqrt(self.head_dim)

        self.W_Q = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O = torch.nn.Linear(emb_dim, emb_dim, bias=False)

        # Low-rank projection matrices: seq_len -> k (shared across heads)
        self.E = torch.nn.Parameter(torch.randn(seq_len, k))
        self.F_proj = torch.nn.Parameter(torch.randn(seq_len, k))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, n, _ = x.shape

        Q = self.W_Q(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        causal_mask = torch.triu(
            torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        F = self.F_proj[:n, :]
        V_proj = torch.einsum('bhnf,nk->bhkf', V, F)

        E = self.E[:n, :]
        attn_proj = torch.einsum('bhnm,mk->bhnk', attn, E)

        context = torch.matmul(attn_proj, V_proj)
        context = context.transpose(1, 2).contiguous().view(B, n, -1)
        return self.W_O(context)


class Linformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hid_dim: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float,
        seq_len: int = 32,
        k: int = 64,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        self.attns = torch.nn.ModuleList([
            LinformerAttention(emb_dim, n_heads, seq_len, k, dropout)
            for _ in range(n_layers)
        ])
        self.ffs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, hid_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hid_dim, emb_dim),
                torch.nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.ln1s = torch.nn.ModuleList(
            [torch.nn.LayerNorm(emb_dim) for _ in range(n_layers)]
        )
        self.ln2s = torch.nn.ModuleList(
            [torch.nn.LayerNorm(emb_dim) for _ in range(n_layers)]
        )
        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))

        for attn, ff, ln1, ln2 in zip(self.attns, self.ffs, self.ln1s, self.ln2s):
            x = ln1(x + attn(x))
            x = ln2(x + ff(x))

        # Only return output of the final word
        return self.project(x[:, -1, :]), None

# PERFORMER

class FAVORPlusAttention(torch.nn.Module):
    def __init__(self, emb_dim: int, n_heads: int,
                 num_features: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.num_features = num_features

        self.W_Q = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O = torch.nn.Linear(emb_dim, emb_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            'omega',
            torch.randn(num_features, self.head_dim)
        )

    def _phi(self, x):
        """
        FAVOR+ random feature map.
        FIX: Scale inputs by d^{-0.25} to correctly approximate scaled
        dot-product attention kernel exp(q^T k / sqrt(d)).
        """
        x_scaled = x / (self.head_dim ** 0.25)

        if self.training:

            omega = torch.randn(
                self.num_features, self.head_dim,
                device=x.device, dtype=x.dtype
            )
        else:
            omega = self.omega.to(x.dtype)

        proj = torch.matmul(x_scaled, omega.T)
        norm_sq = (x_scaled ** 2).sum(dim=-1, keepdim=True) / 2
        return torch.exp(proj - norm_sq) / math.sqrt(self.num_features)

    def forward(self, x):
        B, n, _ = x.shape

        Q = self.W_Q(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)

        Q_prime = self._phi(Q)
        K_prime = self._phi(K)

        KV_outer = torch.einsum('bhnm,bhnd->bhnmd', K_prime, V)

        KV_cumsum = torch.cumsum(KV_outer, dim=2)  

        QKV = torch.einsum('bhnm,bhnmd->bhnd', Q_prime, KV_cumsum)

        K_cumsum = torch.cumsum(K_prime, dim=2)
        normalizer = torch.einsum(
            'bhnm,bhnm->bhn', Q_prime, K_cumsum
        ).unsqueeze(-1).clamp(min=1e-6)        

        context = (QKV / normalizer).transpose(1, 2).contiguous().view(B, n, -1)
        return self.W_O(context)


class Performer(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hid_dim: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float,
        num_features: int = 256,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        self.attns = torch.nn.ModuleList([
            FAVORPlusAttention(emb_dim, n_heads, num_features, dropout)
            for _ in range(n_layers)
        ])
        self.ffs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, hid_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hid_dim, emb_dim),
                torch.nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.ln1s = torch.nn.ModuleList(
            [torch.nn.LayerNorm(emb_dim) for _ in range(n_layers)]
        )
        self.ln2s = torch.nn.ModuleList(
            [torch.nn.LayerNorm(emb_dim) for _ in range(n_layers)]
        )
        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))

        for attn, ff, ln1, ln2 in zip(self.attns, self.ffs, self.ln1s, self.ln2s):
            x = ln1(x + attn(x))
            x = ln2(x + ff(x))

        # Only return output of the final word
        return self.project(x[:, -1, :]), None