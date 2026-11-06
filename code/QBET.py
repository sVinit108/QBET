import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

import warnings
warnings.filterwarnings('ignore')

class PositionAwareQuantumMixing(nn.Module):
    """
    A quantum mixing layer that replaces the PennyLane implementation with TorchQuantum.
    The core logic for amplitude embedding, hybrid positional encoding, variational layers,
    and measurement remains the same.
    """
    
    # Inner class defining the quantum circuit as a TorchQuantum module
    class QuantumCircuitModule(tq.QuantumModule):
        def __init__(self, n_qubits: int, n_layers: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers

            # Define trainable variational layers (RY, RZ, RX)
            self.variational_layers = nn.ModuleList()
            for _ in range(n_layers):
                layer = nn.ModuleList()
                for i in range(n_qubits):
                    layer.append(tq.RY(has_params=True, trainable=True))
                    layer.append(tq.RZ(has_params=True, trainable=True))
                    layer.append(tq.RX(has_params=True, trainable=True))
                self.variational_layers.append(layer)

            # Define fixed entanglement structure
            self.entanglement_layers = nn.ModuleList()
            for i in range(n_qubits - 1):
                self.entanglement_layers.append(tq.CNOT(wires=[i, i + 1]))
            if n_qubits > 1:
                self.entanglement_layers.append(tq.CNOT(wires=[n_qubits - 1, 0]))

        def forward(self, q_device: tq.QuantumDevice, inputs: torch.Tensor, pos_encoding: torch.Tensor):
            """
            Forward pass for the quantum circuit.
            
            Args:
                q_device (tq.QuantumDevice): The quantum device to run the circuit on.
                inputs (torch.Tensor): Amplitude vectors for state preparation. Shape [batch, 2^n_qubits].
                pos_encoding (torch.Tensor): Hybrid positional encoding. Shape [batch, 2, n_qubits].
            """

            inputs_normalized = F.normalize(inputs, p=2, dim=-1)
            inputs_normalized = inputs_normalized.to(dtype=torch.complex64)  # Convert to complex

            q_device.set_states(inputs_normalized)

            # 2. Multi-layer quantum processing
            for layer_idx in range(self.n_layers):

                # a) Hybrid Positional Encoding
                for i in range(self.n_qubits):
                    # Fixed sinusoidal RZ encoding
                    scaling_factor = 1.0 / (10000 ** (2 * i / self.n_qubits))
                    
                    pos_value = pos_encoding[:, 0, i]  # Use i-th qubit's position
                    fixed_angle = torch.sin(pos_value * scaling_factor) if i % 2 == 0 else torch.cos(pos_value * scaling_factor)
                    
                    # Convert to same dtype and device as quantum state
                    fixed_angle = fixed_angle.to(dtype=torch.float32)
                    tqf.rz(q_device, wires=i, params=fixed_angle)
                    
                    # Small learned position adjustment (RY)
                    learned_angle = (pos_encoding[:, 1, i] * 0.1).to(dtype=torch.float32)
                    tqf.ry(q_device, wires=i, params=learned_angle)

                # b) Variational Layers
                for i in range(self.n_qubits):
                    self.variational_layers[layer_idx][i*3+0](q_device, wires=i) # RY
                    self.variational_layers[layer_idx][i*3+1](q_device, wires=i) # RZ
                    self.variational_layers[layer_idx][i*3+2](q_device, wires=i) # RX

                # c) Entanglement
                for ent_layer in self.entanglement_layers:
                    ent_layer(q_device)

            # 3. Measurement
            # Return expectation value of PauliZ on each qubit
            observables = [tq.PauliZ() for _ in range(self.n_qubits)]
            return tq.expval(q_device, list(range(self.n_qubits)), observables)

    def __init__(self, d_model: int, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Linear map from token embedding → 2^n_qubits amplitudes
        self.embed_to_amplitudes = nn.Linear(d_model, 2 ** n_qubits)

        # HYBRID POSITIONAL ENCODING: Small learned component
        self.learned_pos_embed = nn.Parameter(torch.randn(512, n_qubits) * 0.02)

        # Instantiate the TorchQuantum device and circuit module
        self.quantum_circuit = self.QuantumCircuitModule(n_qubits, n_layers)

        # Projection back to model dimension
        self.quantum_project = nn.Linear(n_qubits, d_model)
        
        # Residual connection gate
        self.residual_gate = nn.Parameter(torch.tensor(0.7))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, record_op=False)
        batch_size, seq_len, _ = x.shape
        total_samples = batch_size * seq_len

        # Project embeddings → amplitude vectors
        amps = self.embed_to_amplitudes(x)  # [batch, seq_len, 2^n_qubits]
        
        # HYBRID POSITIONAL ENCODING: Prepare both fixed and learned components
        pos_idx = positions.long()  # [batch, seq_len]
        fixed_component = pos_idx.float().unsqueeze(-1).expand(-1, -1, self.n_qubits)  # [batch, seq, n_qubits]
        learned_component = self.learned_pos_embed[pos_idx.clamp(0, self.learned_pos_embed.shape[0]-1)]  # [batch, seq, n_qubits]
        pos_encodings = torch.stack([fixed_component, learned_component], dim=2)

        # Flatten batch+seq for batched execution
        amps_flat = amps.view(total_samples, -1)
        pos_flat = pos_encodings.view(total_samples, 2, self.n_qubits)
        
        # Set the batch size on the quantum device
        self.q_device.bsz = total_samples
        
        # Execute the quantum circuit
        q_out = self.quantum_circuit(self.q_device, amps_flat, pos_flat)

        # Reshape back
        q_out = q_out.view(batch_size, seq_len, self.n_qubits)
        q_proj = self.quantum_project(q_out)

        # Residual connection
        output = self.residual_gate * q_proj + (1 - self.residual_gate) * x
        
        return output


class SparseAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, sparse_cfg: dict):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.window_size = sparse_cfg.get('window_size', 32)
        self.global_tokens = sparse_cfg.get('global_tokens', 8)
        self.quantum_tokens = sparse_cfg.get('quantum_tokens', 4)
        self.attn_qubits = sparse_cfg.get('attn_qubits', 2)
        
        # Ablation flag
        self.use_q_imp = sparse_cfg.get('use_q_imp', True)
        
        self.importance_net = nn.Linear(d_model, num_heads)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.quantum_attention_scale = nn.Parameter(torch.ones(1))
        self.use_siglip = False
        self.siglip_temperature = nn.Parameter(torch.ones(1))
        
        # === QUANTUM INITIALIZATION ===
        n_qubits = self.attn_qubits
        
        # We explicitly set q_attn_circuit to None if Q-imp is disabled so the training loop can skip it
        self.q_attn_circuit = None
        
        if self.use_q_imp:
            self.q_attn_device = tq.QuantumDevice(n_wires=n_qubits, bsz=1, record_op=False)
            
            class QuantumAttentionCircuit(tq.QuantumModule):
                def __init__(self, n_qubits):
                    super().__init__()
                    self.n_qubits = n_qubits
                    
                    self.ry_layers = nn.ModuleList([
                        tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)
                    ])
                    
                    self.entanglement = nn.ModuleList([
                        tq.CNOT(wires=[i, i+1]) for i in range(n_qubits - 1)
                    ])
                    
                    self.surrogate = nn.Sequential(
                        nn.Linear(2 ** n_qubits, n_qubits * 4),
                        nn.Tanh(),
                        nn.Linear(n_qubits * 4, n_qubits)
                    )
                
                def forward(self, q_device, qk_combined):
                    qk_normalized = F.normalize(qk_combined, p=2, dim=-1).to(dtype=torch.complex64)
                    q_device.set_states(qk_normalized)
                    
                    for i in range(self.n_qubits):
                        self.ry_layers[i](q_device, wires=i)
                    
                    for ent in self.entanglement:
                        ent(q_device)
                    
                    observables = [tq.PauliZ() for _ in range(self.n_qubits)]
                    quantum_output = tq.expval(q_device, list(range(self.n_qubits)), observables)
                    
                    if self.training:
                        quantum_detached = quantum_output.detach()
                        surrogate_output = self.surrogate(qk_combined.real)
                        return quantum_detached + (surrogate_output - surrogate_output.detach())
                    return quantum_output
            
            self.q_attn_circuit = QuantumAttentionCircuit(n_qubits)
            self.to_quantum_attn = nn.Linear(self.head_dim * 2, 2 ** n_qubits)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
    
    def _get_sparse_pattern(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True
            mask[i, :self.global_tokens] = True
            mask[i, -self.global_tokens:] = True
        return mask
    
    def _select_quantum_tokens(self, importance_scores: torch.Tensor) -> List[Tuple]:
        batch_size, seq_len, num_heads = importance_scores.shape
        quantum_indices = []
        for b in range(batch_size):
            aggregated = importance_scores[b].mean(dim=-1)
            topk_val, topk_idx = torch.topk(aggregated, min(self.quantum_tokens, seq_len))
            quantum_indices.append((b, topk_idx))
        return quantum_indices
    
    def _quantum_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        num_q, head_dim = Q.shape
        seq_len = K.shape[0]
        
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        
        q_expanded = Q_norm.unsqueeze(1).expand(-1, seq_len, -1)
        k_expanded = K_norm.unsqueeze(0).expand(num_q, -1, -1)
        qk_combined_batched = torch.cat([q_expanded, k_expanded], dim=-1)
        
        total_pairs = num_q * seq_len
        qk_combined_flat = qk_combined_batched.view(total_pairs, -1)
        
        qk_quantum = self.to_quantum_attn(qk_combined_flat)
        
        self.q_attn_device.bsz = total_pairs
        q_out = self.q_attn_circuit(self.q_attn_device, qk_quantum)
        
        attn_scores = q_out.mean(dim=-1)
        quantum_scores = attn_scores.view(num_q, seq_len)
        quantum_scores = torch.sigmoid(quantum_scores * self.quantum_attention_scale)
        
        return quantum_scores
    
    def _compute_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                          mask: torch.Tensor, quantum_enhancements: Dict) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        attn_scores = torch.full((batch_size, num_heads, seq_len, seq_len), -1e9, device=Q.device)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        for (b, token_indices), quantum_scores in quantum_enhancements.items():
            for h in range(self.num_heads):
                for idx_pos, token_idx in enumerate(token_indices):
                    attn_scores[b, h, token_idx] += quantum_scores[h, idx_pos]
        
        if self.use_siglip:
            attn_weights = torch.sigmoid(attn_scores / self.siglip_temperature)
        else:
            attn_weights = F.softmax(attn_scores / math.sqrt(head_dim), dim=-1)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(context)
    
    def enable_siglip(self):
        self.use_siglip = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        importance_scores = self.importance_net(x)
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        sparse_mask = self._get_sparse_pattern(seq_len, x.device)
        quantum_enhancements = {}
        
        # Apply selective quantum-enhanced attention based on the ablation flag
        if self.use_q_imp:
            quantum_indices = self._select_quantum_tokens(importance_scores)
            for b, token_indices in quantum_indices:
                if len(token_indices) > 0:
                    head_quantum_scores = []
                    for h in range(self.num_heads):
                        Q_quantum = Q[b, h, token_indices]
                        K_all = K[b, h]
                        quantum_scores = self._quantum_attention_scores(Q_quantum, K_all)
                        head_quantum_scores.append(quantum_scores)
                    quantum_enhancements[(b, token_indices)] = torch.stack(head_quantum_scores)
        
        return self._compute_attention(Q, K, V, sparse_mask, quantum_enhancements)

class MoEFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, num_experts: int):
        super().__init__()
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        
        self.shared_ann = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 4),
        )
        
        # Create a dynamic list of experts instead of hard-coding them
        self.experts = nn.ModuleList([
            self._create_expert("linguistic" if i % 2 == 0 else "semantic")
            for i in range(self.num_experts)
        ])
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.router_noise = nn.Parameter(torch.ones(1) * 1e-2)
        self.output_proj = nn.Linear(ffn_dim, d_model)
        
        self.aux_loss = None
    
    def _create_expert(self, expert_type: str) -> nn.Module:
        if expert_type == "linguistic":
            return nn.Sequential(
                nn.Linear(self.d_model + self.d_model // 4, self.ffn_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.ffn_dim, self.ffn_dim),
            )
        else:
            return nn.Sequential(
                nn.Linear(self.d_model + self.d_model // 4, self.ffn_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(self.ffn_dim, self.ffn_dim),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        shared_features = self.shared_ann(x)
        
        router_logits = self.router(x)
        
        if self.training:
            noise = torch.randn_like(router_logits) * self.router_noise
            router_logits = router_logits + noise
        
        expert_weights = F.softmax(router_logits, dim=-1)
        
        combined = torch.cat([x, shared_features], dim=-1)
        
        # Process all experts, stack their outputs, and apply weights
        expert_outputs = [
            expert(combined.view(-1, combined.size(-1))).view(batch_size, seq_len, -1)
            for expert in self.experts
        ]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=2)
        
        expert_output = torch.sum(expert_outputs_stacked * expert_weights.unsqueeze(-1), dim=2)
        
        if self.training:
            self._compute_aux_loss(expert_weights)
        
        return self.output_proj(expert_output)

class StandardFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    
    def _compute_aux_loss(self, expert_weights: torch.Tensor):
        batch_size, seq_len, num_experts = expert_weights.shape

        expert_utilization = expert_weights.mean(dim=(0, 1))
        target_uniform = torch.ones_like(expert_utilization, device=expert_utilization.device) / num_experts
        self.aux_loss = F.mse_loss(expert_utilization, target_uniform)

class HybridTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, cfg: dict, num_experts: int):
        super().__init__()
        self.d_model = d_model
        
        # Apply PAQM ablation flag
        self.use_paqm = cfg.get('use_paqm', True)
        
        if self.use_paqm:
            self.quantum_mixing = PositionAwareQuantumMixing(
                d_model, 
                cfg.get('n_qubits'),
                cfg.get('n_quantum_layers')
            )
        else:
            from QBET.classical_mixing_variants import MLP2xMixing
            self.quantum_mixing = MLP2xMixing(
                d_model,
                cfg.get('n_qubits'),
                cfg.get('n_quantum_layers')
            )
        
        self.attention = SparseAttention(d_model, num_heads, cfg)
        
        if num_experts == 1:
            self.ffn = StandardFFN(d_model, ffn_dim, cfg.get('dropout', 0.1))
        else:
            self.moe_ffn = MoEFFN(d_model, ffn_dim, num_experts)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        quantum_out = self.quantum_mixing(x, positions)
        x = x + quantum_out  # Residual connection
        x = self.norm1(x) 
        
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        
        ffn_out = self.ffn(x) if hasattr(self, 'ffn') else self.moe_ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        
        return x

class QBET(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, ffn_dim: int, cfg: dict, num_experts: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(d_model, num_heads, ffn_dim, cfg, num_experts)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.max_seq_len = cfg.get('max_seq_len', 512)
        self.training_phase = 1
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_training_phase(self, phase: int):
        self.training_phase = phase
        if phase >= 3:
            for block in self.blocks:
                block.attention.enable_siglip()
        print(f"Switched to training phase {phase}")
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x, positions)
        
        x = self.norm(x)
        logits = self.head(x[:, -1, :])
        
        return logits, None
    
    def get_aux_loss(self):
        total_aux_loss = 0.0
        for block in self.blocks:
            if hasattr(block, 'moe_ffn') and hasattr(block.moe_ffn, 'aux_loss') and block.moe_ffn.aux_loss is not None:
                total_aux_loss += block.moe_ffn.aux_loss
        return total_aux_loss
