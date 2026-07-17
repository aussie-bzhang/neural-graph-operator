"""
Graph Neural Operator (GNO) — unified model definition.

Supports:
  - Multiple encoder types (GCN / GAT)
  - Multiple kernel types (Bilinear / MLP / Attention / Low-Rank)
  - Sinkhorn normalization on the kernel
  - Top-k sparsity on the kernel
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# ============================================================
#  Graph Encoders
# ============================================================

class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder (matches paper Eq. 4)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            # Residual connection when dimensions match
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder (ablation variant)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.2, heads: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(GATConv(dims[i], dims[i + 1], heads=heads,
                                      concat=False, dropout=dropout))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        return x


def build_encoder(encoder_type: str, input_dim: int, hidden_dim: int,
                  num_layers: int = 2, dropout: float = 0.2, **kwargs):
    if encoder_type == "gcn":
        return GCNEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif encoder_type == "gat":
        return GATEncoder(input_dim, hidden_dim, num_layers, dropout,
                          heads=kwargs.get("num_heads", 4))
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================
#  Cross-Modal Kernel Operators  (paper §3.2.2)
# ============================================================

class BilinearKernel(nn.Module):
    """K(i,j) = σ(u'(i)^T W v'(j))   — default in the paper."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, src_feat, tgt_feat):
        """
        src_feat : (n_src, d)
        tgt_feat : (n_tgt, d)
        returns  : (n_src, n_tgt) alignment scores in [0, 1]
        """
        # (n_src, d) @ (d, d) @ (d, n_tgt) -> (n_src, n_tgt)
        scores = src_feat @ self.W @ tgt_feat.T
        return torch.sigmoid(scores)


class MLPKernel(nn.Module):
    """K_MLP(i,j) = MLP([u'(i); v'(j)])   — paper Eq. 6."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_feat, tgt_feat):
        n_src, n_tgt = src_feat.size(0), tgt_feat.size(0)
        src_exp = src_feat.unsqueeze(1).expand(-1, n_tgt, -1)
        tgt_exp = tgt_feat.unsqueeze(0).expand(n_src, -1, -1)
        combined = torch.cat([src_exp, tgt_exp], dim=-1)
        scores = self.net(combined).squeeze(-1)
        return torch.sigmoid(scores)


class AttentionKernel(nn.Module):
    """K_Att(i,j) = softmax(q_i^T k_j / sqrt(d))   — paper Eq. 7."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dk = hidden_dim // num_heads
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src_feat, tgt_feat):
        Q = self.Wq(src_feat)  # (n_src, d)
        K = self.Wk(tgt_feat)  # (n_tgt, d)
        scores = Q @ K.T / math.sqrt(Q.size(-1))
        return torch.sigmoid(scores)


class LowRankKernel(nn.Module):
    """Approximate kernel: K ≈ U V^T  (for scalability)."""

    def __init__(self, hidden_dim: int, rank: int = 32):
        super().__init__()
        self.proj_src = nn.Linear(hidden_dim, rank)
        self.proj_tgt = nn.Linear(hidden_dim, rank)

    def forward(self, src_feat, tgt_feat):
        u = self.proj_src(src_feat)   # (n_src, rank)
        v = self.proj_tgt(tgt_feat)   # (n_tgt, rank)
        scores = u @ v.T
        return torch.sigmoid(scores)


def build_kernel(kernel_type: str, hidden_dim: int, **kwargs):
    if kernel_type == "bilinear":
        return BilinearKernel(hidden_dim)
    elif kernel_type == "mlp":
        return MLPKernel(hidden_dim, kwargs.get("dropout", 0.2))
    elif kernel_type == "attention":
        return AttentionKernel(hidden_dim, kwargs.get("num_heads", 4))
    elif kernel_type == "lowrank":
        return LowRankKernel(hidden_dim, kwargs.get("rank", 32))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


# ============================================================
#  Kernel Post-Processing  
# ============================================================

def sinkhorn_normalize(K: torch.Tensor, n_iters: int = 5,
                       eps: float = 1e-8) -> torch.Tensor:
    """Sinkhorn-style doubly-stochastic normalization on kernel K."""
    for _ in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + eps)
        K = K / (K.sum(dim=0, keepdim=True) + eps)
    return K


def topk_sparsify(K: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Keep only top-k entries per target node (column-wise)."""
    n_src, n_tgt = K.shape
    k = min(k, n_src)
    topk_vals, topk_idx = K.topk(k, dim=0)
    sparse_K = torch.zeros_like(K)
    sparse_K.scatter_(0, topk_idx, topk_vals)
    return sparse_K


# ============================================================
#  Graph Function Decoder  (paper §3.2.3, Eq. 8)
# ============================================================

class GraphDecoder(nn.Module):
    """v_hat(j) = Ψ(Σ_i K(i,j) · u'(i), G_text, j)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, K, src_feat):
        """
        K        : (n_src, n_tgt) kernel matrix
        src_feat : (n_src, d)     encoded source features
        returns  : (n_tgt, d)     decoded target features
        """
        # Weighted sum: (n_tgt, d) = K^T @ src_feat
        aggregated = K.T @ src_feat      # (n_tgt, d)
        return self.transform(aggregated)


# ============================================================
#  Full GNO Model
# ============================================================

class GraphNeuralOperator(nn.Module):
    """
    Complete Graph Neural Operator for cross-modal alignment.

    Components (paper §3.2):
      1. Modality-specific graph encoders  Φ_img, Φ_text
      2. Cross-modal kernel operator K(i,j)
      3. Graph function decoder Ψ
    """

    def __init__(self, cfg: dict):
        super().__init__()
        input_dim  = cfg.get("input_dim", 256)
        hidden_dim = cfg.get("hidden_dim", 128)
        enc_type   = cfg.get("encoder_type", "gcn")
        num_layers = cfg.get("num_encoder_layers", 2)
        dropout    = cfg.get("dropout", 0.2)
        kernel_type = cfg.get("kernel_type", "bilinear")
        num_heads  = cfg.get("num_heads", 4)

        # Separate encoders for each modality
        self.scene_encoder = build_encoder(
            enc_type, input_dim, hidden_dim, num_layers, dropout,
            num_heads=num_heads)
        self.text_encoder = build_encoder(
            enc_type, input_dim, hidden_dim, num_layers, dropout,
            num_heads=num_heads)

        # Cross-modal kernel
        self.kernel = build_kernel(
            kernel_type, hidden_dim, dropout=dropout, num_heads=num_heads)

        # Decoder for reconstruction loss
        self.decoder = GraphDecoder(hidden_dim)

        # Config for post-processing
        self.use_sinkhorn = cfg.get("use_sinkhorn", False)
        self.sinkhorn_iters = cfg.get("sinkhorn_iters", 5)
        self.use_topk = cfg.get("use_topk", False)
        self.topk_k = cfg.get("topk_k", 3)

        self.hidden_dim = hidden_dim

    def encode(self, scene_x, scene_edge, text_x, text_edge):
        """Encode both graphs."""
        scene_feat = self.scene_encoder(scene_x, scene_edge)
        text_feat  = self.text_encoder(text_x, text_edge)
        return scene_feat, text_feat

    def compute_kernel(self, scene_feat, text_feat):
        """Compute and optionally post-process the kernel matrix."""
        K = self.kernel(scene_feat, text_feat)
        if self.use_sinkhorn:
            K = sinkhorn_normalize(K, self.sinkhorn_iters)
        if self.use_topk:
            K = topk_sparsify(K, self.topk_k)
        return K

    def forward_single(self, scene_x, scene_edge, text_x, text_edge):
        """Process one graph pair; returns kernel K and decoded features."""
        scene_feat, text_feat = self.encode(
            scene_x, scene_edge, text_x, text_edge)
        K = self.compute_kernel(scene_feat, text_feat)
        v_hat = self.decoder(K, scene_feat)
        return K, v_hat, scene_feat, text_feat

    def forward(self, scene_batch, text_batch):
        """
        Process a batch of graph pairs using PyG Batch objects.

        Returns:
            kernels      : list of (n_src_i, n_tgt_i) alignment matrices
            v_hats       : list of (n_tgt_i, d) decoded features
            text_feats   : list of (n_tgt_i, d) encoded text features
        """
        # Encode full batches through GCN layers
        scene_feat = self.scene_encoder(scene_batch.x, scene_batch.edge_index)
        text_feat  = self.text_encoder(text_batch.x, text_batch.edge_index)

        batch_size = scene_batch.batch.max().item() + 1
        kernels, v_hats, tgt_feats = [], [], []

        for b in range(batch_size):
            s_mask = scene_batch.batch == b
            t_mask = text_batch.batch == b
            s_feat = scene_feat[s_mask]
            t_feat = text_feat[t_mask]

            K = self.compute_kernel(s_feat, t_feat)
            v_hat = self.decoder(K, s_feat)

            kernels.append(K)
            v_hats.append(v_hat)
            tgt_feats.append(t_feat)

        return kernels, v_hats, tgt_feats


# ============================================================
#  Baseline Models 
# ============================================================

class CosineBaseline(nn.Module):
    """Align by cosine similarity on raw / encoded features."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj_src = nn.Linear(input_dim, hidden_dim)
        self.proj_tgt = nn.Linear(input_dim, hidden_dim)

    def forward_single(self, src_x, tgt_x):
        s = F.normalize(self.proj_src(src_x), dim=-1)
        t = F.normalize(self.proj_tgt(tgt_x), dim=-1)
        return torch.sigmoid(s @ t.T)


class GCNCosineBaseline(nn.Module):
    """GCN encoder + cosine similarity (no learned kernel)."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.src_enc = GCNEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.tgt_enc = GCNEncoder(input_dim, hidden_dim, num_layers, dropout)

    def forward_single(self, src_x, src_edge, tgt_x, tgt_edge):
        s = F.normalize(self.src_enc(src_x, src_edge), dim=-1)
        t = F.normalize(self.tgt_enc(tgt_x, tgt_edge), dim=-1)
        return torch.sigmoid(s @ t.T)
