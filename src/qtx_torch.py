# qtx_torch.py
import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional

def irrational_block_permutation(n_blocks: int, alpha: float = math.sqrt(2)-1.0) -> List[int]:
    """Aperiodic permutation of blocks via irrational rotation ordering."""
    keys = [((i * alpha) % 1.0) for i in range(n_blocks)]
    return [i for _, i in sorted(zip(keys, range(n_blocks)), key=lambda t: t[0])]

def build_adj_absa(N: int, degree: int, block: int, leaps: Tuple[int,int]=(2,5)) -> List[List[int]]:
    """Build adjacency lists for Aperiodic Block-Sparse Attention (ABSA)."""
    assert N % block == 0
    B = N // block
    P = irrational_block_permutation(B)
    P_inv = [0]*B
    for i,p in enumerate(P):
        P_inv[p] = i
    r = max(degree // 4, 1)
    cross_budget = degree - 2*r
    n_pairs = min(2, max(cross_budget // 2, 0))
    use_leaps = list(leaps)[:n_pairs]
    def wrap(i,n): return (i % n + n) % n
    adj = [[] for _ in range(N)]
    for pos in range(N):
        b = pos // block; o = pos % block
        # Local neighbors within block
        for d in range(1, r+1):
            adj[pos].append(b*block + wrap(o+d, block))
            adj[pos].append(b*block + wrap(o-d, block))
        # Cross-block neighbors at same offset
        idx = P_inv[b]
        for L in use_leaps:
            nb1 = P[wrap(idx + L, B)]
            nb2 = P[wrap(idx - L, B)]
            adj[pos].append(nb1*block + o)
            adj[pos].append(nb2*block + o)
    return adj

def quasicrystal_pe(positions: torch.Tensor, d_model: int, alphas: Optional[List[float]]=None) -> torch.Tensor:
    """Quasicrystal Positional Encoding using incommensurate frequencies."""
    if alphas is None:
        alphas = [math.sqrt(2), math.sqrt(3), (1+math.sqrt(5))/2]
    assert d_model % 2 == 0
    half = d_model // 2
    freqs = []
    k=1
    while len(freqs) < half:
        for a in alphas:
            freqs.append(k * a / (2*math.pi))
            if len(freqs) >= half: break
        k+=1
    freqs = torch.tensor(freqs[:half], dtype=positions.dtype, device=positions.device).unsqueeze(0)  # [1, half]
    ang = positions.unsqueeze(1) * freqs  # [L, half]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [L, d_model]

class ABSAHead(nn.Module):
    """Single Aperiodic Block-Sparse Attention head."""
    
    def __init__(self, d_model: int, degree: int, block: int, leaps: Tuple[int,int]=(2,5)):
        super().__init__()
        self.d_model = d_model
        self.degree = degree
        self.block = block
        self.leaps = leaps
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
        # Cache adjacency lists for different sequence lengths
        self._adj_cache = {}

    def _get_adjacency(self, N: int) -> List[List[int]]:
        """Get or compute adjacency lists for sequence length N."""
        if N not in self._adj_cache:
            self._adj_cache[N] = build_adj_absa(N, self.degree, self.block, self.leaps)
        return self._adj_cache[N]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ABSA head.
        
        Args:
            X: Input tensor of shape [N, d_model]
        
        Returns:
            Output tensor of shape [N, d_model]
        """
        N, d = X.shape
        assert d == self.d_model, f"Input dimension {d} doesn't match d_model {self.d_model}"
        assert N % self.block == 0, f"Sequence length {N} must be divisible by block size {self.block}"
        
        Q = self.Wq(X)  # [N, d_model]
        K = self.Wk(X)  # [N, d_model]
        V = self.Wv(X)  # [N, d_model]
        
        adj = self._get_adjacency(N)
        out = torch.zeros_like(Q)
        scale = 1.0 / math.sqrt(d)
        
        for i in range(N):
            nbrs = adj[i] + [i]  # Include self-attention
            Ki = K[nbrs]        # [m, d_model]
            Vi = V[nbrs]        # [m, d_model]
            scores = (Q[i:i+1] @ Ki.T) * scale  # [1, m]
            # Numerical stability
            s = scores - scores.max(dim=1, keepdim=True).values
            w = torch.softmax(s, dim=1)  # [1, m]
            out[i] = w @ Vi
        return out

class QTXLayer(nn.Module):
    """Complete QTX Transformer layer with ABSA and standard components."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, degree: int, 
                 block: int, leaps: Tuple[int,int]=(2,5), dropout: float=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head ABSA attention
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.absa_heads = nn.ModuleList([
            ABSAHead(self.head_dim, degree, block, leaps) for _ in range(n_heads)
        ])
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for QTX layer.
        
        Args:
            x: Input tensor of shape [N, d_model]
            pos_enc: Optional positional encoding of shape [N, d_model]
        
        Returns:
            Output tensor of shape [N, d_model]
        """
        # Add positional encoding if provided
        if pos_enc is not None:
            x = x + pos_enc
        
        # Multi-head ABSA attention
        residual = x
        
        # Reshape for multi-head processing
        N, d_model = x.shape
        x_heads = x.view(N, self.n_heads, self.head_dim)
        
        # Apply ABSA heads
        head_outputs = []
        for i, head in enumerate(self.absa_heads):
            head_input = x_heads[:, i, :]  # [N, head_dim]
            head_out = head(head_input)    # [N, head_dim]
            head_outputs.append(head_out)
        
        # Concatenate head outputs
        attn_out = torch.cat(head_outputs, dim=-1)  # [N, d_model]
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        
        # First residual connection and normalization
        x = self.norm1(residual + attn_out)
        
        # Feed-forward network
        residual = x
        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        
        # Second residual connection and normalization
        x = self.norm2(residual + ff_out)
        
        return x

class QTXTransformer(nn.Module):
    """Complete Quasicrystal Transformer model."""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
                 d_ff: int, max_seq_len: int, degree: int, block: int, 
                 leaps: Tuple[int,int]=(2,5), dropout: float=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # QTX layers
        self.layers = nn.ModuleList([
            QTXLayer(d_model, n_heads, d_ff, degree, block, leaps, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for QTX Transformer.
        
        Args:
            input_ids: Token IDs of shape [N]
        
        Returns:
            Logits of shape [N, vocab_size]
        """
        N = input_ids.shape[0]
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [N, d_model]
        
        # Quasicrystal positional encoding
        positions = torch.arange(N, device=input_ids.device)
        pos_enc = quasicrystal_pe(positions, self.d_model)  # [N, d_model]
        
        # Apply QTX layers
        for layer in self.layers:
            x = layer(x, pos_enc)
        
        # Final normalization and output projection
        x = self.norm(x)
        logits = self.out_proj(x)  # [N, vocab_size]
        
        return logits

# Utility functions for easy integration
def create_qtx_model(vocab_size: int = 50257, d_model: int = 768, n_layers: int = 12,
                     n_heads: int = 12, d_ff: int = 3072, max_seq_len: int = 1024,
                     degree: int = 8, block: int = 16, leaps: Tuple[int,int] = (2,5)) -> QTXTransformer:
    """Create a QTX model with standard GPT-like configuration."""
    return QTXTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        degree=degree,
        block=block,
        leaps=leaps
    )

if __name__ == "__main__":
    print("Quasicrystal Transformers (QTX) - PyTorch Implementation")
    print("=" * 57)
    
    # Test basic components
    print("Testing basic components...")
    
    # Test positional encoding
    positions = torch.arange(100)
    pos_enc = quasicrystal_pe(positions, d_model=64)
    print(f"Positional encoding shape: {pos_enc.shape}")
    
    # Test ABSA head
    absa_head = ABSAHead(d_model=64, degree=8, block=16)
    x = torch.randn(128, 64)
    out = absa_head(x)
    print(f"ABSA head output shape: {out.shape}")
    
    # Test QTX layer
    qtx_layer = QTXLayer(d_model=64, n_heads=8, d_ff=256, degree=8, block=16)
    layer_out = qtx_layer(x)
    print(f"QTX layer output shape: {layer_out.shape}")
    
    # Test full model
    model = create_qtx_model(vocab_size=1000, d_model=64, n_layers=2, n_heads=8)
    input_ids = torch.randint(0, 1000, (128,))
    logits = model(input_ids)
    print(f"Full model output shape: {logits.shape}")
    
    print("All PyTorch tests passed successfully!")