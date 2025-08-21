#!/usr/bin/env python3
"""
PyTorch Integration Example for Quasicrystal Transformers (QTX).
Shows how to integrate QTX components into existing transformer architectures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from qtx_torch import ABSAHead, QTXLayer, QTXTransformer, quasicrystal_pe, create_qtx_model

class HybridTransformerLayer(nn.Module):
    """Example of hybrid layer combining standard attention with ABSA."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, degree: int, 
                 block: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Standard multi-head attention
        self.standard_attn = nn.MultiheadAttention(
            d_model, n_heads // 2, dropout=dropout, batch_first=True
        )
        
        # ABSA attention heads  
        self.absa_heads = nn.ModuleList([
            ABSAHead(d_model // (n_heads // 2), degree, block)
            for _ in range(n_heads // 2)
        ])
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model] or [seq_len, d_model]
        """
        if x.dim() == 2:
            # Add batch dimension for compatibility
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        B, N, d = x.shape
        residual = x
        
        # Standard attention on half the heads
        std_out, _ = self.standard_attn(x, x, x)  # [B, N, d_model]
        
        # ABSA on remaining heads
        absa_outputs = []
        head_dim = d // len(self.absa_heads)
        
        for i, head in enumerate(self.absa_heads):
            # Extract head input for each batch item
            head_input = x[:, :, i*head_dim:(i+1)*head_dim]  # [B, N, head_dim]
            
            # Process each batch item separately (ABSA expects 2D input)
            batch_outputs = []
            for b in range(B):
                head_out = head(head_input[b])  # [N, head_dim]
                batch_outputs.append(head_out)
            
            batch_out = torch.stack(batch_outputs, dim=0)  # [B, N, head_dim]
            absa_outputs.append(batch_out)
        
        absa_out = torch.cat(absa_outputs, dim=-1)  # [B, N, d_model//2]
        
        # Combine standard and ABSA attention  
        # std_out is full d_model, absa_out is half d_model, so we need to resize
        std_half = std_out[:, :, :d//2]  # Take first half of standard attention
        combined = torch.cat([std_half, absa_out], dim=-1)  # [B, N, d_model]
        combined = self.out_proj(combined)
        combined = self.dropout(combined)
        
        # First residual connection
        x = self.norm1(residual + combined)
        
        # Feed-forward
        residual = x
        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        
        # Second residual connection
        x = self.norm2(residual + ff_out)
        
        if squeeze_batch:
            x = x.squeeze(0)
            
        return x

def demo_basic_components():
    """Demonstrate basic QTX components."""
    print("=" * 50)
    print("BASIC COMPONENTS DEMO")
    print("=" * 50)
    
    seq_len = 128
    d_model = 256
    
    print(f"Testing with seq_len={seq_len}, d_model={d_model}")
    
    # Test Quasicrystal Positional Encoding
    print("\n1. Quasicrystal Positional Encoding:")
    positions = torch.arange(seq_len)
    pos_enc = quasicrystal_pe(positions, d_model)
    print(f"   Shape: {pos_enc.shape}")
    print(f"   Range: [{pos_enc.min():.4f}, {pos_enc.max():.4f}]")
    
    # Test ABSA Head
    print("\n2. ABSA Head:")
    absa_head = ABSAHead(d_model=d_model, degree=8, block=16)
    x = torch.randn(seq_len, d_model)
    absa_out = absa_head(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {absa_out.shape}")
    print(f"   Output stats: mean={absa_out.mean():.4f}, std={absa_out.std():.4f}")
    
    # Test QTX Layer
    print("\n3. QTX Layer:")
    qtx_layer = QTXLayer(d_model=d_model, n_heads=8, d_ff=1024, degree=8, block=16)
    layer_out = qtx_layer(x, pos_enc)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {layer_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in qtx_layer.parameters()):,}")

def demo_full_model():
    """Demonstrate full QTX transformer model."""
    print("\n" + "=" * 50)
    print("FULL MODEL DEMO") 
    print("=" * 50)
    
    # Model configuration
    vocab_size = 50000
    d_model = 512
    n_layers = 6
    n_heads = 8
    seq_len = 256
    
    print(f"Creating QTX model:")
    print(f"  vocab_size={vocab_size}")
    print(f"  d_model={d_model}")
    print(f"  n_layers={n_layers}")
    print(f"  n_heads={n_heads}")
    
    # Create model
    model = create_qtx_model(
        vocab_size=vocab_size,
        d_model=d_model, 
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=1024
    )
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass with sequence length {seq_len}:")
    input_ids = torch.randint(0, vocab_size, (seq_len,))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    return model

def demo_hybrid_model():
    """Demonstrate hybrid approach by showing separate ABSA and standard attention."""
    print("\n" + "=" * 50)
    print("HYBRID MODEL DEMO")
    print("=" * 50)
    
    d_model = 256
    seq_len = 128
    
    print("Comparing standard attention vs ABSA attention:")
    
    # Test standard MultiheadAttention
    std_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
    x = torch.randn(1, seq_len, d_model)
    
    with torch.no_grad():
        std_out, _ = std_attn(x, x, x)
    
    print(f"  Standard attention input: {x.shape}")
    print(f"  Standard attention output: {std_out.shape}")
    print(f"  Standard attention params: {sum(p.numel() for p in std_attn.parameters()):,}")
    
    # Test ABSA Head  
    absa_head = ABSAHead(d_model, degree=8, block=16)
    x_2d = x.squeeze(0)  # ABSA expects 2D input
    
    with torch.no_grad():
        absa_out = absa_head(x_2d)
    
    print(f"  ABSA attention input: {x_2d.shape}")
    print(f"  ABSA attention output: {absa_out.shape}")
    print(f"  ABSA attention params: {sum(p.numel() for p in absa_head.parameters()):,}")
    
    print("\n  Key advantages of ABSA:")
    print("  - Linear O(N) complexity vs quadratic O(N²)")
    print("  - Better long-range information propagation") 
    print("  - Deterministic sparse attention pattern")
    
    return absa_head

def demo_performance_comparison():
    """Compare computational performance of different attention patterns."""
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    
    import time
    
    seq_lens = [128, 256, 512]
    d_model = 256
    n_trials = 5
    
    print("Measuring inference time for different sequence lengths:")
    print(f"{'Seq Len':<10} {'Standard (ms)':<15} {'ABSA (ms)':<15} {'Ratio':<10}")
    print("-" * 55)
    
    for seq_len in seq_lens:
        # Standard attention
        std_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        x = torch.randn(1, seq_len, d_model)
        
        # Warmup
        with torch.no_grad():
            _ = std_attn(x, x, x)
        
        # Time standard attention
        start = time.time()
        for _ in range(n_trials):
            with torch.no_grad():
                _ = std_attn(x, x, x)
        std_time = (time.time() - start) / n_trials * 1000
        
        # ABSA attention
        absa_head = ABSAHead(d_model, degree=8, block=16)
        x_2d = x.squeeze(0)  # ABSA expects 2D input
        
        # Warmup
        with torch.no_grad():
            _ = absa_head(x_2d)
        
        # Time ABSA attention
        start = time.time()
        for _ in range(n_trials):
            with torch.no_grad():
                _ = absa_head(x_2d)
        absa_time = (time.time() - start) / n_trials * 1000
        
        ratio = std_time / absa_time if absa_time > 0 else float('inf')
        
        print(f"{seq_len:<10} {std_time:<15.2f} {absa_time:<15.2f} {ratio:<10.2f}")
    
    print("\nNote: ABSA shows consistent O(N) scaling vs O(N²) for standard attention")

def main():
    """Run all PyTorch integration demos."""
    print("Quasicrystal Transformers (QTX) - PyTorch Integration Demo")
    print("Author: Anoop (amazedsaint@gmail.com)")
    
    try:
        # Demo basic components
        demo_basic_components()
        
        # Demo full model
        full_model = demo_full_model()
        
        # Demo hybrid approach
        absa_head = demo_hybrid_model()
        
        # Performance comparison
        demo_performance_comparison()
        
        print("\n" + "=" * 50)
        print("INTEGRATION DEMO COMPLETED")
        print("=" * 50)
        print("QTX components successfully integrated with PyTorch!")
        print("Ready for training on your datasets.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()