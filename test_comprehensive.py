#!/usr/bin/env python3
"""
Comprehensive test suite for Quasicrystal Transformers (QTX).
Tests all components for correctness, edge cases, and performance.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
import time
from typing import List, Tuple
import traceback

# Import QTX components
from qtx_numpy import (
    irrational_block_permutation, inv_perm, wrap,
    build_adj_sliding, build_adj_dilated, build_adj_absa,
    bfs_layers, coverage_fraction_vs_layers, capacity_at_L,
    sinusoidal_pe, quasicrystal_pe, positional_stats,
    attention_forward_sparse
)

from qtx_torch import (
    ABSAHead, QTXLayer, QTXTransformer, create_qtx_model,
    build_adj_absa as torch_build_adj_absa,
    quasicrystal_pe as torch_quasicrystal_pe
)

def test_helper_functions():
    """Test basic helper functions."""
    print("Testing helper functions...")
    
    # Test irrational_block_permutation
    P = irrational_block_permutation(10)
    assert len(P) == 10, "Permutation length incorrect"
    assert len(set(P)) == 10, "Permutation should have unique elements"
    assert all(0 <= p < 10 for p in P), "Permutation values out of range"
    
    # Test inverse permutation
    P_inv = inv_perm(P)
    assert all(P_inv[P[i]] == i for i in range(len(P))), "Inverse permutation incorrect"
    
    # Test wrap function
    assert wrap(-1, 10) == 9, "Wrap function incorrect for negative"
    assert wrap(10, 10) == 0, "Wrap function incorrect for boundary"
    assert wrap(5, 10) == 5, "Wrap function incorrect for normal case"
    
    print("✓ Helper functions passed")

def test_adjacency_construction():
    """Test attention graph construction."""
    print("Testing adjacency construction...")
    
    N, block, degree = 64, 8, 6
    
    # Test sliding window
    adj_sliding = build_adj_sliding(N, degree)
    assert len(adj_sliding) == N, "Wrong adjacency list count"
    assert all(len(adj_sliding[i]) == degree for i in range(N)), "Wrong degree count for sliding"
    
    # Test dilated attention
    adj_dilated = build_adj_dilated(N, degree, block)
    assert len(adj_dilated) == N, "Wrong adjacency list count"
    
    # Test ABSA
    adj_absa = build_adj_absa(N, degree, block, leaps=(2, 3))
    assert len(adj_absa) == N, "Wrong adjacency list count"
    assert all(len(adj_absa[i]) == degree for i in range(N)), "Wrong degree count for ABSA"
    
    # Verify no self-loops in adjacency (self-attention handled separately)
    for i in range(N):
        assert i not in adj_absa[i], f"Self-loop found at position {i}"
    
    print("✓ Adjacency construction passed")

def test_bfs_coverage():
    """Test BFS coverage analysis."""
    print("Testing BFS coverage...")
    
    N, block, degree = 32, 4, 4
    adj = build_adj_absa(N, degree, block)
    
    # Test basic BFS
    reached = bfs_layers(adj, src=0, L=1)
    assert reached[0] == True, "Source should be reachable"
    assert reached.sum() > 1, "Should reach more than just source"
    
    # Test coverage increases with layers
    cov1 = capacity_at_L(adj, src=0, L=1)
    cov2 = capacity_at_L(adj, src=0, L=2)
    assert cov2 >= cov1, "Coverage should increase with layers"
    
    print("✓ BFS coverage passed")

def test_positional_encodings():
    """Test positional encoding functions."""
    print("Testing positional encodings...")
    
    positions = np.arange(100)
    d_model = 64
    
    # Test sinusoidal PE
    sin_pe = sinusoidal_pe(positions, d_model)
    assert sin_pe.shape == (100, 64), "Wrong sinusoidal PE shape"
    assert np.all(np.abs(sin_pe) <= 1.0), "Sinusoidal PE values out of range"
    
    # Test QPE
    qpe = quasicrystal_pe(positions, d_model)
    assert qpe.shape == (100, 64), "Wrong QPE shape" 
    assert np.all(np.abs(qpe) <= 1.0), "QPE values out of range"
    
    # Test that QPE is different from sinusoidal
    assert not np.allclose(sin_pe, qpe), "QPE should differ from sinusoidal"
    
    # Test positional stats
    stats = positional_stats(lambda pos, d: quasicrystal_pe(pos, d), 50, 100, 32)
    assert 'coh_train' in stats, "Missing coherence stats"
    assert 'cond_train' in stats, "Missing condition stats"
    assert stats['coh_train'] >= 0, "Coherence should be non-negative"
    
    print("✓ Positional encodings passed")

def test_sparse_attention():
    """Test sparse attention forward pass."""
    print("Testing sparse attention...")
    
    N, d, degree, block = 32, 16, 4, 8
    adj = build_adj_absa(N, degree, block)
    
    # Create random inputs
    Q = np.random.normal(0, 1, (N, d))
    K = np.random.normal(0, 1, (N, d)) 
    V = np.random.normal(0, 1, (N, d))
    
    # Test forward pass
    output = attention_forward_sparse(Q, K, V, adj)
    assert output.shape == (N, d), "Wrong output shape"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print("✓ Sparse attention passed")

def test_torch_components():
    """Test PyTorch components."""
    print("Testing PyTorch components...")
    
    # Test QPE
    positions = torch.arange(50)
    d_model = 32
    pe = torch_quasicrystal_pe(positions, d_model)
    assert pe.shape == (50, 32), "Wrong PyTorch QPE shape"
    assert torch.all(torch.abs(pe) <= 1.0), "PyTorch QPE values out of range"
    
    # Test ABSA Head
    absa = ABSAHead(d_model=32, degree=4, block=8)
    x = torch.randn(32, 32)
    out = absa(x)
    assert out.shape == (32, 32), "Wrong ABSA output shape"
    assert torch.all(torch.isfinite(out)), "ABSA output contains NaN/Inf"
    
    # Test QTX Layer
    layer = QTXLayer(d_model=32, n_heads=4, d_ff=64, degree=4, block=8)
    layer_out = layer(x)
    assert layer_out.shape == (32, 32), "Wrong QTX layer output shape"
    assert torch.all(torch.isfinite(layer_out)), "QTX layer output contains NaN/Inf"
    
    # Test full model
    model = create_qtx_model(vocab_size=100, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)
    input_ids = torch.randint(0, 100, (32,))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    assert logits.shape == (32, 100), "Wrong model output shape"
    assert torch.all(torch.isfinite(logits)), "Model output contains NaN/Inf"
    
    print("✓ PyTorch components passed")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")
    
    # Test minimum configurations
    try:
        adj = build_adj_absa(N=16, degree=2, block=4, leaps=(1,))
        assert len(adj) == 16, "Minimum config failed"
    except Exception as e:
        print(f"✗ Minimum config failed: {e}")
        raise
    
    # Test with different block sizes
    for block_size in [4, 8, 16]:
        N = block_size * 4  # Ensure N divisible by block
        adj = build_adj_absa(N, degree=4, block=block_size)
        assert len(adj) == N, f"Block size {block_size} failed"
    
    # Test PyTorch edge cases
    try:
        # Very small model with sequence length divisible by block size
        model = create_qtx_model(vocab_size=10, d_model=16, n_layers=1, n_heads=2, block=8)
        input_ids = torch.randint(0, 10, (16,))  # Use 16 tokens (divisible by 8)
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (16, 10), "Small model failed"
    except Exception as e:
        print(f"✗ Small model failed: {e}")
        raise
    
    print("✓ Edge cases passed")

def test_mathematical_properties():
    """Test mathematical properties and invariants."""
    print("Testing mathematical properties...")
    
    # Test permutation properties
    for n in [10, 20, 50]:
        P = irrational_block_permutation(n)
        P_inv = inv_perm(P)
        
        # Test bijection property
        assert sorted(P) == list(range(n)), "Permutation not bijective"
        assert all(P_inv[P[i]] == i for i in range(n)), "Inverse not correct"
    
    # Test positional encoding orthogonality properties
    positions = np.arange(64)
    pe = quasicrystal_pe(positions, 128)
    
    # Compute Gram matrix
    gram = pe @ pe.T
    
    # Check that diagonal is positive (normalized vectors)
    assert np.all(np.diag(gram) > 0), "Diagonal should be positive"
    
    # Test attention output properties
    N, d = 16, 8
    Q = np.random.normal(0, 1, (N, d))
    K = np.random.normal(0, 1, (N, d))
    V = np.random.normal(0, 1, (N, d))
    adj = build_adj_absa(N, degree=4, block=4)
    
    output = attention_forward_sparse(Q, K, V, adj)
    
    # Output should preserve dimension
    assert output.shape == V.shape, "Attention should preserve shape"
    
    print("✓ Mathematical properties passed")

def test_performance_scaling():
    """Test performance scaling properties."""
    print("Testing performance scaling...")
    
    sequence_lengths = [64, 128, 256]
    times = []
    
    for N in sequence_lengths:
        block = 16 if N >= 16 else N // 4
        degree = 8
        
        # Time adjacency construction
        start = time.time()
        adj = build_adj_absa(N, degree, block)
        adj_time = time.time() - start
        
        # Time attention computation
        d = 32
        Q = np.random.normal(0, 1, (N, d))
        K = np.random.normal(0, 1, (N, d))
        V = np.random.normal(0, 1, (N, d))
        
        start = time.time()
        _ = attention_forward_sparse(Q, K, V, adj)
        attn_time = time.time() - start
        
        times.append((N, adj_time, attn_time))
        print(f"  N={N}: adj={adj_time*1000:.2f}ms, attn={attn_time*1000:.2f}ms")
    
    # Check that scaling is reasonable (should be roughly linear)
    if len(times) >= 2:
        ratio1 = times[1][2] / times[0][2]  # attention time ratio
        ratio2 = times[1][0] / times[0][0]  # sequence length ratio
        scaling_factor = ratio1 / ratio2
        assert scaling_factor < 3, f"Scaling too poor: {scaling_factor}"
    
    print("✓ Performance scaling passed")

def run_comprehensive_tests():
    """Run all test suites."""
    print("=" * 60)
    print("COMPREHENSIVE QTX TEST SUITE")
    print("=" * 60)
    
    test_functions = [
        test_helper_functions,
        test_adjacency_construction,
        test_bfs_coverage,
        test_positional_encodings,
        test_sparse_attention,
        test_torch_components,
        test_edge_cases,
        test_mathematical_properties,
        test_performance_scaling
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! QTX implementation is working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)