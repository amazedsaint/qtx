# qtx_numpy.py
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict

# ---------- Helpers ----------
def irrational_block_permutation(n_blocks: int, alpha: float = math.sqrt(2) - 1.0) -> List[int]:
    """Aperiodic permutation of blocks via irrational rotation ordering."""
    keys = [((i * alpha) % 1.0) for i in range(n_blocks)]
    return [i for _, i in sorted(zip(keys, range(n_blocks)), key=lambda t: t[0])]

def inv_perm(perm: List[int]) -> List[int]:
    ip = [0]*len(perm)
    for i,p in enumerate(perm):
        ip[p] = i
    return ip

def wrap(i: int, n: int) -> int:
    return (i % n + n) % n

# ---------- Attention graphs ----------
def build_adj_sliding(N: int, degree: int) -> List[List[int]]:
    """Sliding-window ring neighbors: ±1..±(degree//2)."""
    R = degree // 2
    adj = [[] for _ in range(N)]
    for i in range(N):
        for d in range(1, R+1):
            adj[i].append(wrap(i+d, N)); adj[i].append(wrap(i-d, N))
    return adj

def build_adj_dilated(N: int, degree: int, block: int) -> List[List[int]]:
    """Local ±1..±r, plus dilated ±k*block jumps using remaining budget."""
    r = degree // 4
    dil = degree - 2*r
    n_pairs = max(dil // 2, 1)
    adj = [[] for _ in range(N)]
    for i in range(N):
        for d in range(1, r+1):
            adj[i].append(wrap(i+d, N)); adj[i].append(wrap(i-d, N))
        for k in range(1, n_pairs+1):
            jump = k * block
            adj[i].append(wrap(i+jump, N)); adj[i].append(wrap(i-jump, N))
    return adj

def build_adj_absa(N: int, degree: int, block: int, leaps: Tuple[int,int]=(2,5)) -> List[List[int]]:
    """Aperiodic Block-Sparse Attention (ABSA): local ±1..±r in-block, plus aperiodic cross-block edges."""
    assert N % block == 0
    B = N // block
    P = irrational_block_permutation(B, alpha=math.sqrt(2)-1.0)
    P_inv = inv_perm(P)
    r = max(degree // 4, 1)
    cross_budget = degree - 2*r
    n_pairs = min(2, max(cross_budget // 2, 0))
    use_leaps = list(leaps)[:n_pairs]
    adj = [[] for _ in range(N)]
    for pos in range(N):
        b = pos // block; o = pos % block
        # local neighbors
        for d in range(1, r+1):
            adj[pos].append(b*block + wrap(o+d, block))
            adj[pos].append(b*block + wrap(o-d, block))
        # cross-block neighbors at same offset
        idx = P_inv[b]
        for L in use_leaps:
            nb1 = P[wrap(idx + L, B)]
            nb2 = P[wrap(idx - L, B)]
            adj[pos].append(nb1*block + o)
            adj[pos].append(nb2*block + o)
    return adj

# ---------- BFS coverage & capacity ----------
def bfs_layers(adj: List[List[int]], src: int, L: int) -> np.ndarray:
    """Boolean vector: nodes reachable from src within <=L steps (undirected)."""
    N = len(adj)
    undirected = [[] for _ in range(N)]
    for i in range(N):
        for j in adj[i]:
            undirected[i].append(j)
            undirected[j].append(i)
    reached = np.zeros(N, dtype=bool)
    frontier = np.zeros(N, dtype=bool)
    reached[src] = True; frontier[src] = True
    for _ in range(L):
        new_frontier = np.zeros(N, dtype=bool)
        for i in np.where(frontier)[0]:
            for j in undirected[i]:
                if not reached[j]:
                    reached[j] = True
                    new_frontier[j] = True
        frontier = new_frontier
        if not frontier.any(): break
    return reached

def coverage_fraction_vs_layers(adj: List[List[int]], src: int, Lmax: int) -> pd.DataFrame:
    rows=[]
    for L in range(1, Lmax+1):
        reached = bfs_layers(adj, src, L)
        cov = (reached.sum() - 1) / (len(adj) - 1)
        rows.append(dict(L=L, coverage=float(cov)))
    return pd.DataFrame(rows)

def capacity_at_L(adj: List[List[int]], src: int, L: int) -> float:
    """Needle detection capacity proxy = coverage fraction at hop budget L."""
    reached = bfs_layers(adj, src, L)
    return float((reached.sum()-1)/(len(adj)-1))

# ---------- Positional encodings ----------
def sinusoidal_pe(positions: np.ndarray, d_model: int, base: float=10000.0) -> np.ndarray:
    assert d_model % 2 == 0
    half = d_model // 2
    i = np.arange(half)
    freqs = (base ** (-2 * i / d_model)).reshape(1, -1)
    ang = positions.reshape(-1,1) * freqs
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)

def quasicrystal_pe(positions: np.ndarray, d_model: int, alphas=None) -> np.ndarray:
    """Aperiodic Fourier features using incommensurate bases (√2, √3, φ)."""
    if alphas is None:
        alphas = [math.sqrt(2), math.sqrt(3), (1+math.sqrt(5))/2]
    assert d_model % 2 == 0
    half = d_model // 2
    freqs = []
    k = 1
    while len(freqs) < half:
        for a in alphas:
            freqs.append(k * a / (2*math.pi))
            if len(freqs) >= half: break
        k += 1
    freqs = np.array(freqs[:half]).reshape(1,-1)
    ang = positions.reshape(-1,1) * freqs
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)

def positional_stats(pe_fn, L_train: int, L_test: int, d_model: int) -> Dict[str, float]:
    pos_train = np.arange(L_train); pos_test = np.arange(L_test)
    Phi_tr = pe_fn(pos_train, d_model); Phi_te = pe_fn(pos_test, d_model)
    def row_norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    A = row_norm(Phi_tr); B = row_norm(Phi_te)
    C = A @ A.T; np.fill_diagonal(C, 0.0)
    D = B @ B.T; np.fill_diagonal(D, 0.0)
    coh_tr = float(np.max(np.abs(C))); coh_te = float(np.max(np.abs(D)))
    cond_tr = float(np.linalg.cond(Phi_tr.T @ Phi_tr))
    cond_te = float(np.linalg.cond(Phi_te.T @ Phi_te))
    return dict(coh_train=coh_tr, coh_test=coh_te, cond_train=cond_tr, cond_test=cond_te)

# ---------- Sparse attention forward ----------
def attention_forward_sparse(Q: np.ndarray, K: np.ndarray, V: np.ndarray, adj: List[List[int]]) -> np.ndarray:
    """One-head sparse scaled dot-product attention restricted to adjacency lists."""
    N, d = Q.shape
    out = np.zeros_like(Q)
    scale = 1.0 / math.sqrt(d)
    for i in range(N):
        nbrs = adj[i] + [i]
        Ki = K[nbrs]; Vi = V[nbrs]
        scores = (Q[i:i+1] @ Ki.T) * scale
        s = scores - np.max(scores, axis=1, keepdims=True)
        w = np.exp(s) / (np.sum(np.exp(s), axis=1, keepdims=True) + 1e-12)
        out[i] = w @ Vi
    return out

# ---------- Demo & sanity ----------
if __name__ == "__main__":
    print("Quasicrystal Transformers (QTX) - NumPy Implementation")
    print("=" * 55)
    
    N=1024; block=16; degree=8; Lmax=8
    print(f"Configuration: N={N}, block={block}, degree={degree}, Lmax={Lmax}")
    print()
    
    # Build graphs
    print("Building attention graphs...")
    adj_sliding = build_adj_sliding(N, degree)
    adj_dilated = build_adj_dilated(N, degree, block)
    adj_absa    = build_adj_absa(N, degree, block, leaps=(2,5))
    print("Done.")
    print()

    # Coverage vs layers
    print("Computing coverage analysis...")
    tables=[]
    for name,adj in [("sliding",adj_sliding),("dilated",adj_dilated),("absa",adj_absa)]:
        df = coverage_fraction_vs_layers(adj, src=0, Lmax=Lmax); df["pattern"]=name; tables.append(df)
    cov = pd.concat(tables, ignore_index=True)
    
    print("Coverage at selected layers:")
    coverage_table = cov[cov["L"].isin([2,3,4,6,8])].pivot(index="L", columns="pattern", values="coverage")
    print(coverage_table.round(4))
    print()

    # Positional coherence & conditioning
    print("Computing positional encoding statistics...")
    pe = pd.DataFrame([
        {"encoding":"sinusoidal", **positional_stats(lambda pos,d: sinusoidal_pe(pos,d), 512, 4096, 64)},
        {"encoding":"qpe",        **positional_stats(lambda pos,d: quasicrystal_pe(pos,d), 512, 4096, 64)}
    ])
    print("Positional encoding stats:")
    print(pe.round(6))
    print()

    # Sparse attention forward sanity check
    print("Testing sparse attention forward pass...")
    d=32
    Q=np.random.normal(size=(128,d)); K=np.random.normal(size=(128,d)); V=np.random.normal(size=(128,d))
    Y = attention_forward_sparse(Q,K,V, build_adj_absa(128, degree=degree, block=16, leaps=(2,5)))
    print(f"Sparse attention output shape: {Y.shape}")
    print(f"Output statistics: mean={Y.mean():.4f}, std={Y.std():.4f}")
    print()
    
    # Additional validations
    print("Running validation checks...")
    
    # Check adjacency list sizes
    adj_sizes = [len(adj[i]) for i in range(min(100, N)) for adj in [adj_sliding, adj_dilated, adj_absa]]
    expected_degree = degree
    print(f"Adjacency list sizes - min: {min(adj_sizes)}, max: {max(adj_sizes)}, expected: {expected_degree}")
    
    # Check permutation properties
    P = irrational_block_permutation(64)
    P_inv = inv_perm(P)
    assert len(set(P)) == len(P), "Permutation should have unique elements"
    assert all(P_inv[P[i]] == i for i in range(len(P))), "Inverse permutation should be correct"
    print("Permutation validation: PASSED")
    
    print("All validation checks completed successfully!")