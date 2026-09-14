"""Stacked-SVD falsification gate (Gram route; low-RAM safe).

Usage: python3 stacked_svd.py <f32file> <rows> <cols>
Prints relative Frobenius error at ranks 64..1024 for the stacked matrix.
Bar (phase11b): rank-512 rel-err > 0.15 => shared-basis KILL.
"""
import sys
import numpy as np

path, rows, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
W = np.fromfile(path, dtype=np.float32).reshape(rows, cols)
assert np.isfinite(W).all(), "non-finite weights: check slice alignment"
G = W.T @ W
del W
S = np.sqrt(np.maximum(np.linalg.eigvalsh(G)[::-1], 0))
tot = (S ** 2).sum()
print("top-10 SV:", np.round(S[:10], 2))
for r in (64, 128, 256, 384, 512, 768, 1024):
    kept = (S[:r] ** 2).sum() / tot
    print(f"{r:4d} : rel-err {np.sqrt(1 - kept):.4f} : energy {kept:.4f}")
