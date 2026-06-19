import math
import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import gamma
from functools import lru_cache
from numba import njit

# 1. Coordinate PDF and Lloyd-Max Quantizer

def _coord_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of coordinate distribution for a Haar-rotated unit vector (Lemma 1)."""
    C = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return np.where(np.abs(x) < 1.0, C * (1.0 - x**2) ** ((d - 3) / 2), 0.0)


def lloyd_max(d: int, n_levels: int, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Lloyd-Max optimal scalar centroids for d-dimensional Haar-rotated unit coordinates."""
    # Integrate CDF
    def cdf(x):
        val, _ = integrate.quad(lambda t: _coord_pdf(t, d), -1.0, x, limit=100)
        return val

    # Quantile initialization
    quantile_probs = np.linspace(1 / (2 * n_levels), 1 - 1 / (2 * n_levels), n_levels)
    centroids = np.array([
        brentq(lambda x: cdf(x) - p, -1 + 1e-10, 1 - 1e-10) for p in quantile_probs
    ])

    for _ in range(max_iter):
        boundaries = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = integrate.quad(lambda x: x * _coord_pdf(x, d), lo, hi, limit=50)
            den, _ = integrate.quad(lambda x: _coord_pdf(x, d), lo, hi, limit=50)
            new_centroids[i] = num / den if den > 1e-15 else centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids


@lru_cache(maxsize=32)
def get_codebook(d: int, bits: int) -> np.ndarray:
    """Get or compute Lloyd-Max centroids for dimension d and bit-width."""
    return lloyd_max(d, n_levels=2**bits)

# 2. JIT-Accelerated Bit Packing and Coordinate Quantization

@njit(fastmath=True)
def quantize_coords_jit(y: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Argmin index lookup over coordinates."""
    n = y.shape[0]
    indices = np.empty(n, dtype=np.int64)
    for i in range(n):
        val = y[i]
        min_dist = 1e9
        best_idx = 0
        for j in range(len(codebook)):
            dist = (val - codebook[j]) ** 2
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        indices[i] = best_idx
    return indices


def pack(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack int64 indices into a flat uint8 byte array."""
    bit_cols = [(indices >> (bits - 1 - b)) & 1 for b in range(bits)]
    bits_arr = np.stack(bit_cols, axis=1).reshape(-1).astype(np.uint8)
    pad = (-len(bits_arr)) % 8
    if pad:
        bits_arr = np.pad(bits_arr, (0, pad))
    return np.packbits(bits_arr)


def unpack(packed: np.ndarray, bits: int, length: int) -> np.ndarray:
    """Unpack uint8 bytes back into int64 indices."""
    raw = np.unpackbits(packed)
    raw = raw[: length * bits]
    rows = raw.reshape(length, bits).astype(np.int64)
    powers = 1 << np.arange(bits - 1, -1, -1, dtype=np.int64)
    return (rows * powers).sum(axis=1)

# 3. Random Rotations and Projection Generators

def make_rotation(d: int, seed: int) -> np.ndarray:
    """Deterministic Haar-distributed orthogonal matrix generator on CPU."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float32)


def make_projection(m: int, d: int, seed: int) -> np.ndarray:
    """Draw random i.i.d Gaussian projection matrix S."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, d)).astype(np.float32)


def rotation_seed(layer_idx: int, head_idx: int) -> int:
    """Deterministic seed function."""
    return (layer_idx + head_idx) * (layer_idx + head_idx + 1) // 2 + head_idx

# 4. Core Quantizers (MSE & Product)

class TurboQuantMSE:
    """MSE-optimal vector quantizer in NumPy."""
    def __init__(self, d: int, bits: int, layer_idx: int = 0, head_idx: int = 0):
        self.d = d
        self.bits = bits
        seed = rotation_seed(layer_idx, head_idx)
        self.Pi = make_rotation(d, seed)
        self.codebook = get_codebook(d, bits)

    def quantize(self, x: np.ndarray) -> dict:
        x = x.astype(np.float32)
        norm = np.linalg.norm(x)
        x_unit = x / (norm if norm > 1e-12 else 1e-12)
        y = self.Pi @ x_unit
        indices = quantize_coords_jit(y, self.codebook)
        packed = pack(indices, self.bits)
        return {"packed": packed, "norm": norm, "length": self.d}

    def dequantize(self, state: dict) -> np.ndarray:
        indices = unpack(state["packed"], self.bits, state["length"])
        y_hat = self.codebook[indices]
        x_unit = self.Pi.T @ y_hat
        return state["norm"] * x_unit


@njit(fastmath=True)
def inner_product_jit(Sq: np.ndarray, r_tilde: np.ndarray, r_norm: float, d: int) -> float:
    dot_product = 0.0
    for i in range(len(Sq)):
        dot_product += Sq[i] * r_tilde[i]
    return (math.sqrt(math.pi / 2) / d) * r_norm * dot_product

class TurboQuantProd:
    """Inner-product-optimal vector quantizer in NumPy (MSE + QJL)."""
    def __init__(self, d: int, bits: int, layer_idx: int = 0, head_idx: int = 0):
        if bits < 2:
            raise ValueError("TurboQuantProd requires bits >= 2 (MSE stage uses bits-1)")
        self.d = d
        self.bits = bits
        self.mse = TurboQuantMSE(d, bits - 1, layer_idx=layer_idx, head_idx=head_idx)
        qjl_seed = rotation_seed(layer_idx, head_idx) + 1
        self.S = make_projection(d, d, seed=qjl_seed) # m = d

    def quantize(self, x: np.ndarray) -> dict:
        x = x.astype(np.float32)
        mse_state = self.mse.quantize(x)
        x_hat_mse = self.mse.dequantize(mse_state)
        r = x - x_hat_mse
        projected = r @ self.S.T
        r_tilde = np.sign(projected).astype(np.int8)
        r_norm = np.linalg.norm(r)
        return {**mse_state, "r_tilde": r_tilde, "r_norm": r_norm}

    def dequantize(self, state: dict) -> np.ndarray:
        x_hat_mse = self.mse.dequantize(state)
        r_correction = (math.sqrt(math.pi / 2) / self.d) * state["r_norm"] * (
            self.S.T @ state["r_tilde"]
        )
        return x_hat_mse + r_correction

    def inner_product(self, q: np.ndarray, state: dict) -> float:
        """Estimate dot product <q, x> from compressed key state."""
        q = q.astype(np.float32)
        x_hat_mse = self.mse.dequantize(state)
        ip_mse = np.dot(q, x_hat_mse)
        # Precomputed query projection
        Sq = self.S @ q
        ip_residual = inner_product_jit(Sq, state["r_tilde"], state["r_norm"], self.d)
        return ip_mse + ip_residual

# 5. KV Cache Layer Manager

class TurboQuantKVCache:
    """Manages Key-Value cache for a single head in one layer."""
    def __init__(self, d: int, key_bits: int = 4, val_bits: int = 2,
                 layer_idx: int = 0, head_idx: int = 0):
        self.d = d
        self.key_q = TurboQuantProd(d, key_bits, layer_idx=layer_idx, head_idx=head_idx)
        self.val_q = TurboQuantMSE(d, val_bits, layer_idx=layer_idx, head_idx=head_idx)
        self._keys = []
        self._vals = []

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        self._keys.append(self.key_q.quantize(k))
        self._vals.append(self.val_q.quantize(v))

    def attn_scores(self, q: np.ndarray) -> np.ndarray:
        """Estimate attention dot products for all cached keys."""
        return np.array([
            self.key_q.inner_product(q, state) for state in self._keys
        ])

    def values(self) -> np.ndarray:
        """Dequantize and stack all values."""
        return np.stack([self.val_q.dequantize(s) for s in self._vals])

    def __len__(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._vals.clear()
