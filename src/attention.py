import math
import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def gemv_q4_0(x: np.ndarray, qs_packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """JIT-compiled, parallel Matrix-Vector Multiply directly on Q4_0 bytes.
    
    Performs out = W @ x, where W is represented by packed 4-bit bytes and scales.
    Bypasses temporary float matrix allocations, keeping execution RAM extremely small.
    """
    out_features = qs_packed.shape[0]
    num_blocks = scales.shape[1]
    y = np.zeros(out_features, dtype=np.float32)
    
    for i in prange(out_features):
        sum_val = 0.0
        for b in range(num_blocks):
            scale = scales[i, b]
            block_offset = b * 16
            for j in range(16):
                byte_val = qs_packed[i, block_offset + j]
                # Unpack the 4-bit nibbles (range 0-15 mapped back to -8 to 7)
                q_low = np.float32(byte_val & 15) - 8.0
                q_high = np.float32(byte_val >> 4) - 8.0
                
                # Correct index mapping:
                # low nibble corresponds to weight at index b * 32 + j
                # high nibble corresponds to weight at index b * 32 + j + 16
                sum_val += scale * (q_low * x[b * 32 + j] + q_high * x[b * 32 + j + 16])
        y[i] = sum_val
    return y

@njit(fastmath=True, parallel=True)
def gemv_q4_0_sparse_y(x: np.ndarray, qs_packed: np.ndarray, scales: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
    """Computes y = W[active_indices] @ x, calculating only the predicted active rows."""
    num_active = len(active_indices)
    num_blocks = scales.shape[1]
    y = np.zeros(num_active, dtype=np.float32)
    
    for k in prange(num_active):
        i = active_indices[k]
        sum_val = 0.0
        for b in range(num_blocks):
            scale = scales[i, b]
            block_offset = b * 16
            for j in range(16):
                byte_val = qs_packed[i, block_offset + j]
                q_low = np.float32(byte_val & 15) - 8.0
                q_high = np.float32(byte_val >> 4) - 8.0
                sum_val += scale * (q_low * x[b * 32 + j] + q_high * x[b * 32 + j + 16])
        y[k] = sum_val
    return y

@njit(fastmath=True, parallel=True)
def gemv_q4_0_sparse_x(active_indices: np.ndarray, x_active: np.ndarray, qs_packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Computes y = W @ x, where x has non-zero values x_active at active_indices."""
    out_features = qs_packed.shape[0]
    y = np.zeros(out_features, dtype=np.float32)
    
    for i in prange(out_features):
        sum_val = 0.0
        for k in range(len(active_indices)):
            idx = active_indices[k]
            val = x_active[k]
            if val == 0.0:
                continue
            
            b = idx // 32
            j = idx % 32
            
            scale = scales[i, b]
            block_offset = b * 16
            byte_idx = block_offset + (j // 2)
            byte_val = qs_packed[i, byte_idx]
            
            if j % 2 == 0:
                q = np.float32(byte_val & 15) - 8.0
            else:
                q = np.float32(byte_val >> 4) - 8.0
                
            sum_val += scale * q * val
        y[i] = sum_val
    return y

@njit(fastmath=True)
def rms_norm_jit(x: np.ndarray, weight: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Root Mean Square Normalization kernel."""
    variance = np.mean(x ** 2)
    return x * weight * np.float32(1.0 / np.sqrt(variance + epsilon))

@njit(fastmath=True)
def apply_rope_jit(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applies Rotary Position Embeddings to query and key heads."""
    n_heads, d = q.shape
    half_d = d // 2
    
    q_rot = np.empty_like(q)
    k_rot = np.empty_like(k)
    
    for h in range(n_heads):
        for i in range(half_d):
            q_rot[h, i] = q[h, i] * cos[i] - q[h, half_d + i] * sin[i]
            q_rot[h, half_d + i] = q[h, half_d + i] * cos[i] + q[h, i] * sin[i]
            
            k_rot[h, i] = k[h, i] * cos[i] - k[h, half_d + i] * sin[i]
            k_rot[h, half_d + i] = k[h, half_d + i] * cos[i] + k[h, i] * sin[i]
            
    return q_rot, k_rot

@njit(fastmath=True)
def silu_activation_jit(gate_out: np.ndarray, up_out: np.ndarray) -> np.ndarray:
    """SwiGLU activation: SiLU(gate_out) * up_out."""
    n = len(gate_out)
    res = np.empty(n, dtype=np.float32)
    for i in range(n):
        sig = 1.0 / (1.0 + math.exp(-gate_out[i]))
        res[i] = gate_out[i] * sig * up_out[i]
    return res
