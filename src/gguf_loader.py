import os
import struct
import numpy as np
from gguf import GGUFReader

class GGUFLoader:
    """Parses GGUF headers, indexes tensor file offsets, and loads/unpacks Q4_0 slices on-demand."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"Index parsing GGUF model: {model_path}...")
        self.reader = GGUFReader(model_path)
        self.tensor_info = {}
        
        # Build dictionary of tensor offsets and configurations
        for tensor in self.reader.tensors:
            name = tensor.name
            shape = tensor.shape # typically (cols, rows) in GGUF
            # GGUF weights are stored transposed: (out_features, in_features)
            self.tensor_info[name] = {
                "offset": tensor.data_offset,
                "shape": shape, # (hidden_dim, out_features) or similar
                "type": tensor.tensor_type,
                "size": tensor.n_bytes
            }
            
        self.file_handle = open(model_path, 'rb')

    def get_tensor_shape(self, name: str) -> tuple:
        # GGUF shape is columns, rows. We return (out_features, in_features)
        info = self.tensor_info[name]
        shape = info["shape"]
        if len(shape) == 2:
            return (shape[1], shape[0])
        return shape

    def load_full_q4_0(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Loads and parses a full Q4_0 tensor into scales and packed bytes.
        Handles Q6_K tensors by dequantizing to float32 first, then re-encoding as Q4_0."""
        info = self.tensor_info[name]
        out_features, in_features = self.get_tensor_shape(name)
        ttype = int(info["type"])
        
        self.file_handle.seek(info["offset"])
        data = self.file_handle.read(info["size"])
        
        if ttype == 14:  # Q6_K — dequantize to float32, then encode as Q4_0
            w_f32 = self._dequantize_q6_k(data, out_features, in_features)
            return self._encode_f32_as_q4_0(w_f32, out_features, in_features)
        
        # Standard Q4_0 path
        num_blocks = (out_features * in_features) // 32
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 18)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(out_features, in_features // 32)
        packed = blocks[:, 2:].copy().reshape(out_features, (in_features // 32) * 16)
        
        return packed, scales
    
    def _dequantize_q6_k(self, data: bytes, out_features: int, in_features: int) -> np.ndarray:
        """Dequantize Q6_K data to float32 (vectorized). Q6_K: block_size=256, type_size=210 bytes."""
        block_size = 256
        type_size = 210
        total_elements = out_features * in_features
        num_blocks = total_elements // block_size
        
        raw = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, type_size)
        
        # Parse block fields
        ql = raw[:, :128]          # (num_blocks, 128) — low 4 bits
        qh = raw[:, 128:192]       # (num_blocks, 64)  — high 2 bits
        sc = raw[:, 192:208].view(np.int8)  # (num_blocks, 16) — sub-block scales
        d = raw[:, 208:210].copy().view(np.float16).astype(np.float32)  # (num_blocks, 1)
        
        # Extract 4-bit low nibbles for all 256 positions
        # ql[i] contains two 4-bit values: low nibble = positions 2i, high nibble = positions 2i+1
        ql_low = (ql & 0xF).astype(np.int32)    # even indices: 0,2,4,...
        ql_high = (ql >> 4).astype(np.int32)     # odd indices: 1,3,5,...
        # Interleave to get all 256 positions in order
        q4 = np.zeros((num_blocks, 256), dtype=np.int32)
        q4[:, 0::2] = ql_low
        q4[:, 1::2] = ql_high
        
        # Extract 2-bit high parts for all 256 positions
        # qh[i] contains four 2-bit values for positions 4i, 4i+1, 4i+2, 4i+3
        q2 = np.zeros((num_blocks, 256), dtype=np.int32)
        for shift in range(4):
            q2[:, shift::4] = (qh >> (shift * 2)) & 0x3
        
        # Reconstruct 6-bit quantized values: q = (q4 | (q2 << 4)) - 32
        q6 = (q4 | (q2 << 4)) - 32  # (num_blocks, 256), signed 6-bit
        
        # Apply sub-block scales: 16 sub-blocks of 16 weights each
        q6_sub = q6.reshape(num_blocks, 16, 16).astype(np.float32)  # (blocks, 16 sub-blocks, 16 weights)
        scales = (d * sc.astype(np.float32))  # (num_blocks, 16)
        result = q6_sub * scales[:, :, np.newaxis]  # broadcast scale per sub-block
        
        return result.reshape(out_features, in_features)
    
    def _encode_f32_as_q4_0(self, w_f32: np.ndarray, out_features: int, in_features: int) -> tuple[np.ndarray, np.ndarray]:
        """Encode a float32 matrix as Q4_0 packed/scales arrays for the serving engine."""
        blocks = w_f32.reshape(-1, 32)
        max_vals = np.max(np.abs(blocks), axis=1)
        scales_f32 = max_vals / 8.0
        scales_f32 = np.where(scales_f32 < 1e-6, 1e-6, scales_f32)
        
        quantized = np.clip(np.round(blocks / scales_f32[:, np.newaxis]) + 8.0, 0, 15).astype(np.uint8)
        low = quantized[:, :16]
        high = quantized[:, 16:]
        packed = (high << 4) | low  # (num_blocks, 16)
        
        blocks_per_row = in_features // 32
        packed = packed.reshape(out_features, blocks_per_row * 16)
        scales_out = scales_f32.reshape(out_features, blocks_per_row)
        
        return packed, scales_out

    def load_active_rows_q4_0(self, name: str, active_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Loads the full tensor contiguously and slices the active rows in memory.
        
        This is extremely fast compared to random seeking, and the memory is discarded immediately after execution.
        """
        info = self.tensor_info[name]
        out_features, in_features = self.get_tensor_shape(name)
        
        self.file_handle.seek(info["offset"])
        data = self.file_handle.read(info["size"])
        
        blocks_per_row = in_features // 32
        num_active = len(active_indices)
        
        # Reshape to (out_features, blocks_per_row, 18)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(out_features, blocks_per_row, 18)
        
        # Slice only the active rows
        active_blocks = blocks[active_indices]
        
        # Extract scales and packed values
        scales = active_blocks[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(num_active, blocks_per_row)
        packed = active_blocks[:, :, 2:].copy().reshape(num_active, blocks_per_row * 16)
        
        return packed, scales

    def load_full_f32(self, name: str) -> np.ndarray:
        """Loads a full unquantized float32/float16 tensor (e.g., embeddings or norms)."""
        info = self.tensor_info[name]
        self.file_handle.seek(info["offset"])
        data = self.file_handle.read(info["size"])
        
        # Load based on type and convert to float32 to ensure compatibility
        # Check standard GGUF types: GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1
        if info["type"] == 1:
            return np.frombuffer(data, dtype=np.float16).astype(np.float32)
        return np.frombuffer(data, dtype=np.float32).copy()

    def close(self):
        self.file_handle.close()
