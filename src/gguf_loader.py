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
        """Loads and parses a full Q4_0 tensor into scales and packed bytes."""
        info = self.tensor_info[name]
        out_features, in_features = self.get_tensor_shape(name)
        
        self.file_handle.seek(info["offset"])
        data = self.file_handle.read(info["size"])
        
        # Q4_0 format: blocks of 32 weights. 
        # Each block: 2 bytes (float16 scale) + 16 bytes (32 x 4-bit values).
        # Total bytes per block = 18.
        num_blocks = (out_features * in_features) // 32
        
        # Reshape data to (num_blocks, 18)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 18)
        
        # Extract scales (float16 view of first 2 bytes)
        # Reshape scales to (out_features, in_features // 32)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(out_features, in_features // 32)
        
        # Extract packed weights (remaining 16 bytes)
        # Reshape packed to (out_features, (in_features // 32) * 16)
        packed = blocks[:, 2:].copy().reshape(out_features, (in_features // 32) * 16)
        
        return packed, scales

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
