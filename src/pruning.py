import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

def copy_metadata_field(writer, field):
    name = field.name
    types = field.types
    parts = field.parts
    data_indices = field.data
    
    def get_py_val(idx, val_type):
        x = parts[idx]
        if val_type == GGUFValueType.STRING:
            return bytes(x).decode('utf-8', errors='ignore')
        elif val_type in [GGUFValueType.FLOAT32, GGUFValueType.FLOAT64]:
            return float(x.item() if hasattr(x, 'item') else x[0] if hasattr(x, '__getitem__') else x)
        elif val_type == GGUFValueType.BOOL:
            return bool(x.item() if hasattr(x, 'item') else x[0] if hasattr(x, '__getitem__') else x)
        else:
            return int(x.item() if hasattr(x, 'item') else x[0] if hasattr(x, '__getitem__') else x)

    vtype = types[0]
    
    if len(types) == 1:
        py_val = get_py_val(data_indices[0], vtype)
        writer.add_key_value(name, py_val, vtype)
    else:
        sub_type = types[1]
        py_val = [get_py_val(idx, sub_type) for idx in data_indices]
        writer.add_key_value(name, py_val, vtype, sub_type)

def ggml_to_numpy_type(tensor_type) -> np.dtype:
    val = int(tensor_type)
    if val == 0:
        return np.dtype('float32')
    elif val == 1:
        return np.dtype('float16')
    else:
        return np.dtype('uint8')

def dequantize_q4_0(packed: np.ndarray, scales: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    """Dequantizes a Q4_0 tensor back to a float32 NumPy array (vectorized)."""
    num_blocks_per_row = in_features // 32
    # Reshape packed bytes to (out_features, num_blocks_per_row, 16)
    p = packed.reshape(out_features, num_blocks_per_row, 16)
    # Extract low and high nibbles
    low = (p & 0xF).astype(np.float32) - 8.0    # positions 0..15
    high = (p >> 4).astype(np.float32) - 8.0     # positions 16..31
    # Interleave: [low(16), high(16)] per block → 32 values
    dequant = np.concatenate([low, high], axis=2)  # (out, blocks, 32)
    # Scale each block
    dequant *= scales[:, :, np.newaxis]
    return dequant.reshape(out_features, in_features)

def dequantize_q4_1(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    """Dequantizes a Q4_1 tensor back to a float32 NumPy array (vectorized)."""
    num_blocks_per_row = in_features // 32
    p = packed.reshape(out_features, num_blocks_per_row, 16)
    low = (p & 0xF).astype(np.float32)
    high = (p >> 4).astype(np.float32)
    dequant = np.concatenate([low, high], axis=2)
    dequant = dequant * scales[:, :, np.newaxis] + biases[:, :, np.newaxis]
    return dequant.reshape(out_features, in_features)

def quantize_q4_0(weight: np.ndarray) -> bytes:
    """Quantizes a float32 matrix back into Q4_0 byte format (vectorized)."""
    out_features, in_features = weight.shape
    assert in_features % 32 == 0
    # Reshape into blocks of 32
    blocks = weight.reshape(-1, 32)  # (num_blocks, 32)
    # Compute scale per block: max_abs / 8
    max_vals = np.max(np.abs(blocks), axis=1)  # (num_blocks,)
    scales = max_vals / 8.0
    scales = np.where(scales < 1e-6, 1e-6, scales)
    # Quantize all values: round(val / scale) + 8, clip to [0, 15]
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]) + 8.0, 0, 15).astype(np.uint8)
    # Pack nibbles: low = block[j], high = block[j+16] for j in 0..15
    low = quantized[:, :16]
    high = quantized[:, 16:]
    packed = (high << 4) | low  # (num_blocks, 16)
    # Build output: [scale_f16(2 bytes), packed(16 bytes)] per block = 18 bytes
    scale_bytes = scales.astype(np.float16).view(np.uint8).reshape(-1, 2)
    block_data = np.concatenate([scale_bytes, packed], axis=1)  # (num_blocks, 18)
    return block_data.tobytes()

def prune_gguf(src_path: str, dst_path: str, prune_ratio: float = 0.5):
    """Reads a source Q4_0 GGUF file, prunes the FFN intermediate dimension, and saves it."""
    if not os.path.exists(src_path):
        print(f"Error: Source model file not found at {src_path}")
        return
        
    print(f"Reading source GGUF from {src_path}...")
    reader = GGUFReader(src_path)
    
    # 1. Determine model architecture
    arch = "qwen2"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = str(field.parts[-1])
            break
            
    print(f"Detected model architecture: {arch}")
    
    # 2. Get FFN intermediate dimension shape from gate weight
    ffn_gate_tensor = None
    for tensor in reader.tensors:
        if "ffn_gate.weight" in tensor.name:
            ffn_gate_tensor = tensor
            break
            
    if not ffn_gate_tensor:
        print("Error: Could not locate ffn_gate.weight tensor in model.")
        return
        
    # GGUF shape is (cols, rows)
    cols, rows = ffn_gate_tensor.shape
    intermediate_dim = rows  # out_features of ffn_gate
    hidden_dim = cols       # in_features of ffn_gate
    
    new_intermediate_dim = int(intermediate_dim * (1 - prune_ratio))
    # Must be a multiple of 32 for Q4_0 block quantization
    new_intermediate_dim = (new_intermediate_dim // 32) * 32
    if new_intermediate_dim == 0:
        new_intermediate_dim = 32
        
    print(f"Pruning FFN intermediate dimension from {intermediate_dim} -> {new_intermediate_dim}")
    
    writer = GGUFWriter(dst_path, arch)
    
    # 3. Copy all metadata KV pairs
    for field in reader.fields.values():
        name = field.name
        # Skip some metadata that we will write custom or overwrite
        if name in ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]:
            continue
            
        # Update feed_forward_length in metadata
        if name == f"{arch}.feed_forward_length":
            writer.add_feed_forward_length(new_intermediate_dim)
            continue
            
        # Copy raw GGUF value type
        copy_metadata_field(writer, field)

    # 4. Define shapes and prepare tensors
    tensor_buffers = []
    
    print("Preparing tensor headers...")
    for tensor in reader.tensors:
        name = tensor.name
        gguf_shape = list(tensor.shape)
        
        # Check if FFN tensor
        is_gate_up = "ffn_gate.weight" in name or "ffn_up.weight" in name
        is_down = "ffn_down.weight" in name
        
        if is_gate_up:
            # Original logical GGUF shape is [hidden_dim, intermediate_dim]
            # Pruned logical Python shape: [new_intermediate_dim, hidden_dim]
            # Byte shape to pass to add_tensor_info: [new_intermediate_dim, (hidden_dim // 32) * 18]
            bytes_per_row = (hidden_dim // 32) * 18
            shape_to_pass = [new_intermediate_dim, bytes_per_row]
            nbytes = new_intermediate_dim * bytes_per_row
            writer.add_tensor_info(name, shape_to_pass, np.dtype('uint8'), nbytes, raw_dtype=GGMLQuantizationType.Q4_0)
            tensor_buffers.append((name, "prune_gate_up", tensor))
            
        elif is_down:
            # Original logical GGUF shape is [intermediate_dim, hidden_dim]
            # Pruned logical Python shape: [hidden_dim, new_intermediate_dim]
            # Byte shape to pass to add_tensor_info: [hidden_dim, (new_intermediate_dim // 32) * 18]
            bytes_per_row = (new_intermediate_dim // 32) * 18
            shape_to_pass = [hidden_dim, bytes_per_row]
            nbytes = hidden_dim * bytes_per_row
            writer.add_tensor_info(name, shape_to_pass, np.dtype('uint8'), nbytes, raw_dtype=GGMLQuantizationType.Q4_0)
            tensor_buffers.append((name, "prune_down", tensor))
            
        else:
            # Copy other tensors as-is (converting from GGUF order to Python order)
            if len(gguf_shape) == 2:
                cols, rows = gguf_shape[0], gguf_shape[1]
                is_quantized = int(tensor.tensor_type) not in [0, 1]
                if is_quantized:
                    from gguf.quants import GGML_QUANT_SIZES
                    qtype = GGMLQuantizationType(int(tensor.tensor_type))
                    block_size, type_size = GGML_QUANT_SIZES.get(qtype, (32, 18))
                    bytes_per_row = (cols // block_size) * type_size
                    shape_to_pass = [rows, bytes_per_row]
                    dtype_to_pass = np.dtype('uint8')
                else:
                    shape_to_pass = [rows, cols]
                    dtype_to_pass = ggml_to_numpy_type(tensor.tensor_type)
            else:
                shape_to_pass = gguf_shape
                dtype_to_pass = ggml_to_numpy_type(tensor.tensor_type)
                
            writer.add_tensor_info(name, shape_to_pass, dtype_to_pass, tensor.n_bytes, raw_dtype=tensor.tensor_type)
            tensor_buffers.append((name, "copy", tensor))
            
    # 5. Write GGUF structural sections
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    
    # 6. Read, process, and write actual tensor bytes
    file_handle = open(src_path, 'rb')
    
    print("Writing tensor data blocks...")
    for idx, (name, mode, tensor) in enumerate(tensor_buffers):
        print(f"  [{idx+1}/{len(tensor_buffers)}] Writing {name} ({mode})...")
        
        file_handle.seek(tensor.data_offset)
        data = file_handle.read(tensor.n_bytes)
        
        if mode == "copy":
            # Direct write
            writer.write_tensor_data(np.frombuffer(data, dtype=np.uint8))
            
        elif mode == "prune_gate_up":
            # Dequantize, slice, quantize
            # Original GGUF shape: (hidden_dim, intermediate_dim)
            # Python shape: (intermediate_dim, hidden_dim)
            num_blocks = (intermediate_dim * hidden_dim) // 32
            blocks = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 18)
            scales = blocks[:, :2].copy().view(np.float16).reshape(intermediate_dim, hidden_dim // 32)
            packed = blocks[:, 2:].copy().reshape(intermediate_dim, (hidden_dim // 32) * 16)
            
            # Dequantize full
            w_float32 = dequantize_q4_0(packed, scales, intermediate_dim, hidden_dim)
            # Slice row-wise in python (intermediate dimension)
            w_sliced = w_float32[:new_intermediate_dim, :]
            # Quantize
            w_packed_bytes = quantize_q4_0(w_sliced)
            writer.write_tensor_data(np.frombuffer(w_packed_bytes, dtype=np.uint8))
            
        elif mode == "prune_down":
            # Dequantize, slice, quantize
            # Original GGUF shape: (intermediate_dim, hidden_dim)
            # Python shape: (hidden_dim, intermediate_dim)
            # ffn_down can be Q4_0 (18 bytes/block) or Q4_1 (20 bytes/block) depending on the layer
            num_blocks = (hidden_dim * intermediate_dim) // 32
            ttype = int(tensor.tensor_type)
            
            if ttype == 3:  # Q4_1: 20 bytes per block (2 scale + 2 bias + 16 packed)
                blocks = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 20)
                scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(hidden_dim, intermediate_dim // 32)
                biases = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(hidden_dim, intermediate_dim // 32)
                packed = blocks[:, 4:].copy().reshape(hidden_dim, (intermediate_dim // 32) * 16)
                w_float32 = dequantize_q4_1(packed, scales, biases, hidden_dim, intermediate_dim)
            else:  # Q4_0: 18 bytes per block (2 scale + 16 packed)
                blocks = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 18)
                scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(hidden_dim, intermediate_dim // 32)
                packed = blocks[:, 2:].copy().reshape(hidden_dim, (intermediate_dim // 32) * 16)
                w_float32 = dequantize_q4_0(packed, scales, hidden_dim, intermediate_dim)
            
            # Slice column-wise in python (intermediate dimension)
            w_sliced = w_float32[:, :new_intermediate_dim]
            # Quantize to Q4_0 (which fits the serving engine)
            w_packed_bytes = quantize_q4_0(w_sliced)
            writer.write_tensor_data(np.frombuffer(w_packed_bytes, dtype=np.uint8))
            
    file_handle.close()
    writer.close()
    print(f"Pruned GGUF successfully saved to {dst_path}!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pruning.py <src_path> <dst_path> [prune_ratio]")
        sys.exit(1)
        
    src = sys.argv[1]
    dst = sys.argv[2]
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    prune_gguf(src, dst, ratio)
