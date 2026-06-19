import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

def copy_metadata_field(writer, field):
    name = field.name
    types = field.types
    parts = field.parts
    
    if len(types) == 1:
        vtype = types[0]
        val = parts[0]
        if vtype == GGUFValueType.STRING:
            py_val = bytes(val).decode('utf-8', errors='ignore')
        elif vtype in [GGUFValueType.FLOAT32, GGUFValueType.FLOAT64]:
            py_val = float(val.item() if hasattr(val, 'item') else val[0] if hasattr(val, '__getitem__') else val)
        elif vtype == GGUFValueType.BOOL:
            py_val = bool(val.item() if hasattr(val, 'item') else val[0] if hasattr(val, '__getitem__') else val)
        else:
            py_val = int(val.item() if hasattr(val, 'item') else val[0] if hasattr(val, '__getitem__') else val)
            
        writer.add_key_value(name, py_val, vtype)
    else:
        vtype = types[0]
        sub_type = types[1]
        
        if sub_type == GGUFValueType.STRING:
            py_val = [bytes(x) for x in parts]
        elif sub_type in [GGUFValueType.FLOAT32, GGUFValueType.FLOAT64]:
            py_val = [float(x.item() if hasattr(x, 'item') else x) for x in parts]
        elif sub_type == GGUFValueType.BOOL:
            py_val = [bool(x.item() if hasattr(x, 'item') else x) for x in parts]
        else:
            py_val = [int(x.item() if hasattr(x, 'item') else x) for x in parts]
            
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
    """Dequantizes a Q4_0 tensor back to a float32 NumPy array."""
    # packed shape: (out_features, (in_features // 32) * 16)
    # scales shape: (out_features, in_features // 32)
    num_blocks = in_features // 32
    unpacked = np.zeros((out_features, in_features), dtype=np.float32)
    
    for i in range(out_features):
        for b in range(num_blocks):
            scale = scales[i, b]
            block_offset = b * 16
            for j in range(16):
                byte_val = packed[i, block_offset + j]
                q_low = np.float32(byte_val & 15) - 8.0
                q_high = np.float32(byte_val >> 4) - 8.0
                
                # GGML Q4_0 layout:
                # low nibble corresponds to weight at index b * 32 + j
                # high nibble corresponds to weight at index b * 32 + j + 16
                unpacked[i, b * 32 + j] = q_low * scale
                unpacked[i, b * 32 + j + 16] = q_high * scale
                
    return unpacked

def quantize_q4_0(weight: np.ndarray) -> bytes:
    """Quantizes a float32 matrix back into Q4_0 byte format."""
    out_features, in_features = weight.shape
    assert in_features % 32 == 0
    
    num_blocks = (out_features * in_features) // 32
    flat_weight = weight.flatten()
    blocks_data = bytearray()
    
    for b in range(num_blocks):
        block = flat_weight[b*32 : (b+1)*32]
        max_val = np.max(np.abs(block))
        scale = max_val / 8.0 if max_val > 1e-6 else 0.0
        
        blocks_data.extend(np.float16(scale).tobytes())
        
        packed_bytes = bytearray(16)
        for j in range(16):
            val_low = block[j]
            val_high = block[j + 16]
            
            q_low = int(np.clip(np.round(val_low / scale) + 8.0, 0, 15)) if scale > 0 else 8
            q_high = int(np.clip(np.round(val_high / scale) + 8.0, 0, 15)) if scale > 0 else 8
            
            packed_bytes[j] = (q_high << 4) | q_low
            
        blocks_data.extend(packed_bytes)
        
    return bytes(blocks_data)

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
                    bytes_per_row = (cols // 32) * 18
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
            num_blocks = (hidden_dim * intermediate_dim) // 32
            blocks = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 18)
            scales = blocks[:, :2].copy().view(np.float16).reshape(hidden_dim, intermediate_dim // 32)
            packed = blocks[:, 2:].copy().reshape(hidden_dim, (intermediate_dim // 32) * 16)
            
            # Dequantize full
            w_float32 = dequantize_q4_0(packed, scales, hidden_dim, intermediate_dim)
            # Slice column-wise in python (intermediate dimension)
            w_sliced = w_float32[:, :new_intermediate_dim]
            # Quantize
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
