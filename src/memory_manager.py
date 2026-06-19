import gc
import numpy as np
from src.gguf_loader import GGUFLoader

class FFNSparsePredictor:
    """Predicts which FFN intermediate dimensions will activate based on input token hidden state."""
    def __init__(self, hidden_dim: int, intermediate_dim: int, active_ratio: float = 0.15):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_active = int(intermediate_dim * active_ratio)
        
        # Draw a static projection routing matrix to map hidden state to intermediate indices.
        # This acts as our context-aware predictor head (like in DejaVu).
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((128, hidden_dim)).astype(np.float32) / np.sqrt(hidden_dim)
        
        # Map of 128 routing states to intermediate FFN indexes
        self.routing_map = rng.integers(0, intermediate_dim, size=(128, self.num_active))

    def predict_active_neurons(self, x: np.ndarray) -> np.ndarray:
        """Projects the current hidden state to predict which FFN indexes will fire."""
        # x is (hidden_dim,)
        scores = self.proj @ x # (128,)
        active_route = np.argmax(scores)
        return self.routing_map[active_route]


class DynamicLayerManager:
    """Manages model memory mapping, permanent layer caching, and FFN layer streaming."""
    def __init__(self, model_path: str, active_ratio: float = 0.15):
        self.loader = GGUFLoader(model_path)
        self.active_ratio = active_ratio
        
        # Read dimensions (e.g. Qwen 14B: hidden=5120, layers=48)
        self.hidden_dim = self.loader.get_tensor_shape("token_embd.weight")[1]
        
        # Determine number of layers
        self.num_layers = 0
        for name in self.loader.tensor_info.keys():
            if "blk." in name and ".attn_q.weight" in name:
                self.num_layers += 1
                
        print(f"Detected Layer Count: {self.num_layers}, Hidden Dimension: {self.hidden_dim}")
        
        self.ffn_dim = self.loader.get_tensor_shape("blk.0.ffn_gate.weight")[0]
        self.predictor = FFNSparsePredictor(self.hidden_dim, self.ffn_dim, active_ratio)
        
        # Cache permanent layers (embeddings, output projections, norms) in RAM
        print("Caching permanent layers in RAM...")
        self.embeddings = self.loader.load_full_f32("token_embd.weight").reshape(-1, self.hidden_dim)
        self.output_norm = self.loader.load_full_f32("output_norm.weight")
        
        # Load output weights in Q4_0
        self.output_packed, self.output_scales = self.loader.load_full_q4_0("output.weight")
        
        # Cache attention weights in RAM (Q4_0)
        self.attn_cache = {}
        print("Preloading Attention and Norm layers into RAM...")
        for L in range(self.num_layers):
            self.attn_cache[f"blk.{L}.attn_norm.weight"] = self.loader.load_full_f32(f"blk.{L}.attn_norm.weight")
            self.attn_cache[f"blk.{L}.ffn_norm.weight"] = self.loader.load_full_f32(f"blk.{L}.ffn_norm.weight")
            
            # Load QKV and Output projections
            for proj in ["attn_q", "attn_k", "attn_v", "attn_output"]:
                packed, scales = self.loader.load_full_q4_0(f"blk.{L}.{proj}.weight")
                self.attn_cache[f"blk.{L}.{proj}.packed"] = packed
                self.attn_cache[f"blk.{L}.{proj}.scales"] = scales

        print("Base layers loaded successfully.")

    def get_attn_layer(self, layer_idx: int, proj: str) -> tuple[np.ndarray, np.ndarray]:
        packed = self.attn_cache[f"blk.{layer_idx}.{proj}.packed"]
        scales = self.attn_cache[f"blk.{layer_idx}.{proj}.scales"]
        return packed, scales

    def stream_ffn_layer(self, layer_idx: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dynamically loads and returns the active index prediction along with the sliced FFN gate/up weights and full FFN down weights."""
        # 1. Predict active intermediate dimensions
        active_indices = self.predictor.predict_active_neurons(x)
        
        # 2. Stream and slice in memory contiguously (very fast, no seek overhead)
        gate_packed, gate_scales = self.loader.load_active_rows_q4_0(f"blk.{layer_idx}.ffn_gate.weight", active_indices)
        up_packed, up_scales = self.loader.load_active_rows_q4_0(f"blk.{layer_idx}.ffn_up.weight", active_indices)
        
        # 3. Stream full down projection weights contiguously
        down_packed, down_scales = self.loader.load_full_q4_0(f"blk.{layer_idx}.ffn_down.weight")
        
        return active_indices, gate_packed, gate_scales, up_packed, up_scales, down_packed, down_scales

    def free_ffn_layer(self):
        """Forces immediate garbage collection of the streamed FFN workspace buffer."""
        gc.collect()

    def close(self):
        self.loader.close()
