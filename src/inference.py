import time
import os
import psutil
import numpy as np
from src.memory_manager import DynamicLayerManager
from src.turboquant_numpy import TurboQuantKVCache
from src.attention import gemv_q4_0, rms_norm_jit, apply_rope_jit, silu_activation_jit, gemv_q4_0_sparse_x

class SparseQuantModel:
    def __init__(self, model_path: str, active_ratio: float = 0.15):
        self.layer_manager = DynamicLayerManager(model_path, active_ratio)
        self.num_layers = self.layer_manager.num_layers
        self.hidden_dim = self.layer_manager.hidden_dim
        
        # Determine num_heads dynamically based on typical head dimension of 128
        self.head_dim = 128
        self.num_heads = self.hidden_dim // self.head_dim  # Q heads (e.g. 40)
        
        # Detect GQA: K/V may have fewer heads than Q
        k_shape = self.layer_manager.loader.get_tensor_shape("blk.0.attn_k.weight")
        self.kv_dim = k_shape[0]  # K output dimension (e.g. 1024)
        self.num_kv_heads = self.kv_dim // self.head_dim  # KV heads (e.g. 8)
        self.num_groups = self.num_heads // self.num_kv_heads  # heads per KV group (e.g. 5)
        print(f"GQA Config: {self.num_heads} Q-heads, {self.num_kv_heads} KV-heads, group_size={self.num_groups}")
        
        # Initialize TurboQuant Key-Value caches — one per KV head (not Q head)
        self.kv_caches = [
            [TurboQuantKVCache(d=self.head_dim, key_bits=4, val_bits=2, layer_idx=L, head_idx=H)
             for H in range(self.num_kv_heads)]
            for L in range(self.num_layers)
        ]
        
        # Generate rotary embeddings sine/cosine caches
        self.cos_cache, self.sin_cache = self._precompute_rope(2048, self.head_dim)

    def _precompute_rope(self, max_seq_len: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
        inv_freq = 1.0 / (1000000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        return np.cos(emb), np.sin(emb)

    def forward_token(self, token_id: int, pos: int) -> int:
        """Runs the forward pass for a single token, dynamically loading layers and caching activations."""
        # 1. Embeddings lookup
        x = self.layer_manager.embeddings[token_id].copy() # (hidden_dim,)
        
        cos = self.cos_cache[pos]
        sin = self.sin_cache[pos]
        
        for L in range(self.num_layers):
            # --- Self Attention ---
            residual = x.copy()
            norm_x = rms_norm_jit(x, self.layer_manager.attn_cache[f"blk.{L}.attn_norm.weight"])
            
            # Load QKV weight projections from RAM
            q_packed, q_scales = self.layer_manager.get_attn_layer(L, "attn_q")
            k_packed, k_scales = self.layer_manager.get_attn_layer(L, "attn_k")
            v_packed, v_scales = self.layer_manager.get_attn_layer(L, "attn_v")
            
            # Compute QKV projections at JIT speed
            q_proj = gemv_q4_0(norm_x, q_packed, q_scales)
            k_proj = gemv_q4_0(norm_x, k_packed, k_scales)
            v_proj = gemv_q4_0(norm_x, v_packed, v_scales)
            
            # Reshape to heads — Q has more heads than K/V in GQA
            q_heads = q_proj.reshape(self.num_heads, self.head_dim)       # (40, 128)
            k_heads = k_proj.reshape(self.num_kv_heads, self.head_dim)    # (8, 128)
            v_heads = v_proj.reshape(self.num_kv_heads, self.head_dim)    # (8, 128)
            
            # Apply rotary embeddings (RoPE applies per-head independently)
            q_rot = apply_rope_jit(q_heads, q_heads, cos, sin)[0]  # Only need rotated Q
            k_rot = apply_rope_jit(k_heads, k_heads, cos, sin)[0]  # Only need rotated K
            
            # Append K/V to caches (one cache per KV head)
            for kv_h in range(self.num_kv_heads):
                self.kv_caches[L][kv_h].append(k_rot[kv_h], v_heads[kv_h])
            
            # Attention scores and outputs per Q head (GQA: multiple Q heads share one KV head)
            attn_heads = np.empty((self.num_heads, self.head_dim), dtype=np.float32)
            for H in range(self.num_heads):
                kv_h = H // self.num_groups  # Which KV head this Q head reads from
                cache = self.kv_caches[L][kv_h]
                
                # Calculate attention scores using this Q head against the shared KV cache
                scores = cache.attn_scores(q_rot[H])
                weights = np.exp(scores - np.max(scores))
                weights /= np.sum(weights)
                
                # Attention weighted sum over compressed Values
                values = cache.values()
                attn_heads[H] = np.dot(weights, values)
                
            # Concatenate heads and project output
            attn_concated = attn_heads.flatten()
            o_packed, o_scales = self.layer_manager.get_attn_layer(L, "attn_output")
            o_proj = gemv_q4_0(attn_concated, o_packed, o_scales)
            
            x = residual + o_proj
            
            # --- Feed-Forward Network (FFN) ---
            residual = x.copy()
            norm_ffn = rms_norm_jit(x, self.layer_manager.attn_cache[f"blk.{L}.ffn_norm.weight"])
            
            # Dynamically stream only the active FFN columns from SSD
            (active_indices, gate_packed, gate_scales, 
             up_packed, up_scales, 
             down_packed, down_scales) = self.layer_manager.stream_ffn_layer(L, norm_ffn)
            
            # Compute SwiGLU projections on active neurons using standard JIT kernel (pre-sliced weights)
            gate_out = gemv_q4_0(norm_ffn, gate_packed, gate_scales)
            up_out = gemv_q4_0(norm_ffn, up_packed, up_scales)
            ffn_activated = silu_activation_jit(gate_out, up_out)
            
            # Down projection: projects back to hidden dimension using our sparse inputs JIT kernel
            down_out = gemv_q4_0_sparse_x(active_indices, ffn_activated, down_packed, down_scales)
            
            # Free the active FFN memory immediately
            self.layer_manager.free_ffn_layer()
            
            x = residual + down_out
            
        # 3. Final logits generation
        final_norm = rms_norm_jit(x, self.layer_manager.output_norm)
        logits = gemv_q4_0(final_norm, self.layer_manager.output_packed, self.layer_manager.output_scales)
        
        # Greedy sampling
        return int(np.argmax(logits))

    def generate(self, prompt_tokens: list[int], max_new_tokens: int = 32) -> list[int]:
        print(f"Ingesting prompt tokens ({len(prompt_tokens)})...")
        generated = []
        
        # Process prompt tokens (prefill stage)
        pos = 0
        for token in prompt_tokens[:-1]:
            _ = self.forward_token(token, pos)
            pos += 1
            
        # Start generation stage
        current_token = prompt_tokens[-1]
        process = psutil.Process(os.getpid())
        
        print("Generating response tokens...")
        for i in range(max_new_tokens):
            start_time = time.time()
            next_token = self.forward_token(current_token, pos)
            latency = time.time() - start_time
            
            generated.append(next_token)
            current_token = next_token
            pos += 1
            
            ram_mb = process.memory_info().rss / (1024 * 1024)
            print(f"  Token {i+1:02d}: ID={next_token:<5} | Latency={latency:.2f}s | Process RSS={ram_mb:.1f} MB")
            
            # Exit loop if end of text token reached (typically Qwen uses 151643 for end-of-text)
            if next_token in [151643, 151645]: 
                break
                
        return generated

    def generate_profiled(self, prompt_tokens: list[int], max_new_tokens: int = 32) -> tuple[list[int], dict]:
        """Runs generation while collecting detailed telemetry metrics for analysis."""
        process = psutil.Process(os.getpid())
        
        metrics = {
            "prefill_tokens": len(prompt_tokens),
            "prefill_time": 0.0,
            "prefill_rss_start": process.memory_info().rss / (1024 * 1024),
            "prefill_rss_end": 0.0,
            "decode_latencies": [],
            "decode_rss": [],
            "total_rss_peak": 0.0,
        }
        
        # Prefill stage
        start_prefill = time.time()
        pos = 0
        for token in prompt_tokens[:-1]:
            _ = self.forward_token(token, pos)
            pos += 1
        metrics["prefill_time"] = time.time() - start_prefill
        metrics["prefill_rss_end"] = process.memory_info().rss / (1024 * 1024)
        
        generated = []
        current_token = prompt_tokens[-1]
        
        peak_rss = metrics["prefill_rss_end"]
        
        for i in range(max_new_tokens):
            start_decode = time.time()
            next_token = self.forward_token(current_token, pos)
            latency = time.time() - start_decode
            
            generated.append(next_token)
            current_token = next_token
            pos += 1
            
            rss_now = process.memory_info().rss / (1024 * 1024)
            peak_rss = max(peak_rss, rss_now)
            
            metrics["decode_latencies"].append(latency)
            metrics["decode_rss"].append(rss_now)
            
            # Exit loop if end of text token reached
            if next_token in [151643, 151645]: 
                break
                
        metrics["total_rss_peak"] = peak_rss
        return generated, metrics
