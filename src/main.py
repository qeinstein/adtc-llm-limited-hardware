import os
import sys
import json
import psutil
from llama_cpp import Llama
from src.inference import SparseQuantModel
from src.evaluator import LocalEvaluator

def load_metadata_config() -> tuple[str, str]:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    meta_path = os.path.join(ROOT, "metadata.json")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    model_path = os.path.join(ROOT, meta["_runtime"]["model_path"])
    return model_path, meta["domain"]

def main():
    try:
        model_path, domain = load_metadata_config()
    except Exception as e:
        print(f"Error loading metadata.json: {e}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}.")
        print("Please configure metadata.json and download the model using download_model.sh first.")
        sys.exit(1)

    print("======================================================================")
    print("      ADTC 2026 Laptop LLM Challenge: Custom Sparse-FFN Serving Engine")
    print("======================================================================")
    
    # 1. Load vocabulary-only helper (takes <10 MB RAM) to tokenize/detokenize GGUF text
    print("Loading GGUF vocabulary compiler...")
    try:
        vocab_engine = Llama(model_path=model_path, vocab_only=True, verbose=False)
    except Exception as e:
        print(f"Warning: Failed to load llama.cpp vocabulary parser ({e}). Using robust fallback tokenizer.")
        class FallbackTokenizer:
            def tokenize(self, text_bytes):
                text = text_bytes.decode('utf-8')
                return [ord(c) % 1000 for c in text]
            def detokenize(self, token_ids):
                return "".join(chr(t if t >= 32 and t < 127 else 63) for t in token_ids).encode('utf-8')
            def close(self):
                pass
        vocab_engine = FallbackTokenizer()
    
    # 2. Load our custom memory-swapped execution sandbox
    print("Initializing Custom SparseQuantModel...")
    model = SparseQuantModel(model_path=model_path, active_ratio=0.15)
    
    # Test queries
    test_queries = [
        "Joto la mwili la nyuzi 39 Celsius na kikohozi kikavu kwa mtoto wa miaka miwili. Nifanye nini?",
        "What are the common symptoms of Malaria?"
    ]

    for idx, query in enumerate(test_queries):
        print("\n----------------------------------------------------------------------")
        print(f"Processing Query {idx + 1}: '{query}'")
        print("----------------------------------------------------------------------")
        
        # Tokenize using the GGUF vocabulary
        prompt_bytes = vocab_engine.tokenize(bytes(query, "utf-8"))
        prompt_tokens = [int(t) for t in prompt_bytes]
        
        # Generate tokens using our custom layer-swapping pipeline
        start_time = os.times()
        generated_tokens = model.generate(prompt_tokens, max_new_tokens=32)
        
        # Detokenize response back to text
        output_bytes = vocab_engine.detokenize(generated_tokens)
        response_text = output_bytes.decode("utf-8", errors="ignore")
        
        print("\nResponse Generated:")
        print(response_text)
        
    # 3. Evaluate Swahili clinical concept accuracy
    print("\n======================================================================")
    print("Running Swahili Clinical Concept Accuracy Evaluation...")
    print("======================================================================")
    eval_set_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "swahili_eval_set.json")
    if os.path.exists(eval_set_path):
        evaluator = LocalEvaluator(vocab_engine, model, eval_set_path)
        evaluator.evaluate_model()
    else:
        print(f"Warning: Evaluation set not found at {eval_set_path}.")

    print("\n======================================================================")
    print("Execution completed.")
    print("======================================================================")
    
    vocab_engine.close()
    model.layer_manager.close()

if __name__ == "__main__":
    main()
