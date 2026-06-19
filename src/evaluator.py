import json
import os
from llama_cpp import Llama
from src.inference import SparseQuantModel

class LocalEvaluator:
    def __init__(self, vocab_engine: Llama, model: SparseQuantModel, eval_set_path: str):
        self.vocab_engine = vocab_engine
        self.model = model
        with open(eval_set_path, 'r', encoding='utf-8') as f:
            self.eval_cases = json.load(f)

    def evaluate_model(self) -> dict:
        print("\n======================================================================")
        print("         Evaluating Mother-Tongue (Swahili) Model Accuracy")
        print("======================================================================")
        
        results = []
        total_score = 0.0

        for idx, case in enumerate(self.eval_cases):
            print(f"\nEvaluating Case {idx + 1}: {case['description']}")
            query = case['query']
            
            # Tokenize using vocab_engine
            prompt_bytes = self.vocab_engine.tokenize(bytes(query, "utf-8"))
            prompt_tokens = [int(t) for t in prompt_bytes]
            
            # Generate tokens using our custom 14B pipeline
            generated_tokens = self.model.generate(prompt_tokens, max_new_tokens=48)
            
            # Detokenize
            output_bytes = self.vocab_engine.detokenize(generated_tokens)
            response_text = output_bytes.decode("utf-8", errors="ignore").strip()
            
            output = response_text.lower()
            matched_keywords = []
            
            for kw in case['gold_keywords']:
                if kw.lower() in output:
                    matched_keywords.append(kw)
            
            case_accuracy = len(matched_keywords) / len(case['gold_keywords']) * 100
            total_score += case_accuracy
            
            print(f"  Query: '{query}'")
            print(f"  Advisory: '{response_text}'")
            print(f"  Matched keywords: {matched_keywords} / {case['gold_keywords']}")
            print(f"  Case Accuracy Score: {case_accuracy:.1f}%")
            
            results.append({
                "query": query,
                "response": response_text,
                "accuracy_score": case_accuracy,
                "matched_keywords": matched_keywords
            })

        mean_accuracy = total_score / len(self.eval_cases) if self.eval_cases else 0.0
        print("\n----------------------------------------------------------------------")
        print(f"Overall Swahili Clinical Accuracy: {mean_accuracy:.2f}%")
        print("----------------------------------------------------------------------")
        
        return {
            "mean_accuracy_score": mean_accuracy,
            "individual_results": results
        }
