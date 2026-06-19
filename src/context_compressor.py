import re
import numpy as np

def clean_text(text: str) -> str:
    """Standardize spacing and strip markdown tags for cleaner analysis."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[#\*\_]', '', text)
    return text.strip()

def get_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def compress_context(query: str, document_text: str, max_words: int = 400) -> str:
    """Compresses a long clinical document into a dense set of highly relevant sentences.
    
    This keeps prompt lengths short, which directly prevents KV Cache RAM expansion
    and speeds up prompt prefill on Core i5 CPUs.
    """
    cleaned_doc = clean_text(document_text)
    sentences = get_sentences(cleaned_doc)
    
    if not sentences:
        return ""
        
    # Extract query terms for term-frequency matching
    query_words = set(re.findall(r'\w+', query.lower()))
    
    # Score sentences based on intersection with query terms + position bias
    scores = []
    for idx, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\w+', sentence.lower())
        if not sentence_words:
            scores.append(0.0)
            continue
            
        intersection = query_words.intersection(set(sentence_words))
        
        # Jaccard-like similarity coefficient
        overlap_score = len(intersection) / np.log(len(sentence_words) + 2)
        
        # Clinical bias: earlier sentences in manuals usually contain summary definitions
        position_bias = 1.0 / (idx + 1.0)
        
        total_score = overlap_score + 0.1 * position_bias
        scores.append(total_score)
        
    # Sort sentences by score descending
    ranked_indices = np.argsort(scores)[::-1]
    
    selected_sentences = []
    current_word_count = 0
    
    # Reassemble high-scoring sentences in their original chronological order
    chosen_indices = []
    for idx in ranked_indices:
        sent = sentences[idx]
        word_count = len(sent.split())
        if current_word_count + word_count <= max_words:
            chosen_indices.append(idx)
            current_word_count += word_count
        if current_word_count >= max_words:
            break
            
    # Sort chosen sentences chronologically to preserve document narrative flow
    chosen_indices.sort()
    compressed_text = " ".join([sentences[i] for i in chosen_indices])
    
    return compressed_text
