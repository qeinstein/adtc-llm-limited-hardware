import json
import re
import numpy as np

class BM25Retriever:
    """A lightweight, pure NumPy implementation of the BM25 search algorithm.
    
    Operates offline with zero memory footprint and zero external dependencies.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.vocab = {}
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.doc_frequencies = {}
        self.idf = {}
        self.tf = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def fit(self, documents: list[dict]):
        """Fits the BM25 index on a list of document dicts containing {'title': ..., 'text': ...}."""
        self.documents = documents
        self.doc_lengths = []
        self.tf = []
        
        # Build vocabulary and term frequencies
        for doc in documents:
            words = self._tokenize(doc['title'] + " " + doc['text'])
            self.doc_lengths.append(len(words))
            
            # Compute term frequencies for this document
            doc_tf = {}
            for word in words:
                doc_tf[word] = doc_tf.get(word, 0) + 1
            self.tf.append(doc_tf)
            
            # Track document frequencies for IDF calculation
            unique_words = set(words)
            for word in unique_words:
                self.doc_frequencies[word] = self.doc_frequencies.get(word, 0) + 1
                
        self.avg_doc_len = np.mean(self.doc_lengths) if self.doc_lengths else 0.0
        N = len(documents)
        
        # Calculate IDF (Inverse Document Frequency) for each term
        for word, df in self.doc_frequencies.items():
            # Standard BM25 IDF formulation
            self.idf[word] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def retrieve(self, query: str, top_n: int = 2) -> list[dict]:
        """Calculates BM25 scores for all documents and returns the top_n matches."""
        query_words = self._tokenize(query)
        scores = np.zeros(len(self.documents))
        
        for idx, doc_len in enumerate(self.doc_lengths):
            doc_tf = self.tf[idx]
            score = 0.0
            for word in query_words:
                if word in doc_tf:
                    tf_val = doc_tf[word]
                    word_idf = self.idf.get(word, 0.0)
                    
                    # BM25 weight formulation:
                    # Score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    numerator = tf_val * (self.k1 + 1.0)
                    denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += word_idf * (numerator / denominator)
            scores[idx] = score
            
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for i in range(min(top_n, len(self.documents))):
            doc_idx = ranked_indices[i]
            if scores[doc_idx] > 0.0:  # Only return documents with matching terms
                results.append({
                    **self.documents[doc_idx],
                    "score": float(scores[doc_idx])
                })
        return results
