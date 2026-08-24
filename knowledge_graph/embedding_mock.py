import hashlib
import numpy as np


class MockEmbeddingGenerator:
    """
    Generates deterministic mock embeddings for knowledge graph nodes based on string hashing.
    Used for local testing when vLLM is not available.
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.seed = seed
        
    def generate(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding for a given text.
        
        Args:
            text: The input text to embed.
            
        Returns:
            A normalized 1D numpy array of shape (dim,).
        """
        # Create a deterministic seed from the combination of the global seed and the text hash
        text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        combined_seed = (self.seed + text_hash) % (2**32 - 1)
        
        rng = np.random.RandomState(combined_seed)
        
        # Generate random vector
        vec = rng.randn(self.dim).astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return vec
