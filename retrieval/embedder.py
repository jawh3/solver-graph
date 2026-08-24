import hashlib
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
from knowledge_graph.embedding_mock import MockEmbeddingGenerator

log = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract interface for text embedding generation.

    Note (spec §7, BUILD_CHECKLIST C2): the retrieval embedder is a DEDICATED
    embedding model, NOT the solver model. Use a real `VLLMEmbedder` against an
    embedding endpoint in coordinated runs; `MockEmbedder` for GPU-free dev.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate a dense vector embedding for the given text."""
        pass

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Default batch = per-item embed; real backends may override for speed."""
        return [self.embed(t) for t in texts]


class MockEmbedder(Embedder):
    """
    Deterministic mock embedder wrapping MockEmbeddingGenerator.
    Used for testing and deterministic experiment sweeps without a GPU.
    """
    
    def __init__(self, dim: int = 1536, seed: int = 42):
        self.generator = MockEmbeddingGenerator(dim=dim, seed=seed)
        
    def embed(self, text: str) -> np.ndarray:
        return self.generator.generate(text=text)


class VLLMEmbedder(Embedder):
    """
    Real embedder that hits an OpenAI-compatible /v1/embeddings endpoint
    provided by a managed vLLM instance.
    """
    
    def __init__(self, endpoint_url: str, model_name: str, timeout: int = 10):
        """
        Args:
            endpoint_url: Full URL to the embeddings endpoint (e.g., http://localhost:8000/v1/embeddings)
            model_name: Name of the model to request (e.g., Qwen3.5-27B-FP8)
            timeout: Request timeout in seconds
        """
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout = timeout
        
    def embed(self, text: str) -> np.ndarray:
        payload = {
            "model": self.model_name,
            "input": text
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.endpoint_url, 
            data=data, 
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                embedding = result['data'][0]['embedding']
                return np.array(embedding, dtype=np.float32)
        except urllib.error.URLError as e:
            log.error(f"Failed to connect to vLLM endpoint at {self.endpoint_url}: {e}")
            raise
        except (KeyError, IndexError) as e:
            log.error(f"Unexpected response format from vLLM endpoint: {e}")
            raise


class DiskCacheEmbedder(Embedder):
    """Wraps any Embedder with a content-addressed on-disk cache.

    Node/query text -> vector is expensive on the real path and embeddings never
    change for fixed (model, text). Caching makes coordinated GPU runs cheap to
    resume and keeps dev fully reproducible (BUILD_CHECKLIST C2: "with disk
    response cache"). The cache key includes the wrapped embedder's identity so a
    model swap does not return stale vectors.
    """

    def __init__(self, inner: Embedder, cache_dir: str | Path = "results/embed_cache",
                 namespace: Optional[str] = None):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace or inner.__class__.__name__
        self._mem: dict[str, np.ndarray] = {}

    def _key(self, text: str) -> str:
        h = hashlib.sha1(f"{self.namespace}\x00{text}".encode("utf-8")).hexdigest()
        return h

    def embed(self, text: str) -> np.ndarray:
        k = self._key(text)
        if k in self._mem:
            return self._mem[k]
        path = self.cache_dir / f"{k}.npy"
        if path.exists():
            vec = np.load(path)
            self._mem[k] = vec
            return vec
        vec = self.inner.embed(text)
        np.save(path, vec)
        self._mem[k] = vec
        return vec


class SentenceTransformerEmbedder(Embedder):
    """Real embedder via sentence-transformers (used in the laptop pilot).

    The lab uses a dedicated vLLM embedding endpoint (`VLLMEmbedder`); on the
    laptop a small local bi-encoder (default all-MiniLM-L6-v2, 384-d, ~90 MB) is a
    disclosed substitution — it is NOT the solver model (spec §7 keeps the
    retrieval embedder separate). Runs on GPU if available, else CPU.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # lazy (heavy dep)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        vs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [np.asarray(v, dtype=np.float32) for v in vs]


def build_embedder(mock: bool = True, dim: int = 1536, seed: int = 42,
                   endpoint_url: str = "", model_name: str = "",
                   cache_dir: Optional[str] = None,
                   backend: str = "mock") -> Embedder:
    """Factory honoring mock+real duality. backend ∈ {mock, vllm, sentence_transformers}."""
    if backend == "sentence_transformers":
        base: Embedder = SentenceTransformerEmbedder(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")
    elif backend == "vllm" or (not mock and endpoint_url):
        base = VLLMEmbedder(endpoint_url=endpoint_url, model_name=model_name)
    else:
        base = MockEmbedder(dim=dim, seed=seed)
    if cache_dir:
        return DiskCacheEmbedder(base, cache_dir=cache_dir)
    return base
