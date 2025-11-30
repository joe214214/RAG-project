#!/usr/bin/env python3
"""
Cross-Encoder Rerankers for Query-Aware RAG

Implements differentiated reranking strategies per docs/Approach/reranker_recommendations.md:
- OLTP: No reranker or lightweight TinyBERT (speed priority)
- OLAP: High-quality cross-encoders like Zerank-2, BGE (quality priority)

Models supported:
- cross-encoder/ms-marco-TinyBERT-L-2-v2: Fastest (~14M params)
- cross-encoder/ms-marco-MiniLM-L-6-v2: Balanced (~22M params)
- BAAI/bge-reranker-base: High quality (~109M params)
- BAAI/bge-reranker-large: Higher quality (~335M params)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RerankResult:
    """Result from reranking."""
    text: str
    score: float
    original_score: float
    original_rank: int
    metadata: Dict


class BaseReranker:
    """Base class for rerankers."""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device  # None = auto-detect
        self._model = None
    
    def _get_device(self):
        """Auto-detect best device if not specified."""
        if self.device is not None:
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _load_model(self):
        """Lazy load model."""
        raise NotImplementedError
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank documents for a query."""
        raise NotImplementedError


class NoReranker(BaseReranker):
    """
    No reranking - just pass through original scores.
    Use for OLTP when speed is critical.
    """
    
    def __init__(self):
        super().__init__("none", "cpu")
    
    def _load_model(self):
        pass  # No model needed
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Pass through without reranking."""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            results.append(RerankResult(
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                original_score=doc.get("score", 0.0),
                original_rank=i,
                metadata=doc.get("metadata", {}),
            ))
        return results


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.
    
    Recommended models by use case:
    
    OLTP (Fast):
    - cross-encoder/ms-marco-TinyBERT-L-2-v2: ~14M params, fastest
    
    OLAP (Quality):
    - cross-encoder/ms-marco-MiniLM-L-6-v2: ~22M params, balanced
    - BAAI/bge-reranker-base: ~109M params, high quality
    - BAAI/bge-reranker-large: ~335M params, higher quality
    """
    
    # Models that require trust_remote_code
    # Note: zerank models removed due to size constraints
    TRUST_REMOTE_CODE_MODELS = {
        # "zeroentropy/zerank-1",  # Too large (~1B params), not recommended
        # "zeroentropy/zerank-2",  # Too large (4B params), not recommended
    }
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,  # None = auto-detect GPU/CPU
        max_length: int = 512,
    ):
        super().__init__(model_name, device)
        self.max_length = max_length
    
    def _load_model(self):
        """Load CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            
            device = self._get_device()
            trust_remote = self.model_name in self.TRUST_REMOTE_CODE_MODELS
            
            print(f"Loading reranker: {self.model_name} on {device}")
            if trust_remote:
                print(f"  (Using trust_remote_code=True for {self.model_name})")
            
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=device,
                trust_remote_code=trust_remote,
            )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: List of dicts with 'text', 'score', 'metadata' keys
            top_k: Number of results to return
            
        Returns:
            List of RerankResult sorted by reranker score
        """
        if not documents:
            return []
        
        model = self._load_model()
        
        # Prepare pairs for cross-encoder
        texts = [doc.get("text", "") for doc in documents]
        pairs = [(query, text) for text in texts]
        
        # Score all pairs
        start_time = time.time()
        scores = model.predict(pairs, show_progress_bar=False)
        elapsed = time.time() - start_time
        
        # Create results with both scores
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                text=doc.get("text", ""),
                score=float(score),
                original_score=doc.get("score", 0.0),
                original_rank=i,
                metadata=doc.get("metadata", {}),
            ))
        
        # Sort by reranker score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]


class OLTPReranker:
    """
    OLTP-optimized reranker (speed priority).
    
    Options:
    - No reranking (fastest) - DEFAULT
    - TinyBERT (fast, slight quality boost)
    """
    
    def __init__(
        self,
        use_reranker: bool = False,
        device: str = None,
    ):
        if use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
                device=device,
            )
        else:
            self.reranker = NoReranker()
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[RerankResult]:
        """Rerank for OLTP queries."""
        return self.reranker.rerank(query, documents, top_k)


class OLAPReranker:
    """
    OLAP-optimized reranker (quality priority).
    
    Recommended models (per reranker_recommendations.md):
    - "cross-encoder/ms-marco-MiniLM-L-6-v2": Fast, decent quality (~22M params)
    - "BAAI/bge-reranker-base": High quality (~109M params)
    - "BAAI/bge-reranker-large": Higher quality (~335M params)
    
    Note: zerank-1 (~1B params) and zerank-2 (4B params) are too large for most systems and not recommended.
    """
    
    RECOMMENDED_MODELS = {
        "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "bge-base": "BAAI/bge-reranker-base",
        "bge-large": "BAAI/bge-reranker-large",
        # "zerank-1": "zeroentropy/zerank-1",  # Too large (~1B params), not recommended
        # "zerank-2": "zeroentropy/zerank-2",  # Too large (4B params), not recommended
    }
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
    ):
        # Allow shorthand names
        if model_name in self.RECOMMENDED_MODELS:
            model_name = self.RECOMMENDED_MODELS[model_name]
        
        self.reranker = CrossEncoderReranker(
            model_name=model_name,
            device=device,
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank for OLAP queries."""
        return self.reranker.rerank(query, documents, top_k)


class RerankerFactory:
    """
    Factory to create appropriate reranker based on query type.
    """
    
    def __init__(
        self,
        device: str = None,  # None = auto-detect
        oltp_use_reranker: bool = False,
        olap_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.device = device
        self._oltp_reranker = None
        self._olap_reranker = None
        self.oltp_use_reranker = oltp_use_reranker
        self.olap_model = olap_model
    
    def get_oltp_reranker(self) -> OLTPReranker:
        """Get OLTP reranker (lazy loaded)."""
        if self._oltp_reranker is None:
            self._oltp_reranker = OLTPReranker(
                use_reranker=self.oltp_use_reranker,
                device=self.device,
            )
        return self._oltp_reranker
    
    def get_olap_reranker(self) -> OLAPReranker:
        """Get OLAP reranker (lazy loaded)."""
        if self._olap_reranker is None:
            self._olap_reranker = OLAPReranker(
                model_name=self.olap_model,
                device=self.device,
            )
        return self._olap_reranker
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        query_type: str = "oltp",
        top_k: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank documents based on query type.
        
        Args:
            query: Search query
            documents: Retrieved documents
            query_type: "oltp" or "olap"
            top_k: Number of results
            
        Returns:
            Reranked results
        """
        if query_type == "oltp":
            reranker = self.get_oltp_reranker()
            return reranker.rerank(query, documents, top_k=min(top_k, 5))
        else:
            reranker = self.get_olap_reranker()
            return reranker.rerank(query, documents, top_k=top_k)


# Testing
if __name__ == "__main__":
    import torch
    
    print("=" * 60)
    print("Reranker Test")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠ No GPU, using CPU")
        device = "cpu"
    
    # Sample documents
    query = "What is machine learning?"
    documents = [
        {"text": "Machine learning is a subset of AI that learns from data.", "score": 0.8, "metadata": {"id": 1}},
        {"text": "Deep learning uses neural networks with many layers.", "score": 0.75, "metadata": {"id": 2}},
        {"text": "Python is a programming language.", "score": 0.6, "metadata": {"id": 3}},
        {"text": "ML algorithms can identify patterns in large datasets.", "score": 0.7, "metadata": {"id": 4}},
        {"text": "The weather today is sunny.", "score": 0.5, "metadata": {"id": 5}},
    ]
    
    # Test OLTP (no reranker)
    print("\n1. OLTP (No Reranker):")
    oltp = OLTPReranker(use_reranker=False)
    results = oltp.rerank(query, documents, top_k=3)
    for r in results:
        print(f"   Score: {r.score:.4f} | {r.text[:50]}...")
    
    # Test OLTP (with TinyBERT)
    print("\n2. OLTP (TinyBERT Reranker):")
    oltp_rerank = OLTPReranker(use_reranker=True, device=device)
    results = oltp_rerank.rerank(query, documents, top_k=3)
    for r in results:
        print(f"   Score: {r.score:.4f} (was rank {r.original_rank}) | {r.text[:50]}...")
    
    # Test OLAP (MiniLM)
    print("\n3. OLAP (MiniLM Cross-Encoder):")
    olap = OLAPReranker(model_name="minilm", device=device)
    results = olap.rerank(query, documents, top_k=3)
    for r in results:
        print(f"   Score: {r.score:.4f} (was rank {r.original_rank}) | {r.text[:50]}...")
    
    # Test OLAP (BGE) - optional, larger model
    # print("\n4. OLAP (BGE-reranker-base):")
    # olap_bge = OLAPReranker(model_name="bge-base", device=device)
    # results = olap_bge.rerank(query, documents, top_k=3)
    
    # Test OLAP (Zerank-2) - optional, requires more VRAM
    # print("\n5. OLAP (Zerank-2 - SOTA):")
    # olap_zerank = OLAPReranker(model_name="zerank-2", device=device)
    # results = olap_zerank.rerank(query, documents, top_k=3)
    
    print("\n✅ Reranker tests complete!")
    print("\nAvailable OLAP models:")
    for name, full_name in OLAPReranker.RECOMMENDED_MODELS.items():
        print(f"   {name}: {full_name}")
