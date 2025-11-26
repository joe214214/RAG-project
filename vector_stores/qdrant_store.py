"""
Qdrant Vector Store Implementation.

Provides a unified interface for storing and searching vectors using Qdrant.
Supports both OLTP (optimized for speed) and OLAP (optimized for recall) collections.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff


@dataclass
class QdrantCollectionConfig:
    """Configuration for a Qdrant collection."""
    name: str
    vector_size: int = 384  # Default for all-MiniLM-L6-v2
    distance: str = "Cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    search_ef: int = 50
    on_disk: bool = False  # Store vectors on disk for large collections


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False  # gRPC is faster but REST is simpler to debug


class QdrantVectorStore:
    """
    Vector store implementation using Qdrant.
    
    Features:
    - HNSW index with configurable parameters
    - Support for payload filtering
    - Batch upsert and search
    - On-disk storage option for large collections
    """
    
    def __init__(self, config: QdrantConfig, collection_config: QdrantCollectionConfig):
        self.config = config
        self.collection_config = collection_config
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            prefer_grpc=config.prefer_grpc,
            check_compatibility=False,  # Suppress version warnings
        )
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        cc = self.collection_config
        
        # Map distance string to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance = distance_map.get(cc.distance, Distance.COSINE)
        
        if not self.client.collection_exists(cc.name):
            self.client.create_collection(
                collection_name=cc.name,
                vectors_config=VectorParams(
                    size=cc.vector_size,
                    distance=distance,
                    on_disk=cc.on_disk,
                ),
                hnsw_config=HnswConfigDiff(
                    m=cc.hnsw_m,
                    ef_construct=cc.hnsw_ef_construct,
                    on_disk=cc.on_disk,
                ),
            )
            print(f"Created collection '{cc.name}' with HNSW (m={cc.hnsw_m}, ef_construct={cc.hnsw_ef_construct})")
        else:
            print(f"Collection '{cc.name}' already exists")
    
    def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Insert or update vectors in the collection.
        
        Args:
            ids: Unique identifiers for each vector
            vectors: Numpy array of shape (n, dim)
            payloads: Optional metadata for each vector
            batch_size: Number of vectors to upsert per batch
        """
        if payloads is None:
            payloads = [{}] * len(ids)
        
        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            
            points = [
                models.PointStruct(
                    id=idx,
                    vector=vec.tolist(),
                    payload=payload,
                )
                for idx, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]
            
            self.client.upsert(
                collection_name=self.collection_config.name,
                points=points,
                wait=True,
            )
        
        print(f"Upserted {len(ids)} vectors to '{self.collection_config.name}'")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector of shape (dim,)
            top_k: Number of results to return
            filter_conditions: Optional Qdrant filter
            score_threshold: Minimum score threshold
            
        Returns:
            List of (id, score, payload) tuples
        """
        # Build search params with custom ef
        search_params = models.SearchParams(
            hnsw_ef=self.collection_config.search_ef,
            exact=False,  # Use HNSW, not exact search
        )
        
        results = self.client.query_points(
            collection_name=self.collection_config.name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
            search_params=search_params,
        )
        
        return [
            (str(hit.id), hit.score, hit.payload or {})
            for hit in results.points
        ]
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ) -> List[List[Tuple[str, float, Dict]]]:
        """
        Search for multiple query vectors in a single request.
        
        Args:
            query_vectors: Array of shape (n_queries, dim)
            top_k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        # Use query_points for each vector (batch via loop for compatibility)
        all_results = []
        for vec in query_vectors:
            results = self.client.query_points(
                collection_name=self.collection_config.name,
                query=vec.tolist(),
                limit=top_k,
            )
            all_results.append([
                (str(hit.id), hit.score, hit.payload or {})
                for hit in results.points
            ])
        return all_results
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_config.name)
        return {
            "name": self.collection_config.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
        }
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_config.name)
        print(f"Deleted collection '{self.collection_config.name}'")


def create_oltp_store(
    host: str = "localhost",
    port: int = 6333,
    vector_size: int = 384,
) -> QdrantVectorStore:
    """Create a vector store optimized for OLTP (fast factoid) queries."""
    return QdrantVectorStore(
        config=QdrantConfig(host=host, port=port),
        collection_config=QdrantCollectionConfig(
            name="rag_oltp_chunks",
            vector_size=vector_size,
            distance="Cosine",
            hnsw_m=16,
            hnsw_ef_construct=128,
            search_ef=50,  # Lower ef = faster search
        ),
    )


def create_olap_store(
    host: str = "localhost",
    port: int = 6333,
    vector_size: int = 384,
) -> QdrantVectorStore:
    """Create a vector store optimized for OLAP (analytical) queries."""
    return QdrantVectorStore(
        config=QdrantConfig(host=host, port=port),
        collection_config=QdrantCollectionConfig(
            name="rag_olap_chunks",
            vector_size=vector_size,
            distance="Cosine",
            hnsw_m=32,       # More edges for better recall
            hnsw_ef_construct=256,
            search_ef=200,   # Higher ef = better accuracy
        ),
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing Qdrant Vector Store...")
    
    # Create OLTP store
    oltp_store = create_oltp_store()
    
    # Generate random test data
    n_vectors = 1000
    dim = 384
    np.random.seed(42)
    
    # Use integer IDs (Qdrant prefers int or UUID)
    ids = list(range(n_vectors))
    vectors = np.random.rand(n_vectors, dim).astype(np.float32)
    # Normalize for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    payloads = [
        {"source": f"doc_{i // 100}", "chunk_type": "oltp"}
        for i in range(n_vectors)
    ]
    
    # Upsert
    start = time.time()
    oltp_store.upsert(ids, vectors, payloads)
    print(f"Upsert time: {time.time() - start:.2f}s")
    
    # Search
    query = np.random.rand(dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    start = time.time()
    results = oltp_store.search(query, top_k=10)
    print(f"Search time: {(time.time() - start) * 1000:.2f}ms")
    
    print(f"\nTop 3 results:")
    for id, score, payload in results[:3]:
        print(f"  {id}: score={score:.4f}, source={payload.get('source')}")
    
    # Collection info
    info = oltp_store.get_collection_info()
    print(f"\nCollection info: {info}")
    
    print("\nQdrant Vector Store test PASSED!")

