#!/usr/bin/env python3
"""
Main RAG Pipeline with Query-Aware Routing

Automatically routes queries to appropriate collections based on classifier:
- OLTP queries → oltp_chunks (fine-grained, fast retrieval)
- OLAP queries → olap_chunks (coarse-grained, high-recall retrieval)

Usage:
    # Interactive mode
    python scripts/rag_pipeline.py --interactive --qdrant-host ecetesla0
    
    # Single query
    python scripts/rag_pipeline.py --query "What is machine learning?" --qdrant-host ecetesla0
    
    # With reranking
    python scripts/rag_pipeline.py --query "Compare AI and ML" --use-reranker --qdrant-host ecetesla0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class RetrievalResult:
    """Single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    score: float
    chunk_type: str
    source_file: str
    section: str
    level: str


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    query: str
    query_type: str  # "oltp" or "olap"
    confidence: float
    collection: str
    retrieved_chunks: List[RetrievalResult]
    reranked: bool
    metadata: Dict
    answer: Optional[str] = None  # Generated answer from LLM
    answer_cost_usd: Optional[float] = None  # Cost for answer generation
    answer_latency_ms: Optional[float] = None  # Answer generation latency


class QueryAwareRAGPipeline:
    """
    Main RAG pipeline with automatic query routing.
    
    Flow:
    1. Classify query (OLTP vs OLAP)
    2. Select appropriate collection
    3. Retrieve from collection
    4. Optionally rerank
    5. Return results
    """
    
    def __init__(
        self,
        router_model_path: str = "models/feature_router.pkl",
        classifier_type: str = "feature",  # "feature" or "transformer"
        classifier_model: str = None,  # For transformer: model path or name
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_hybrid: bool = False,
        hybrid_alpha: float = 0.5,
        use_query_expansion: bool = True,
        use_llm_answer: bool = False,
        llm_model: str = "gpt-4o-mini",
        llm_max_tokens: int = 512,
        skip_classifier: bool = False,
        default_collection: str = "oltp_chunks",
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            router_model_path: Path to feature router model (if classifier_type="feature")
            classifier_type: "feature" (feature-based) or "transformer" (DistilBERT/BERT)
            classifier_model: Model path/name for transformer classifier (e.g., "distilbert-base-uncased", "classifier2/distilbert-base-uncased")
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            embed_model: Embedding model name
            device: Device for embedding model
            use_hybrid: Enable hybrid retrieval (BM25 + dense) for OLTP queries
            hybrid_alpha: BM25 weight in hybrid retrieval (0-1). 0.5 = 50% BM25, 50% dense (tuned for optimal Recall@10)
            use_query_expansion: Enable LLM-based query expansion for BM25 retrieval (requires OPENAI_API_KEY)
            use_llm_answer: Enable LLM answer generation (requires OPENAI_API_KEY)
            llm_model: OpenAI model for answer generation (default: gpt-4o-mini)
            llm_max_tokens: Max tokens for answer generation (default: 512)
            skip_classifier: Skip classification entirely, use default_collection for all queries (minimal baseline)
            default_collection: Collection to use when skip_classifier=True (default: "oltp_chunks")
        """
        # Skip classifier for minimal baseline
        self.skip_classifier = skip_classifier
        self.default_collection = default_collection
        
        if skip_classifier:
            print("⚠️  Classifier disabled - using minimal baseline mode")
            print(f"   All queries will use collection: {default_collection}")
            self.classifier_type = None
            self.router = None
        else:
            # Load query router/classifier (no fallbacks)
            self.classifier_type = classifier_type
            if classifier_type == "feature":
                print(f"Loading feature-based router from {router_model_path}...")
                from scripts.feature_router import FeatureRouter
                self.router = FeatureRouter(router_model_path)
                print("✓ Feature router loaded")
            elif classifier_type == "transformer":
                model_path = classifier_model or "distilbert-base-uncased"
                print(f"Loading transformer classifier: {model_path}...")

                # Require classifier2 to exist; no silent fallbacks
                classifier2_path = project_root / "classifier2"
                if not classifier2_path.exists():
                    raise ImportError(
                        f"classifier2 directory not found at {classifier2_path}. "
                        f"Copy classifier2/ to the project root or use --classifier-type feature."
                    )
                
                # Check for required ml/classifier structure
                ml_classifier_path = classifier2_path / "ml" / "classifier" / "classifier_inference.py"
                if not ml_classifier_path.exists():
                    raise ImportError(
                        f"classifier2 structure incomplete. Expected {ml_classifier_path} but not found. "
                        f"Make sure classifier2/ml/classifier/classifier_inference.py exists, "
                        f"or use --classifier-type feature."
                    )

                # Add classifier2 to sys.path and import directly
                if str(classifier2_path) not in sys.path:
                    sys.path.insert(0, str(classifier2_path))

                try:
                    from ml.classifier.classifier_inference import QueryClassifier, ClassifierConfig
                except ImportError as e:
                    raise ImportError(
                        f"Failed to import transformer classifier: {e}. "
                        f"Make sure classifier2/ml/classifier/classifier_inference.py exists and is complete, "
                        f"or use --classifier-type feature."
                    ) from e

                # Resolve relative paths to absolute paths
                if not os.path.isabs(model_path):
                    # If it starts with "classifier2/", resolve relative to project_root
                    if model_path.startswith("classifier2/"):
                        actual_model_path = project_root / model_path
                    else:
                        # Try relative to project_root first, then classifier2
                        actual_model_path = project_root / model_path
                        if not actual_model_path.exists():
                            actual_model_path = classifier2_path / model_path
                    # Convert to absolute path string
                    model_path = str(actual_model_path.resolve())
                    
                    # Verify the path exists
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Model path not found: {model_path}. "
                            f"Make sure the model directory exists, or use --classifier-type feature."
                        )
                    
                    # Verify required tokenizer files exist (for better error messages)
                    required_files = ["config.json", "tokenizer_config.json"]
                    missing_files = []
                    for req_file in required_files:
                        if not os.path.exists(os.path.join(model_path, req_file)):
                            missing_files.append(req_file)
                    
                    # Check for vocab.txt or tokenizer.json (at least one is needed)
                    has_vocab = os.path.exists(os.path.join(model_path, "vocab.txt"))
                    has_tokenizer_json = os.path.exists(os.path.join(model_path, "tokenizer.json"))
                    
                    if missing_files:
                        raise FileNotFoundError(
                            f"Model directory incomplete. Missing files: {missing_files}. "
                            f"Path: {model_path}. Make sure all model files are copied."
                        )
                    
                    if not has_vocab and not has_tokenizer_json:
                        raise FileNotFoundError(
                            f"Tokenizer files missing. Need either 'vocab.txt' or 'tokenizer.json'. "
                            f"Path: {model_path}. Make sure all tokenizer files are copied."
                        )

                config = ClassifierConfig(
                    model_name_or_path=model_path,
                    device=device,
                )
                self.router = QueryClassifier(config)
                print("✓ Transformer classifier loaded")
            else:
                raise ValueError(f"Unknown classifier_type: {classifier_type}. Use 'feature' or 'transformer'")
        
        # If skipping classifier, don't load router
        if skip_classifier:
            pass  # Router already set to None
        
        # Store device for rerankers
        self.device = device
        
        # Load embedding model
        print(f"Loading embedding model ({embed_model}) on {device}...")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model, device=device)
        print("✓ Embedding model loaded")
        
        # Qdrant client (use gRPC for ~30% better performance)
        from qdrant_client import QdrantClient, models
        self.qdrant_client = QdrantClient(
            host=qdrant_host, 
            port=qdrant_port, 
            prefer_grpc=True,  # gRPC is faster than REST
            check_compatibility=False
        )
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Hybrid retrieval settings
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        self.use_query_expansion = use_query_expansion
        self.bm25_indexes = {}  # Lazy-loaded: {collection_name: BM25Retriever}
        
        # Rerankers (lazy loaded, supports multiple models)
        self.oltp_rerankers = {}
        self.olap_rerankers = {}
        
        # Query expansion cache (to avoid repeated LLM calls)
        self._query_expansion_cache = {}
        
        # LLM answer generator (optional)
        self.use_llm_answer = use_llm_answer
        self.llm_generator = None
        if use_llm_answer:
            try:
                from scripts.llm_answer_generator import LLMAnswerGenerator
                self.llm_generator = LLMAnswerGenerator(
                    model=llm_model,
                    max_tokens=llm_max_tokens,
                )
                print(f"✓ LLM answer generator initialized ({llm_model})")
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM answer generator: {e}")
                print("   Answer generation will be disabled.")
                self.use_llm_answer = False
        
        print(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        if use_hybrid:
            print(f"✓ Hybrid retrieval enabled (α={hybrid_alpha:.2f})")
    
    def classify_query(self, query: str) -> Tuple[str, float]:
        """
        Classify query as OLTP or OLAP using the selected classifier.
        
        Returns:
            (query_type, confidence) tuple
        """
        if self.skip_classifier:
            # Minimal baseline: return default query type
            return "oltp", 1.0  # Default to OLTP with full confidence
        
        if self.classifier_type == "feature":
            return self.router.classify_with_confidence(query)
        else:  # transformer
            label, confidence = self.router.classify(query)
            return label, confidence
    
    def _reciprocal_rank_fusion(
        self,
        bm25_ranks: List[Tuple[str, float]],
        dense_ranks: List[str],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF) for combining BM25 and dense rankings.
        
        RRF formula: score = 1/(rank + k) for each method, then sum.
        This avoids score normalization issues and is more robust than weighted combination.
        
        Args:
            bm25_ranks: List of (chunk_id, bm25_score) tuples, sorted by score descending
            dense_ranks: List of chunk_ids from dense retrieval, sorted by score descending
            k: RRF constant (typically 60, lower = more weight to top ranks)
            
        Returns:
            List of (chunk_id, rrf_score) tuples, sorted by score descending
        """
        rrf_scores = {}
        
        # BM25 RRF scores
        for rank, (chunk_id, _) in enumerate(bm25_ranks, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + k)
        
        # Dense RRF scores
        for rank, chunk_id in enumerate(dense_ranks, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + k)
        
        # Sort by combined RRF score (descending)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _get_hybrid_alpha(self, query_type: str) -> float:
        """
        Get hybrid alpha parameter based on query type.
        
        Research shows:
        - OLTP queries benefit from more dense retrieval (lower alpha: 0.3-0.5)
        - OLAP queries benefit from more BM25/keyword matching (higher alpha: 0.5-0.7)
        
        If self.hybrid_alpha is set to a non-default value (not 0.5), use that instead.
        
        Args:
            query_type: "oltp" or "olap"
            
        Returns:
            Optimal hybrid alpha for the query type
        """
        # Allow instance-level override (if user explicitly set hybrid_alpha)
        if self.hybrid_alpha != 0.5:  # Default is 0.5
            return self.hybrid_alpha
        
        # Otherwise, use query-type-specific defaults
        if query_type == "oltp":
            # OLTP: Precision focus, more dense retrieval
            return 0.4  # 40% BM25, 60% dense
        else:
            # OLAP: Recall focus, more BM25 for keyword matching
            return 0.6  # 60% BM25, 40% dense
    
    def _expand_query(self, query: str, top_n: int = 3, use_llm: bool = True) -> str:
        """
        Expand query with similar terms for BM25 retrieval using LLM.
        
        Uses OpenAI API to generate query variations and synonyms to improve BM25 keyword matching.
        Results are cached to avoid repeated API calls for the same query.
        
        Args:
            query: Original query
            top_n: Number of expansion terms/variations to add
            use_llm: Whether to use LLM expansion (if False, returns original query)
            
        Returns:
            Expanded query string with original query + variations
        """
        if not use_llm:
            return query
        
        # Check cache first
        cache_key = f"{query}_{top_n}"
        if cache_key in self._query_expansion_cache:
            return self._query_expansion_cache[cache_key]
        
        # Try LLM-based expansion
        try:
            expanded = self._expand_query_with_llm(query, top_n)
            self._query_expansion_cache[cache_key] = expanded
            return expanded
        except Exception as e:
            # Fallback to original query if LLM expansion fails
            print(f"⚠ Warning: Query expansion failed ({e}), using original query")
            self._query_expansion_cache[cache_key] = query
            return query
    
    def _expand_query_with_llm(self, query: str, top_n: int = 3) -> str:
        """
        Expand query using LLM (OpenAI API) to generate synonyms and variations.
        
        Args:
            query: Original query
            top_n: Number of expansion terms to generate
            
        Returns:
            Expanded query string
        """
        import os
        
        # Check for OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try to get from config or fallback
            return query
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Use a cheap, fast model for query expansion
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            
            prompt = f"""Generate {top_n} query variations and synonyms for the following search query to improve keyword-based retrieval (BM25).

Original query: "{query}"

Generate:
1. Synonyms for key terms
2. Alternative phrasings
3. Related terms that users might use

Return ONLY a comma-separated list of {top_n} terms/phrases (no explanations, no numbering, no quotes). Each term should be concise (1-4 words).

Example:
Query: "machine learning algorithms"
Output: deep learning models, AI techniques, neural networks

Query: "{query}"
Output:"""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a search query expansion assistant. Generate concise query variations and synonyms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3,  # Low temperature for consistency
            )
            
            expansions = response.choices[0].message.content.strip()
            
            # Parse expansions (comma-separated)
            expansion_terms = [term.strip() for term in expansions.split(",") if term.strip()]
            expansion_terms = expansion_terms[:top_n]  # Limit to top_n
            
            # Combine original query with expansions
            if expansion_terms:
                expanded_query = f"{query} {' '.join(expansion_terms)}"
                return expanded_query
            else:
                return query
                
        except ImportError:
            print("⚠ Warning: OpenAI library not installed. Install with: pip install openai")
            return query
        except Exception as e:
            print(f"⚠ Warning: LLM query expansion failed: {e}")
            return query
    
    def _build_bm25_index(self, collection: str) -> None:
        """Build BM25 index from Qdrant collection (lazy loading, thread-safe)."""
        # Double-check pattern for thread safety
        if collection in self.bm25_indexes:
            return
        
        # Use a lock to prevent concurrent index building
        # Note: In practice, this check-then-act pattern has a small race window,
        # but worst case is building the index twice (acceptable overhead)
        import threading
        if not hasattr(self, '_bm25_lock'):
            self._bm25_lock = threading.Lock()
        
        with self._bm25_lock:
            # Check again inside lock
            if collection in self.bm25_indexes:
                return
            
            print(f"Building BM25 index for {collection}...")
            from retrievers.bm25_retriever import BM25Retriever
            
            # Fetch chunks in batches (Qdrant scroll limit is typically 10K-100K)
            # For large collections, we need to scroll through all chunks
            id_to_text = {}
            offset = None
            batch_size = 10000  # Qdrant scroll limit
            
            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_offset = scroll_result
                
                # Process batch
                for point in points:
                    chunk_id = str(point.id)
                    text = point.payload.get("text", "")
                    if text:
                        id_to_text[chunk_id] = text
                
                # Check if we've fetched all chunks
                if next_offset is None or len(points) == 0:
                    break
                
                offset = next_offset
                
                # Progress indicator for large collections
                if len(id_to_text) % 50000 == 0:
                    print(f"  Fetched {len(id_to_text):,} chunks for BM25 index...")
            
            if not id_to_text:
                print(f"⚠ Warning: No chunks found in {collection} for BM25 index")
                return
            
            print(f"  Building BM25 index from {len(id_to_text):,} chunks...")
            self.bm25_indexes[collection] = BM25Retriever.from_id_to_text(id_to_text)
            print(f"✓ BM25 index built for {collection} ({len(id_to_text):,} chunks)")
    
    def retrieve(
        self,
        query: str,
        query_type: str,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks from appropriate collection with query-type-specific strategies.
        
        Strategy differences:
        - OLTP: Precision focus, smaller initial pool (k=20), faster reranker, hybrid enabled
        - OLAP: Recall focus, larger initial pool (k=50), better reranker, dense-only
        
        Args:
            query: User query
            query_type: "oltp" or "olap"
            top_k: Number of results to retrieve
            use_reranker: Whether to apply reranking
            reranker_model: Specific reranker model to use
            
        Returns:
            List of RetrievalResult objects
        """
        # Select collection based on query type
        collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
        
        # Query-type-specific retrieval strategies
        if query_type == "oltp":
            # OLTP: Precision focus, use hybrid retrieval if enabled
            if self.use_hybrid:
                return self._retrieve_hybrid(query, collection, top_k, use_reranker, reranker_model)
            else:
                return self._retrieve_dense_only(query, collection, query_type, top_k, use_reranker, reranker_model)
        else:
            # OLAP: Recall focus, dense-only (hybrid not typically used for OLAP)
            # Use larger initial retrieval pool for better recall
            return self._retrieve_dense_only(query, collection, query_type, top_k, use_reranker, reranker_model)
    
    def _retrieve_dense_only(
        self,
        query: str,
        collection: str,
        query_type: str,
        top_k: int,
        use_reranker: bool,
        reranker_model: Optional[str],
    ) -> List[RetrievalResult]:
        """Dense-only retrieval (baseline)."""
        from qdrant_client import models
        
        # Set HNSW search parameters based on query type
        # Higher ef_search for better recall (optimized for evaluation with large corpus)
        # Evaluation: ef=400 (OLTP), ef=600 (OLAP) for better recall on 813K chunks
        # Production: ef=200 (OLTP), ef=400 (OLAP) for balanced latency/recall
        search_ef = 400 if query_type == "oltp" else 600  # Optimized for evaluation
        
        # Embed query
        query_vector = self.embedder.encode(query, normalize_embeddings=True)
        
        # Search Qdrant with query-type-specific strategies
        # OLTP: Smaller pool (k=20) for precision, OLAP: Larger pool (k=50) for recall
        if query_type == "oltp":
            # OLTP: Precision focus - smaller initial pool
            initial_k = max(top_k * 2, 20) if use_reranker else top_k
        else:
            # OLAP: Recall focus - larger initial pool for better coverage
            initial_k = max(top_k * 5, 50) if use_reranker else top_k
        
        search_params = models.SearchParams(hnsw_ef=search_ef, exact=False)
        results = self.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=initial_k,
            search_params=search_params,
        )
        
        # Convert to RetrievalResult objects
        retrieved = []
        for hit in results.points:
            retrieved.append(RetrievalResult(
                chunk_id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=float(hit.score),
                chunk_type=hit.payload.get("chunk_type", query_type),
                source_file=hit.payload.get("source_file", "unknown"),
                section=hit.payload.get("section", ""),
                level=hit.payload.get("level", "leaf"),
            ))
        
        # Apply reranking if requested
        if use_reranker:
            retrieved = self._rerank(query, retrieved, query_type, reranker_model)
            retrieved = retrieved[:top_k]
        
        return retrieved[:top_k]
    
    def _retrieve_hybrid(
        self,
        query: str,
        collection: str,
        top_k: int,
        use_reranker: bool,
        reranker_model: Optional[str],
    ) -> List[RetrievalResult]:
        """Hybrid retrieval (BM25 + dense) for OLTP queries."""
        from qdrant_client import models
        
        # Build BM25 index if not already built
        if collection not in self.bm25_indexes:
            self._build_bm25_index(collection)
        
        if collection not in self.bm25_indexes:
            # Fallback to dense-only if BM25 index failed
            return self._retrieve_dense_only(query, collection, "oltp", top_k, use_reranker, reranker_model)
        
        # OPTIMIZATION: Parallelize BM25 and dense retrieval
        import threading
        
        bm25_results = None
        bm25_scores = None
        dense_results = None
        dense_error = None
        
        def run_bm25():
            """Run BM25 retrieval in parallel."""
            nonlocal bm25_results, bm25_scores
            try:
                # Expand query for BM25 (improves keyword matching)
                expanded_query = self._expand_query(query, use_llm=self.use_query_expansion)
                # Get BM25 results (retrieve more for RRF)
                # Optimized: Reduced candidate pool to minimize BM25 scoring overhead
                # Use smaller pool since BM25 now uses argpartition optimization
                bm25_results = self.bm25_indexes[collection].retrieve(expanded_query, top_k=max(top_k, 20))
                bm25_scores = {doc_id: score for doc_id, score in bm25_results}
            except Exception as e:
                print(f"Warning: BM25 retrieval failed: {e}")
                bm25_results = []
                bm25_scores = {}
        
        def run_dense():
            """Run dense retrieval in parallel."""
            nonlocal dense_results, dense_error
            try:
                # Get dense results
                query_vector = self.embedder.encode(query, normalize_embeddings=True)
                # Increased ef_search for better recall in hybrid retrieval (optimized for large corpus)
                search_params = models.SearchParams(hnsw_ef=400, exact=False)  # Optimized for evaluation
                # Increase candidate pool for reranking: retrieve 100 candidates for better reranking quality
                rerank_limit = max(top_k * 10, 100) if use_reranker else max(top_k * 5, 50)
                dense_results = self.qdrant_client.query_points(
                    collection_name=collection,
                    query=query_vector.tolist(),
                    limit=rerank_limit,
                    search_params=search_params,
                )
            except Exception as e:
                dense_error = e
        
        # Run BM25 and dense retrieval in parallel
        thread_bm25 = threading.Thread(target=run_bm25)
        thread_dense = threading.Thread(target=run_dense)
        
        thread_bm25.start()
        thread_dense.start()
        
        thread_bm25.join()
        thread_dense.join()
        
        # Handle errors
        if dense_error:
            raise dense_error
        
        if bm25_results is None or bm25_scores is None:
            # Fallback to dense-only if BM25 failed
            bm25_results = []
            bm25_scores = {}
        
        # Get chunk metadata for scoring
        chunk_metadata = {}
        dense_ranked_ids = []
        for hit in dense_results.points:
            chunk_id = str(hit.id)
            dense_ranked_ids.append(chunk_id)
            chunk_metadata[chunk_id] = {
                "text": hit.payload.get("text", ""),
                "chunk_type": hit.payload.get("chunk_type", "oltp"),
                "source_file": hit.payload.get("source_file", "unknown"),
                "section": hit.payload.get("section", ""),
                "level": hit.payload.get("level", "leaf"),
            }
        
        # Use Reciprocal Rank Fusion (RRF) instead of weighted combination
        # RRF is more robust and avoids score normalization issues
        rrf_results = self._reciprocal_rank_fusion(
            bm25_ranks=bm25_results,
            dense_ranks=dense_ranked_ids,
            k=60,  # RRF constant (standard value)
        )
        
        # Fetch metadata for BM25-only results (not in dense results)
        # OPTIMIZATION: Batch retrieve instead of individual calls
        all_chunk_ids = set(dense_ranked_ids) | set(bm25_scores.keys())
        missing_chunk_ids = [chunk_id for chunk_id in all_chunk_ids if chunk_id not in chunk_metadata]
        
        if missing_chunk_ids:
            # Batch retrieve all missing chunks in one call (much faster than individual retrieves)
            try:
                # Convert string IDs to integers for Qdrant (Qdrant stores IDs as integers)
                def to_int_id(chunk_id_str):
                    """Convert chunk ID string to integer for Qdrant."""
                    try:
                        return int(chunk_id_str)
                    except (ValueError, TypeError):
                        # If it's not a valid integer, return as-is (might be UUID)
                        return chunk_id_str
                
                # Qdrant retrieve can handle multiple IDs at once
                batch_size = 100  # Qdrant may have limits, batch if needed
                for i in range(0, len(missing_chunk_ids), batch_size):
                    batch_ids_str = missing_chunk_ids[i:i + batch_size]
                    # Convert to integers for Qdrant API
                    batch_ids_int = [to_int_id(cid) for cid in batch_ids_str]
                    points = self.qdrant_client.retrieve(
                        collection_name=collection,
                        ids=batch_ids_int,
                        with_payload=True,
                    )
                    for point in points:
                        chunk_id = str(point.id)
                        chunk_metadata[chunk_id] = {
                            "text": point.payload.get("text", ""),
                            "chunk_type": point.payload.get("chunk_type", "oltp"),
                            "source_file": point.payload.get("source_file", "unknown"),
                            "section": point.payload.get("section", ""),
                            "level": point.payload.get("level", "leaf"),
                        }
            except Exception as e:
                # Fallback: if batch retrieve fails, try individual (shouldn't happen)
                print(f"Warning: Batch retrieve failed, falling back to individual: {e}")
                for chunk_id in missing_chunk_ids:
                    if chunk_id not in chunk_metadata:
                        try:
                            # Convert to integer for Qdrant
                            int_id = int(chunk_id) if chunk_id.isdigit() else chunk_id
                            point = self.qdrant_client.retrieve(
                                collection_name=collection,
                                ids=[int_id],
                                with_payload=True,
                            )[0]
                            chunk_metadata[chunk_id] = {
                                "text": point.payload.get("text", ""),
                                "chunk_type": point.payload.get("chunk_type", "oltp"),
                                "source_file": point.payload.get("source_file", "unknown"),
                                "section": point.payload.get("section", ""),
                                "level": point.payload.get("level", "leaf"),
                            }
                        except:
                            continue
        
        # Sort by RRF score (already sorted by _reciprocal_rank_fusion)
        sorted_results = rrf_results
        
        retrieved = []
        # Return more candidates for reranking (100 candidates for better reranking quality)
        rerank_limit = top_k * 10 if use_reranker else top_k
        for chunk_id, rrf_score in sorted_results[:rerank_limit]:
            metadata = chunk_metadata.get(chunk_id, {})
            retrieved.append(RetrievalResult(
                chunk_id=chunk_id,
                text=metadata.get("text", ""),
                score=rrf_score,  # RRF score
                chunk_type=metadata.get("chunk_type", "oltp"),
                source_file=metadata.get("source_file", "unknown"),
                section=metadata.get("section", ""),
                level=metadata.get("level", "leaf"),
            ))
        
        # Apply reranking if requested
        if use_reranker:
            retrieved = self._rerank(query, retrieved, "oltp", reranker_model)
            retrieved = retrieved[:top_k]
        
        return retrieved[:top_k]
    
    def _rerank(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        query_type: str,
        reranker_model: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Apply reranking to retrieved results."""
        try:
            from rerankers.cross_encoder_reranker import OLTPReranker, OLAPReranker
            
            # Convert to format expected by reranker
            documents = [
                {"text": r.text, "score": r.score, "metadata": {"id": r.chunk_id}}
                for r in retrieved
            ]
            
            if query_type == "oltp":
                # OLTP reranker: use_reranker=True enables TinyBERT, False means no reranking
                use_reranker_flag = bool(reranker_model) and str(reranker_model).lower() != "none"
                reranker_key = f"oltp_{'tinybert' if use_reranker_flag else 'none'}"
                if not hasattr(self, 'oltp_rerankers'):
                    self.oltp_rerankers = {}
                if reranker_key not in self.oltp_rerankers:
                    self.oltp_rerankers[reranker_key] = OLTPReranker(use_reranker=use_reranker_flag, device=self.device)
                reranked = self.oltp_rerankers[reranker_key].rerank(query, documents, top_k=len(documents))
            else:
                # Use provided reranker_model or default
                reranker_key = f"olap_{reranker_model or 'minilm'}"
                if not hasattr(self, 'olap_rerankers'):
                    self.olap_rerankers = {}
                if reranker_key not in self.olap_rerankers:
                    self.olap_rerankers[reranker_key] = OLAPReranker(model_name=reranker_model or "minilm")
                reranked = self.olap_rerankers[reranker_key].rerank(query, documents, top_k=len(documents))
            
            # Convert back to RetrievalResult
            reranked_results = []
            for r in reranked:
                # RerankResult is a dataclass, not a dict
                # Extract chunk_id from metadata
                chunk_id = r.metadata.get("id", "") if isinstance(r.metadata, dict) else ""
                
                # Find original result to preserve metadata
                original = next((ret for ret in retrieved if ret.chunk_id == chunk_id), None)
                if original:
                    reranked_results.append(RetrievalResult(
                        chunk_id=original.chunk_id,
                        text=original.text,
                        score=float(r.score),  # RerankResult.score is already a float
                        chunk_type=original.chunk_type,
                        source_file=original.source_file,
                        section=original.section,
                        level=original.level,
                    ))
            
            return reranked_results if reranked_results else retrieved
            
        except Exception as e:
            print(f"⚠ Warning: Reranking failed ({e}), returning original results")
            return retrieved
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
        use_hybrid: Optional[bool] = None,
    ) -> RAGResult:
        """
        Complete RAG pipeline: classify → retrieve → (optionally rerank).
        
        Args:
            query: User query
            top_k: Number of results to return
            use_reranker: Whether to apply reranking
            reranker_model: Specific reranker model
            use_hybrid: Override hybrid retrieval setting (None = use instance default)
            
        Returns:
            RAGResult with query type, retrieved chunks, and metadata
        """
        import time
        
        # Step 1: Classify query (or skip for minimal baseline)
        if self.skip_classifier:
            query_type = "oltp"  # Default type
            confidence = 1.0
            collection = self.default_collection
        else:
            query_type, confidence = self.classify_query(query)
            collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
        
        # Step 2: Retrieve from appropriate collection (with timing)
        # Temporarily override hybrid setting if provided
        original_use_hybrid = self.use_hybrid
        if use_hybrid is not None:
            self.use_hybrid = use_hybrid
        
        retrieval_start = time.time()
        retrieved = self.retrieve(
            query=query,
            query_type=query_type,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
        )
        retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        
        # Restore original setting
        self.use_hybrid = original_use_hybrid
        
        # Step 3: Generate answer if LLM is enabled
        answer = None
        answer_cost_usd = None
        answer_latency_ms = None
        if self.use_llm_answer and self.llm_generator:
            try:
                contexts = [chunk.text for chunk in retrieved]
                answer_result = self.llm_generator.generate(query, contexts, query_type)
                answer = answer_result.answer
                answer_cost_usd = answer_result.cost.total_cost_usd
                answer_latency_ms = answer_result.latency_ms
            except Exception as e:
                print(f"⚠️  Answer generation failed: {e}")
                # Continue without answer
        
        # Step 4: Build result (collection already set if skip_classifier)
        if not self.skip_classifier:
            collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
        
        return RAGResult(
            query=query,
            query_type=query_type,
            confidence=confidence,
            collection=collection,
            retrieved_chunks=retrieved,
            reranked=use_reranker,
            metadata={
                "num_results": len(retrieved),
                "search_ef": 50 if query_type == "oltp" else 200,
                "hybrid_retrieval": self.use_hybrid if use_hybrid is None else use_hybrid,
                "hybrid_alpha": self.hybrid_alpha if (self.use_hybrid or (use_hybrid is True)) else None,
                "classifier_type": self.classifier_type,
                "total_latency_ms": retrieval_latency_ms,  # Retrieval latency in milliseconds
            },
            answer=answer,
            answer_cost_usd=answer_cost_usd,
            answer_latency_ms=answer_latency_ms,
        )


def print_result(result: RAGResult):
    """Pretty print RAG result."""
    print("\n" + "=" * 80)
    print(f"Query: {result.query}")
    print(f"Classifier: {result.metadata.get('classifier_type', 'unknown').upper()}")
    print(f"Type: {result.query_type.upper()} (confidence: {result.confidence:.2f})")
    print(f"Collection: {result.collection}")
    print(f"Retrieval: {'Hybrid (BM25 + Dense)' if result.metadata.get('hybrid_retrieval') else 'Dense-only'}")
    if result.metadata.get('hybrid_alpha'):
        print(f"Hybrid α: {result.metadata['hybrid_alpha']:.2f}")
    print(f"Reranked: {'Yes' if result.reranked else 'No'}")
    print("=" * 80)
    
    if result.answer:
        print(f"\n💬 Generated Answer:")
        print(f"{result.answer}")
        if result.answer_cost_usd:
            print(f"   Cost: ${result.answer_cost_usd:.6f} | Latency: {result.answer_latency_ms:.1f}ms")
        print()
    
    print(f"\n📚 Retrieved {len(result.retrieved_chunks)} chunks:\n")
    
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        print(f"{i}. [Score: {chunk.score:.4f}] {chunk.source_file}")
        if chunk.section:
            print(f"   Section: {chunk.section}")
        print(f"   {chunk.text[:200]}...")
        print()


def run_interactive(
    qdrant_host: str,
    qdrant_port: int,
    use_reranker: bool,
    use_hybrid: bool,
    classifier_type: str,
    classifier_model: str,
    oltp_reranker: str,
    olap_reranker: str,
    use_query_expansion: bool = True,
):
    """Interactive query mode."""
    print("\n" + "=" * 80)
    print("Query-Aware RAG Pipeline - Interactive Mode")
    print("=" * 80)
    print(f"Classifier: {classifier_type.upper()}" + (f" ({classifier_model})" if classifier_model else ""))
    print(f"Hybrid Retrieval: {'Enabled' if use_hybrid else 'Disabled'}")
    print(f"Reranking: {'Enabled' if use_reranker else 'Disabled'}")
    if use_reranker:
        print(f"  OLTP reranker: {oltp_reranker}")
        print(f"  OLAP reranker: {olap_reranker}")
    print("\nEnter queries (Ctrl+C to exit):\n")
    
    pipeline = QueryAwareRAGPipeline(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        use_hybrid=use_hybrid,
        classifier_type=classifier_type,
        classifier_model=classifier_model,
        use_query_expansion=use_query_expansion,
    )
    
    while True:
        try:
            query = input("🔍 Query: ").strip()
            if not query:
                continue
            
            # Classify to determine reranker
            query_type, _ = pipeline.classify_query(query)
            if use_reranker:
                reranker_model = oltp_reranker if query_type == "oltp" else olap_reranker
                if reranker_model.lower() == "none":
                    actual_reranker = False
                    reranker_model = None
                else:
                    actual_reranker = True
            else:
                actual_reranker = False
                reranker_model = None
            
            result = pipeline.query(
                query,
                top_k=5,
                use_reranker=actual_reranker,
                reranker_model=reranker_model,
            )
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Query-Aware RAG Pipeline")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--use-reranker", action="store_true", help="Apply reranking")
    parser.add_argument("--oltp-reranker", type=str, default="tinybert", 
                       help="OLTP reranker model: 'none', 'tinybert', 'minilm' (default: tinybert)")
    parser.add_argument("--olap-reranker", type=str, default="minilm",
                       help="OLAP reranker model: 'minilm', 'bge-base', 'bge-large' (default: minilm)")
    parser.add_argument("--classifier-type", choices=["feature", "transformer"], default="feature",
                       help="Classifier type: 'feature' (feature-based) or 'transformer' (DistilBERT/BERT)")
    parser.add_argument("--classifier-model", type=str, default=None,
                       help="Classifier model path/name (for transformer: 'distilbert-base-uncased', 'bert-base-uncased', 'classifier2/distilbert-base-uncased')")
    parser.add_argument("--router-model", default="models/feature_router.pkl", help="Feature router model path (if classifier-type=feature)")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid retrieval (BM25 + dense) for OLTP queries")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, help="BM25 weight in hybrid retrieval (0-1, default 0.7)")
    parser.add_argument("--use-query-expansion", action="store_true", default=True, help="Enable LLM-based query expansion for BM25 (requires OPENAI_API_KEY, default: True)")
    parser.add_argument("--no-query-expansion", dest="use_query_expansion", action="store_false", help="Disable LLM-based query expansion")
    parser.add_argument("--use-llm-answer", action="store_true", help="Enable LLM answer generation (requires OPENAI_API_KEY)")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="OpenAI model for answer generation (default: gpt-4o-mini)")
    parser.add_argument("--llm-max-tokens", type=int, default=512, help="Max tokens for answer generation (default: 512)")
    parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive(
            args.qdrant_host,
            args.qdrant_port,
            args.use_reranker,
            args.use_hybrid,
            args.classifier_type,
            args.classifier_model,
            args.oltp_reranker,
            args.olap_reranker,
            args.use_query_expansion,
        )
        return
    
    if not args.query:
        parser.error("--query required (or use --interactive)")
    
    # Initialize pipeline
    pipeline = QueryAwareRAGPipeline(
        router_model_path=args.router_model,
        classifier_type=args.classifier_type,
        classifier_model=args.classifier_model,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        use_hybrid=args.use_hybrid,
        hybrid_alpha=args.hybrid_alpha,
        use_query_expansion=args.use_query_expansion,
        use_llm_answer=args.use_llm_answer,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
    )
    
    # Process query
    # First classify to determine query type
    query_type, _ = pipeline.classify_query(args.query)
    
    # Select reranker model based on query type
    if args.use_reranker:
        reranker_model = args.oltp_reranker if query_type == "oltp" else args.olap_reranker
        if reranker_model.lower() == "none":
            use_reranker = False
            reranker_model = None
        else:
            use_reranker = True
    else:
        use_reranker = False
        reranker_model = None
    
    result = pipeline.query(
        query=args.query,
        top_k=args.top_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )
    
    # Output
    if args.output_json:
        # Convert to JSON-serializable format
        output = {
            "query": result.query,
            "query_type": result.query_type,
            "confidence": result.confidence,
            "collection": result.collection,
            "reranked": result.reranked,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": c.score,
                    "source_file": c.source_file,
                    "section": c.section,
                }
                for c in result.retrieved_chunks
            ],
            "metadata": result.metadata,
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()

