# Cohere API Implementation Guide

This guide demonstrates how to integrate Cohere API for hybrid search and reranking into your existing RAG system, based on research from Context7 and Perplexity MCP.

## Overview

Cohere API provides three main components for RAG systems:
1. **Embed API**: Generate semantic embeddings for documents and queries
2. **Rerank API**: Intelligently rerank search results for better relevance
3. **Chat API**: Generate responses grounded in retrieved documents

## Installation

```bash
pip install cohere
```

## Setup

```python
import cohere
import os

# Initialize Cohere client
# Option 1: Direct API key
co = cohere.ClientV2(api_key="your-api-key-here")

# Option 2: Environment variable (recommended)
# Set COHERE_API_KEY environment variable
co = cohere.ClientV2()
```

## 1. Cohere Embed API Integration

### Generate Document Embeddings

```python
def generate_cohere_embeddings(texts: list[str], model: str = "embed-english-v3.0"):
    """
    Generate embeddings using Cohere Embed API
    
    Args:
        texts: List of document texts to embed
        model: Cohere embedding model (embed-english-v3.0, embed-multilingual-v3.0)
        
    Returns:
        List of embedding vectors
    """
    response = co.embed(
        model=model,
        texts=texts,
        input_type="search_document"  # Use "search_query" for queries
    )
    return response.embeddings

# Example: Embed documents for storage
documents = [
    "Qdrant is a vector database designed for semantic search",
    "Retrieval-augmented generation improves LLM accuracy",
    "BM25 is a probabilistic retrieval model for keyword search"
]

doc_embeddings = generate_cohere_embeddings(documents)
print(f"Generated {len(doc_embeddings)} embeddings")
print(f"Embedding dimension: {len(doc_embeddings[0])}")
```

### Generate Query Embeddings

```python
def generate_query_embedding(query: str, model: str = "embed-english-v3.0"):
    """
    Generate query embedding using Cohere Embed API
    
    Args:
        query: User query string
        model: Cohere embedding model
        
    Returns:
        Query embedding vector
    """
    response = co.embed(
        model=model,
        texts=[query],
        input_type="search_query"  # Important: use search_query for queries
    )
    return response.embeddings[0]

# Example
query = "How does vector database improve semantic search?"
query_embedding = generate_query_embedding(query)
print(f"Query embedding dimension: {len(query_embedding)}")
```

## 2. Cohere Rerank API Integration

### Basic Reranking

```python
def rerank_with_cohere(query: str, documents: list[str], 
                      top_n: int = 3, model: str = "rerank-english-v3.0"):
    """
    Rerank documents using Cohere Rerank API
    
    Args:
        query: User query
        documents: List of document texts to rerank
        top_n: Number of top results to return
        model: Cohere rerank model (rerank-english-v3.0, rerank-v3.5)
        
    Returns:
        List of reranked documents with relevance scores
    """
    response = co.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        max_tokens_per_doc=4096  # Maximum tokens per document
    )
    
    reranked_results = []
    for result in response.results:
        reranked_results.append({
            "index": result.index,
            "document": documents[result.index],
            "relevance_score": result.relevance_score
        })
    
    return reranked_results

# Example usage
query = "What is retrieval augmented generation?"
candidate_docs = [
    "RAG combines retrieval with generation",
    "Vector databases store embeddings",
    "RAG improves LLM accuracy by grounding responses in documents",
    "BM25 is a keyword search algorithm"
]

reranked = rerank_with_cohere(query, candidate_docs, top_n=2)
print("\nReranked Results:")
for i, result in enumerate(reranked, 1):
    print(f"{i}. [Score: {result['relevance_score']:.4f}] {result['document']}")
```

### Advanced Reranking with Metadata

```python
def rerank_with_metadata(query: str, documents: list[dict], 
                        top_n: int = 3):
    """
    Rerank documents with preserved metadata
    
    Args:
        query: User query
        documents: List of dicts with 'text' and metadata fields
        top_n: Number of top results
        
    Returns:
        Reranked documents with all original metadata
    """
    # Extract texts for reranking
    texts = [doc["text"] for doc in documents]
    
    # Rerank
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n
    )
    
    # Reconstruct results with metadata
    reranked_results = []
    for result in response.results:
        original_doc = documents[result.index]
        reranked_results.append({
            **original_doc,  # Preserve all metadata
            "relevance_score": result.relevance_score,
            "rerank_index": result.index
        })
    
    return reranked_results

# Example with metadata
documents_with_meta = [
    {"text": "RAG improves accuracy", "source": "paper1", "id": 1},
    {"text": "Vector databases are fast", "source": "paper2", "id": 2},
    {"text": "RAG combines retrieval and generation", "source": "paper1", "id": 3}
]

reranked = rerank_with_metadata("What is RAG?", documents_with_meta, top_n=2)
for result in reranked:
    print(f"ID: {result['id']}, Source: {result['source']}, Score: {result['relevance_score']:.4f}")
```

## 3. Integration with Your Existing RAG Pipeline

### Replace BM25 + Dense with Cohere Embeddings

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

def store_documents_with_cohere_embeddings(
    client: QdrantClient,
    collection_name: str,
    documents: list[dict],
    co: cohere.ClientV2
):
    """
    Store documents in Qdrant using Cohere embeddings
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        documents: List of dicts with 'id', 'text', and optional metadata
        co: Cohere client instance
    """
    # Extract texts for batch embedding
    texts = [doc["text"] for doc in documents]
    
    # Generate embeddings in batch
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = generate_cohere_embeddings(texts)
    
    # Get embedding dimension
    embedding_dim = len(embeddings[0])
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
    
    # Prepare points for insertion
    points = []
    for doc, embedding in zip(documents, embeddings):
        point = PointStruct(
            id=doc["id"],
            vector=embedding,
            payload={
                "text": doc["text"],
                **{k: v for k, v in doc.items() if k not in ["id", "text"]}
            }
        )
        points.append(point)
    
    # Batch upsert
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Stored {len(points)} documents in {collection_name}")

# Example usage
documents = [
    {"id": 1, "text": "Qdrant is a vector database", "source": "docs"},
    {"id": 2, "text": "RAG improves LLM accuracy", "source": "research"},
    {"id": 3, "text": "BM25 is a keyword search algorithm", "source": "docs"}
]

store_documents_with_cohere_embeddings(
    qdrant_client, "cohere_collection", documents, co
)
```

### Hybrid Search with Cohere Embeddings + BM25

```python
def hybrid_search_with_cohere(
    query: str,
    client: QdrantClient,
    co: cohere.ClientV2,
    collection_name: str,
    top_k: int = 10
):
    """
    Perform hybrid search using Cohere embeddings + BM25
    
    Args:
        query: User query
        client: Qdrant client
        co: Cohere client
        collection_name: Collection name
        top_k: Number of results
        
    Returns:
        Combined search results
    """
    # Step 1: Semantic search with Cohere embeddings
    query_embedding = generate_query_embedding(query)
    
    semantic_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k * 2,  # Get more for fusion
        with_payload=True
    )
    
    # Step 2: Keyword search with BM25 (if available)
    # Note: This requires BM25 index in Qdrant
    # For now, we'll use semantic search only
    
    # Step 3: Format results
    results = []
    for scored_point in semantic_results:
        results.append({
            "id": scored_point.id,
            "text": scored_point.payload.get("text"),
            "score": scored_point.score,
            "search_type": "semantic"
        })
    
    return results[:top_k]

# Example
results = hybrid_search_with_cohere(
    "What is vector database?",
    qdrant_client,
    co,
    "cohere_collection",
    top_k=5
)

for i, result in enumerate(results, 1):
    print(f"{i}. [Score: {result['score']:.4f}] {result['text']}")
```

### Complete RAG Pipeline with Cohere Reranking

```python
def complete_rag_with_cohere(
    query: str,
    client: QdrantClient,
    co: cohere.ClientV2,
    collection_name: str,
    retrieve_top_k: int = 10,
    rerank_top_n: int = 3
):
    """
    Complete RAG pipeline: Retrieve -> Rerank -> Generate
    
    Args:
        query: User query
        client: Qdrant client
        co: Cohere client
        collection_name: Collection name
        retrieve_top_k: Documents to retrieve before reranking
        rerank_top_n: Final documents after reranking
        
    Returns:
        Dictionary with response and sources
    """
    # Stage 1: Retrieve candidate documents
    print(f"[Stage 1] Retrieving {retrieve_top_k} candidates...")
    candidates = hybrid_search_with_cohere(
        query, client, co, collection_name, top_k=retrieve_top_k
    )
    
    candidate_texts = [r["text"] for r in candidates]
    
    # Stage 2: Rerank with Cohere
    print(f"[Stage 2] Reranking to top {rerank_top_n}...")
    reranked = rerank_with_cohere(query, candidate_texts, top_n=rerank_top_n)
    
    # Stage 3: Generate response (optional - requires Chat API)
    # For now, return reranked documents
    return {
        "query": query,
        "retrieved_documents": reranked,
        "candidates_count": len(candidates),
        "final_count": len(reranked)
    }

# Example usage
result = complete_rag_with_cohere(
    "How does RAG improve LLM accuracy?",
    qdrant_client,
    co,
    "cohere_collection",
    retrieve_top_k=10,
    rerank_top_n=3
)

print(f"\nQuery: {result['query']}")
print(f"\nTop {result['final_count']} Reranked Documents:")
for i, doc in enumerate(result['retrieved_documents'], 1):
    print(f"\n{i}. [Relevance: {doc['relevance_score']:.4f}]")
    print(f"   {doc['document']}")
```

## 4. Integration with Your Existing Code

### Modify `scripts/rag_pipeline.py`

Add Cohere integration as an alternative to your current BM25 + dense approach:

```python
# Add to imports
import cohere

class QueryAwareRAGPipeline:
    def __init__(self, ..., use_cohere: bool = False, cohere_api_key: str = None):
        # ... existing initialization ...
        
        if use_cohere:
            self.cohere_client = cohere.ClientV2(api_key=cohere_api_key)
            self.use_cohere = True
        else:
            self.use_cohere = False
    
    def _retrieve_with_cohere(self, query: str, collection: str, top_k: int):
        """
        Retrieve using Cohere embeddings + reranking
        """
        # Generate query embedding
        query_embedding = self._generate_cohere_query_embedding(query)
        
        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k * 2,  # Get more for reranking
            with_payload=True
        )
        
        # Extract texts for reranking
        candidate_texts = [r.payload.get("text") for r in results]
        
        # Rerank with Cohere
        reranked = self.cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=candidate_texts,
            top_n=top_k
        )
        
        # Format results
        final_results = []
        for result in reranked.results:
            original_result = results[result.index]
            final_results.append({
                "id": original_result.id,
                "text": candidate_texts[result.index],
                "score": result.relevance_score,
                "payload": original_result.payload
            })
        
        return final_results
    
    def _generate_cohere_query_embedding(self, query: str):
        """Generate query embedding using Cohere"""
        response = self.cohere_client.embed(
            model="embed-english-v3.0",
            texts=[query],
            input_type="search_query"
        )
        return response.embeddings[0]
```

## 5. Performance Comparison

### Expected Performance

Based on research findings:

| Metric | Your BM25+Dense | Cohere API (Estimate) |
|--------|----------------|----------------------|
| **Retrieval Latency** | 943ms | ~1,100-1,400ms |
| **Rerank Latency** | +146ms (local) | +100-300ms (API) |
| **Cost per Query** | $0.0002 | $0.001-0.01 |
| **Recall@10** | 0.273 (ID-based) | Unknown (needs testing) |
| **Control** | Full | Limited |

### When to Use Cohere API

**Use Cohere API when:**
- ✅ Quality is more important than latency
- ✅ You want managed service (no infrastructure)
- ✅ Small-scale application (<10K queries/day)
- ✅ You need multilingual support
- ✅ Development speed is priority

**Stick with your current system when:**
- ✅ Latency is critical (<100ms requirement)
- ✅ Cost optimization is important
- ✅ Large-scale application (>100K queries/day)
- ✅ You need full control over indexing
- ✅ You want to leverage your 1.87× speedup with distributed processing

## 6. Testing Cohere API Integration

### A/B Testing Script

```python
def compare_retrieval_methods(
    queries: list[str],
    client: QdrantClient,
    co: cohere.ClientV2,
    collection_name: str
):
    """
    Compare your current BM25+Dense vs Cohere API
    
    Args:
        queries: List of test queries
        client: Qdrant client
        co: Cohere client
        collection_name: Collection name
    """
    results = {
        "current_system": [],
        "cohere_api": []
    }
    
    for query in queries:
        # Your current system
        start = time.time()
        current_results = your_current_retrieve_method(query, client, collection_name)
        current_time = (time.time() - start) * 1000
        
        # Cohere API
        start = time.time()
        cohere_results = hybrid_search_with_cohere(query, client, co, collection_name)
        cohere_time = (time.time() - start) * 1000
        
        results["current_system"].append({
            "query": query,
            "latency_ms": current_time,
            "results_count": len(current_results)
        })
        
        results["cohere_api"].append({
            "query": query,
            "latency_ms": cohere_time,
            "results_count": len(cohere_results)
        })
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"Current System Avg Latency: {np.mean([r['latency_ms'] for r in results['current_system']]):.2f}ms")
    print(f"Cohere API Avg Latency: {np.mean([r['latency_ms'] for r in results['cohere_api']]):.2f}ms")
    
    return results

# Run comparison
test_queries = [
    "What is retrieval augmented generation?",
    "How does vector database improve search?",
    "What is BM25 algorithm?"
]

comparison = compare_retrieval_methods(
    test_queries,
    qdrant_client,
    co,
    "cohere_collection"
)
```

## 7. Cost Estimation

### Cohere API Pricing (Approximate)

- **Embed API**: ~$0.001 per 1K tokens
- **Rerank API**: ~$0.001-0.01 per 1K tokens
- **Chat API**: Varies by model

### Cost Calculation

```python
def estimate_cohere_cost(num_queries: int, avg_query_tokens: int = 20, 
                        avg_doc_tokens: int = 500, docs_per_query: int = 10):
    """
    Estimate Cohere API costs
    
    Args:
        num_queries: Number of queries per day
        avg_query_tokens: Average tokens per query
        avg_doc_tokens: Average tokens per document
        docs_per_query: Documents retrieved per query
    """
    # Embed API cost (query embedding)
    query_embed_cost = (num_queries * avg_query_tokens / 1000) * 0.001
    
    # Rerank API cost (reranking retrieved documents)
    rerank_cost = (num_queries * docs_per_query * avg_doc_tokens / 1000) * 0.001
    
    total_cost = query_embed_cost + rerank_cost
    
    print(f"\nCohere API Cost Estimation:")
    print(f"Queries per day: {num_queries}")
    print(f"Query embedding cost: ${query_embed_cost:.4f}/day")
    print(f"Reranking cost: ${rerank_cost:.4f}/day")
    print(f"Total cost: ${total_cost:.4f}/day (${total_cost * 30:.2f}/month)")
    
    return total_cost

# Example: 1,000 queries/day
estimate_cohere_cost(1000)
```

## 8. Recommendations

Based on your current system performance:

1. **Test Cohere API** on a subset of queries (100 queries) to measure:
   - Actual latency vs your 943ms
   - Quality improvement (Recall@10, MRR)
   - Cost per query

2. **If quality improves significantly** (>10% Recall@10 improvement):
   - Consider Cohere for quality-critical queries
   - Keep your current system for high-throughput scenarios

3. **If quality is similar**:
   - Stick with your current system (better cost, latency, control)

4. **Hybrid approach**:
   - Use Cohere API for complex OLAP queries
   - Use your current system for simple OLTP queries

## References

- [Cohere Python SDK](https://github.com/cohere-ai/cohere-python)
- [Cohere Rerank API Docs](https://docs.cohere.com/reference/rerank)
- [Cohere Embed API Docs](https://docs.cohere.com/reference/embed)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/search/)

