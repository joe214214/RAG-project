"""Test Qdrant HNSW functionality."""
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, PointStruct
import numpy as np

client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

# Delete if exists, then create
if client.collection_exists("test_oltp"):
    client.delete_collection("test_oltp")

client.create_collection(
    collection_name="test_oltp",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
)
print("Created OLTP collection")

# Insert test vectors
vectors = np.random.rand(100, 384).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

points = [PointStruct(id=i, vector=v.tolist(), payload={"type": "test"}) for i, v in enumerate(vectors)]
client.upsert(collection_name="test_oltp", points=points)
print("Inserted 100 vectors")

# Search using query_points (new API)
query = np.random.rand(384).astype(np.float32)
query = query / np.linalg.norm(query)
results = client.query_points(collection_name="test_oltp", query=query.tolist(), limit=5)
print(f"Search returned {len(results.points)} results")
for r in results.points[:3]:
    print(f"  id={r.id}, score={r.score:.4f}")

print("\nQdrant HNSW test PASSED!")
