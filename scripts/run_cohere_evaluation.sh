#!/bin/bash
# Run Comprehensive Evaluation with Cohere API
# ID-Based (Exact Matching) Evaluation with Cohere embeddings and reranking

QDRANT_HOST="ecetesla0"
QDRANT_PORT=6333
DEVICE="cpu"
COHERE_API_KEY="${COHERE_API_KEY:-B7TpCvUuJGkxmpNMPtQl9Pc3lrmWzgYDd0O2x3iS}"

echo "=========================================="
echo "Running ID-Based Evaluation with Cohere API"
echo "=========================================="
echo ""
echo "Qdrant Host: $QDRANT_HOST"
echo "Cohere Embed Model: embed-english-light-v3.0 (384 dims)"
echo "Cohere Rerank Model: rerank-english-v3.0"
echo ""

python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host $QDRANT_HOST \
    --qdrant-port $QDRANT_PORT \
    --device $DEVICE \
    --use-ground-truth \
    --oltp-limit 50 \
    --olap-limit 50 \
    --use-cohere \
    --cohere-api-key "$COHERE_API_KEY" \
    --cohere-embed-model "embed-english-light-v3.0" \
    --cohere-rerank-model "rerank-english-v3.0" \
    --output results/comprehensive_eval_cohere_id_based.json

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: results/comprehensive_eval_cohere_id_based.json"
echo "=========================================="


