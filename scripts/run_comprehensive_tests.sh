#!/bin/bash
# Comprehensive Evaluation Tests
# Based on ID_VS_TEXT_SIMILARITY_COMPARISON.md and COMPREHENSIVE_EVAL_IMPROVED_ANALYSIS.md

QDRANT_HOST="ecetesla0"
QDRANT_PORT=6333
DEVICE="cpu"

echo "=========================================="
echo "Running Comprehensive Evaluation Tests"
echo "=========================================="
echo ""

# Test 1: ID-Based Evaluation (Exact Matching)
# This matches COMPREHENSIVE_EVAL_IMPROVED_ANALYSIS.md
echo "Test 1: ID-Based Evaluation (Exact Matching)"
echo "--------------------------------------------"
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host $QDRANT_HOST \
    --qdrant-port $QDRANT_PORT \
    --device $DEVICE \
    --use-ground-truth \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_id_based.json

echo ""
echo "✓ ID-based evaluation complete"
echo ""

# Test 2: Text Similarity - Jaccard (Word-Level)
echo "Test 2: Text Similarity - Jaccard (Word-Level)"
echo "--------------------------------------------"
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host $QDRANT_HOST \
    --qdrant-port $QDRANT_PORT \
    --device $DEVICE \
    --use-ground-truth \
    --use-text-first \
    --similarity-method jaccard \
    --similarity-threshold 0.15 \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_jaccard.json

echo ""
echo "✓ Jaccard evaluation complete"
echo ""

# Test 3: Text Similarity - N-gram (Character 3-grams)
echo "Test 3: Text Similarity - N-gram (Character 3-grams)"
echo "--------------------------------------------"
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host $QDRANT_HOST \
    --qdrant-port $QDRANT_PORT \
    --device $DEVICE \
    --use-ground-truth \
    --use-text-first \
    --similarity-method ngram \
    --similarity-threshold 0.15 \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_ngram.json

echo ""
echo "✓ N-gram evaluation complete"
echo ""

# Test 4: Text Similarity - Hybrid (Combined)
echo "Test 4: Text Similarity - Hybrid (Combined)"
echo "--------------------------------------------"
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host $QDRANT_HOST \
    --qdrant-port $QDRANT_PORT \
    --device $DEVICE \
    --use-ground-truth \
    --use-text-first \
    --similarity-method hybrid \
    --similarity-threshold 0.15 \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_hybrid_similarity.json

echo ""
echo "✓ Hybrid similarity evaluation complete"
echo ""

echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/comprehensive_eval_id_based.json"
echo "  - results/comprehensive_eval_jaccard.json"
echo "  - results/comprehensive_eval_ngram.json"
echo "  - results/comprehensive_eval_hybrid_similarity.json"
echo ""


