#!/usr/bin/env python3
"""
Feature-Based Query Router for OLTP/OLAP Classification

This script implements a linguistically-grounded query complexity classifier
that avoids the dataset bias issues of naive neural approaches.

Features:
1. LLM-assisted labeling (GPT-5.1/GPT-4) for silver labels
2. Linguistic feature extraction using spaCy
3. Logistic regression classifier with feature importance
4. Cross-validation and detailed evaluation

Usage:
    # Step 1: Create labeled dataset using LLM
    python scripts/feature_router.py --create-dataset --num-queries 500
    
    # Step 2: Train classifier on labeled data
    python scripts/feature_router.py --train --dataset data/labeled_queries.jsonl
    
    # Step 3: Evaluate and get feature importance
    python scripts/feature_router.py --evaluate --model models/feature_router.pkl
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from collections import Counter

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Will be imported lazily to avoid errors if not installed
spacy = None
nlp = None


def load_spacy():
    """Lazy load spaCy to avoid import errors."""
    global spacy, nlp
    if spacy is None:
        import spacy as sp
        spacy = sp
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class QueryFeatures:
    """Linguistic and structural features for query complexity classification."""
    # Basic
    word_count: int
    char_count: int
    
    # Syntactic
    dependency_depth: int
    clause_count: int
    noun_phrase_count: int
    verb_count: int
    
    # Semantic - OLAP indicators
    has_causal: bool      # "because", "why", "cause", "impact"
    has_comparative: bool  # "compare", "difference", "vs", "between"
    has_temporal: bool     # "over time", "trend", "changed", "history"
    has_aggregation: bool  # "summarize", "overview", "main", "key"
    has_explanation: bool  # "explain", "how does", "why does"
    
    # Semantic - OLTP indicators
    has_factoid_start: bool  # starts with who/what/when/where
    has_single_entity_lookup: bool  # "capital of", "population of", "CEO of"
    
    # Entity features
    entity_count: int
    entity_type_count: int
    has_person: bool
    has_org: bool
    has_gpe: bool  # Geo-political entity
    has_date: bool
    
    # Question type
    starts_with_why: bool
    starts_with_how: bool
    starts_with_what: bool
    starts_with_who: bool
    starts_with_when: bool
    starts_with_where: bool
    
    # Complexity indicators
    has_conjunction: bool  # "and", "or", "but"
    has_multiple_clauses: bool
    question_mark_count: int


def extract_features(query: str) -> QueryFeatures:
    """Extract linguistic features from a query using spaCy."""
    nlp = load_spacy()
    doc = nlp(query)
    query_lower = query.lower().strip()
    
    # Basic features
    word_count = len([t for t in doc if not t.is_punct])
    char_count = len(query)
    
    # Dependency depth (max depth in parse tree)
    def get_depth(token, depth=0):
        if token.head == token:
            return depth
        return get_depth(token.head, depth + 1)
    
    dependency_depth = max(get_depth(t) for t in doc) if doc else 0
    
    # Clause count (approximate by counting verbs with subjects)
    clause_count = len([t for t in doc if t.dep_ in ('ROOT', 'ccomp', 'advcl', 'relcl')])
    
    # Noun phrases and verbs
    noun_phrase_count = len(list(doc.noun_chunks))
    verb_count = len([t for t in doc if t.pos_ == 'VERB'])
    
    # Semantic patterns - OLAP indicators
    causal_patterns = ['because', 'why', 'cause', 'impact', 'effect', 'result', 'lead to', 'due to']
    comparative_patterns = ['compare', 'comparison', 'difference', 'different', 'vs', 'versus', 'between', 'contrast', 'similar']
    temporal_patterns = ['over time', 'trend', 'changed', 'history', 'evolution', 'decade', 'years', 'growth']
    aggregation_patterns = ['summarize', 'summary', 'overview', 'main', 'key', 'important', 'significant', 'major']
    explanation_patterns = ['explain', 'how does', 'why does', 'what makes', 'what causes', 'describe how']
    
    has_causal = any(p in query_lower for p in causal_patterns)
    has_comparative = any(p in query_lower for p in comparative_patterns)
    has_temporal = any(p in query_lower for p in temporal_patterns)
    has_aggregation = any(p in query_lower for p in aggregation_patterns)
    has_explanation = any(p in query_lower for p in explanation_patterns)
    
    # Semantic patterns - OLTP indicators
    factoid_starts = ['who is', 'who was', 'what is', 'what was', 'when did', 'when was', 'where is', 'where was']
    single_entity_patterns = ['capital of', 'population of', 'ceo of', 'founder of', 'president of', 'height of', 'birthday of', 'born in']
    
    has_factoid_start = any(query_lower.startswith(p) for p in factoid_starts)
    has_single_entity_lookup = any(p in query_lower for p in single_entity_patterns)
    
    # Entity features
    entities = list(doc.ents)
    entity_count = len(entities)
    entity_types = set(e.label_ for e in entities)
    entity_type_count = len(entity_types)
    
    has_person = 'PERSON' in entity_types
    has_org = 'ORG' in entity_types
    has_gpe = 'GPE' in entity_types
    has_date = 'DATE' in entity_types or 'TIME' in entity_types
    
    # Question type
    starts_with_why = query_lower.startswith('why')
    starts_with_how = query_lower.startswith('how')
    starts_with_what = query_lower.startswith('what')
    starts_with_who = query_lower.startswith('who')
    starts_with_when = query_lower.startswith('when')
    starts_with_where = query_lower.startswith('where')
    
    # Complexity indicators
    has_conjunction = any(t.pos_ == 'CCONJ' for t in doc)
    has_multiple_clauses = clause_count > 1
    question_mark_count = query.count('?')
    
    return QueryFeatures(
        word_count=word_count,
        char_count=char_count,
        dependency_depth=dependency_depth,
        clause_count=clause_count,
        noun_phrase_count=noun_phrase_count,
        verb_count=verb_count,
        has_causal=has_causal,
        has_comparative=has_comparative,
        has_temporal=has_temporal,
        has_aggregation=has_aggregation,
        has_explanation=has_explanation,
        has_factoid_start=has_factoid_start,
        has_single_entity_lookup=has_single_entity_lookup,
        entity_count=entity_count,
        entity_type_count=entity_type_count,
        has_person=has_person,
        has_org=has_org,
        has_gpe=has_gpe,
        has_date=has_date,
        starts_with_why=starts_with_why,
        starts_with_how=starts_with_how,
        starts_with_what=starts_with_what,
        starts_with_who=starts_with_who,
        starts_with_when=starts_with_when,
        starts_with_where=starts_with_where,
        has_conjunction=has_conjunction,
        has_multiple_clauses=has_multiple_clauses,
        question_mark_count=question_mark_count,
    )


def features_to_vector(features: QueryFeatures) -> np.ndarray:
    """Convert QueryFeatures to numpy array for classification."""
    return np.array([
        features.word_count,
        features.char_count,
        features.dependency_depth,
        features.clause_count,
        features.noun_phrase_count,
        features.verb_count,
        int(features.has_causal),
        int(features.has_comparative),
        int(features.has_temporal),
        int(features.has_aggregation),
        int(features.has_explanation),
        int(features.has_factoid_start),
        int(features.has_single_entity_lookup),
        features.entity_count,
        features.entity_type_count,
        int(features.has_person),
        int(features.has_org),
        int(features.has_gpe),
        int(features.has_date),
        int(features.starts_with_why),
        int(features.starts_with_how),
        int(features.starts_with_what),
        int(features.starts_with_who),
        int(features.starts_with_when),
        int(features.starts_with_where),
        int(features.has_conjunction),
        int(features.has_multiple_clauses),
        features.question_mark_count,
    ], dtype=np.float32)


FEATURE_NAMES = [
    'word_count', 'char_count', 'dependency_depth', 'clause_count',
    'noun_phrase_count', 'verb_count', 'has_causal', 'has_comparative',
    'has_temporal', 'has_aggregation', 'has_explanation', 'has_factoid_start',
    'has_single_entity_lookup', 'entity_count', 'entity_type_count',
    'has_person', 'has_org', 'has_gpe', 'has_date', 'starts_with_why',
    'starts_with_how', 'starts_with_what', 'starts_with_who', 'starts_with_when',
    'starts_with_where', 'has_conjunction', 'has_multiple_clauses', 'question_mark_count'
]


# =============================================================================
# LLM-Assisted Labeling
# =============================================================================

LABELING_PROMPT = """You are classifying queries for a RAG (Retrieval-Augmented Generation) system.

Classify the following query as either OLTP or OLAP:

**OLTP (Online Transaction Processing)** - Simple factoid queries:
- Single fact lookup (who, what, when, where)
- One entity, one relation
- Direct answer exists in a single document
- Examples: "What year was Apple founded?", "Who is the CEO of Tesla?", "Capital of France?"

**OLAP (Online Analytical Processing)** - Complex analytical queries:
- Requires synthesis from multiple sources
- Comparison, aggregation, or explanation
- Multi-hop reasoning needed
- Causal or temporal analysis
- Examples: "Compare Apple and Microsoft's AI strategies", "Why did the 2008 crisis happen?", "What factors contribute to climate change?"

**Query:** {query}

**Classification (respond with only "OLTP" or "OLAP"):**"""


def get_openai_model() -> str:
    """Get the OpenAI model to use, with fallback."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return model


def label_with_openai(queries: List[str], api_key: str) -> List[str]:
    """Label queries using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    model = get_openai_model()
    print(f"  Using model: {model}")
    
    labels = []
    for i, query in enumerate(queries):
        if (i + 1) % 50 == 0:
            print(f"  Labeled {i + 1}/{len(queries)} queries...")
        
        try:
            # GPT-5 models use max_completion_tokens and don't support temperature
            # Older models use max_tokens and support temperature
            is_gpt5 = model.startswith("gpt-5")
            
            if is_gpt5:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a query classifier. Respond with only 'OLTP' or 'OLAP'."},
                        {"role": "user", "content": LABELING_PROMPT.format(query=query)}
                    ],
                    max_completion_tokens=10,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a query classifier. Respond with only 'OLTP' or 'OLAP'."},
                        {"role": "user", "content": LABELING_PROMPT.format(query=query)}
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
            label = response.choices[0].message.content.strip().upper()
            labels.append("oltp" if "OLTP" in label else "olap")
        except Exception as e:
            error_str = str(e)
            # If model not found, try fallback
            if "model" in error_str.lower() and "not found" in error_str.lower():
                print(f"  Model {model} not available, trying gpt-4o-mini...")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a query classifier. Respond with only 'OLTP' or 'OLAP'."},
                            {"role": "user", "content": LABELING_PROMPT.format(query=query)}
                        ],
                        max_tokens=10,  # gpt-4o-mini uses max_tokens
                        temperature=0.0,
                    )
                    label = response.choices[0].message.content.strip().upper()
                    labels.append("oltp" if "OLTP" in label else "olap")
                    continue
                except Exception as e2:
                    print(f"  Fallback also failed: {e2}")
            print(f"  Error labeling query {i}: {e}")
            labels.append("oltp")  # Default fallback
    
    return labels


def label_with_anthropic(queries: List[str], api_key: str) -> List[str]:
    """Label queries using Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    labels = []
    for i, query in enumerate(queries):
        if (i + 1) % 50 == 0:
            print(f"  Labeled {i + 1}/{len(queries)} queries...")
        
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Cost-effective
                max_tokens=10,
                messages=[
                    {"role": "user", "content": LABELING_PROMPT.format(query=query)}
                ],
            )
            label = response.content[0].text.strip().upper()
            labels.append("oltp" if "OLTP" in label else "olap")
        except Exception as e:
            print(f"  Error labeling query {i}: {e}")
            labels.append("oltp")
    
    return labels


# =============================================================================
# Dataset Creation
# =============================================================================

def load_source_queries(num_queries: int) -> List[Tuple[str, str]]:
    """Load queries from multiple sources for diverse training data."""
    from datasets import load_dataset
    
    queries_with_source = []
    per_source = num_queries // 3  # Split between 3 reliable sources
    
    # MS MARCO (factoid - but we'll let LLM decide the label)
    print("Loading MS MARCO queries...")
    try:
        msmarco = load_dataset("microsoft/ms_marco", "v1.1", split=f"train[:{per_source}]")
        for item in msmarco:
            if "query" in item and item["query"]:
                queries_with_source.append((item["query"], "msmarco"))
        print(f"  Loaded {len([q for q in queries_with_source if q[1]=='msmarco'])} MS MARCO queries")
    except Exception as e:
        print(f"  Warning: Could not load MS MARCO: {e}")
    
    # HotpotQA (multi-hop - but we'll let LLM decide)
    print("Loading HotpotQA queries...")
    try:
        hotpot = load_dataset("hotpot_qa", "fullwiki", split=f"train[:{per_source}]")
        for item in hotpot:
            if "question" in item and item["question"]:
                queries_with_source.append((item["question"], "hotpotqa"))
        print(f"  Loaded {len([q for q in queries_with_source if q[1]=='hotpotqa'])} HotpotQA queries")
    except Exception as e:
        print(f"  Warning: Could not load HotpotQA: {e}")
    
    # Skip Natural Questions - it's 307GB and hangs
    # Instead use SQuAD which is much smaller
    print("Loading SQuAD queries...")
    try:
        squad = load_dataset("squad", split=f"train[:{per_source}]")
        for item in squad:
            if "question" in item and item["question"]:
                queries_with_source.append((item["question"], "squad"))
        print(f"  Loaded {len([q for q in queries_with_source if q[1]=='squad'])} SQuAD queries")
    except Exception as e:
        print(f"  Warning: Could not load SQuAD: {e}")
    
    # Add synthetic analytical queries to ensure OLAP representation
    print("Adding synthetic analytical queries...")
    synthetic_olap = [
        "Compare the market strategies of Apple and Microsoft over the past decade",
        "What factors contributed to the success of electric vehicles?",
        "Explain the relationship between inflation and unemployment",
        "Summarize the key trends in renewable energy adoption",
        "How has social media impacted political discourse?",
        "What are the pros and cons of remote work?",
        "Analyze the impact of AI on job markets",
        "Why do some startups succeed while others fail?",
        "Compare different approaches to climate change mitigation",
        "What lessons can be learned from the 2008 financial crisis?",
        "How do different countries approach healthcare policy?",
        "What are the main factors driving urbanization globally?",
        "Compare the education systems of Finland and the United States",
        "Why has cryptocurrency adoption varied across different regions?",
        "What environmental impacts result from fast fashion?",
        "What are the long-term effects of social media on mental health?",
        "How do economic policies affect income inequality?",
        "Compare renewable energy sources in terms of efficiency and cost",
        "What role does technology play in modern education?",
        "Analyze the relationship between diet and chronic diseases",
        "How has globalization impacted local economies?",
        "What factors determine a country's economic growth?",
        "Compare different approaches to urban planning",
        "Why do some companies thrive during recessions while others fail?",
        "What are the ethical implications of artificial intelligence?",
        "How does climate change affect biodiversity?",
        "Compare healthcare systems across developed nations",
        "What drives consumer behavior in the digital age?",
        "Analyze the impact of remote work on productivity",
        "How do political systems influence economic development?",
    ]
    
    # Ensure we have enough synthetic OLAP queries
    synthetic_needed = max(50, num_queries // 5)  # At least 50 or 20% of total
    for i in range(min(synthetic_needed, len(synthetic_olap) * 10)):
        queries_with_source.append((synthetic_olap[i % len(synthetic_olap)], "synthetic_olap"))
    
    return queries_with_source[:num_queries]


def create_labeled_dataset(
    num_queries: int = 500,
    output_path: str = "data/labeled_queries.jsonl",
    llm_provider: str = "openai",
    api_key: Optional[str] = None,
):
    """Create a labeled dataset using LLM-assisted labeling."""
    
    # Get API key from environment if not provided
    if api_key is None:
        if llm_provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(f"Please set {llm_provider.upper()}_API_KEY environment variable or pass --api-key")
    
    print(f"\n{'='*60}")
    print(f"Creating labeled dataset with {num_queries} queries")
    print(f"LLM Provider: {llm_provider}")
    print(f"{'='*60}\n")
    
    # Load queries from multiple sources
    queries_with_source = load_source_queries(num_queries)
    queries = [q[0] for q in queries_with_source]
    sources = [q[1] for q in queries_with_source]
    
    print(f"\nLoaded {len(queries)} queries from sources:")
    source_counts = Counter(sources)
    for src, count in source_counts.items():
        print(f"  {src}: {count}")
    
    # Label with LLM
    print(f"\nLabeling queries with {llm_provider}...")
    if llm_provider == "openai":
        labels = label_with_openai(queries, api_key)
    else:
        labels = label_with_anthropic(queries, api_key)
    
    # Extract features
    print("\nExtracting linguistic features...")
    records = []
    for i, (query, label, source) in enumerate(zip(queries, labels, sources)):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries...")
        
        features = extract_features(query)
        records.append({
            "query": query,
            "label": label,
            "source": source,
            "features": asdict(features),
        })
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    # Print statistics
    label_counts = Counter(labels)
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total queries: {len(records)}")
    print(f"OLTP: {label_counts.get('oltp', 0)} ({100*label_counts.get('oltp', 0)/len(records):.1f}%)")
    print(f"OLAP: {label_counts.get('olap', 0)} ({100*label_counts.get('olap', 0)/len(records):.1f}%)")
    print(f"\nSaved to: {output_path}")
    
    return records


# =============================================================================
# Training
# =============================================================================

def load_labeled_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load labeled dataset and extract feature vectors."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    X = []
    y = []
    queries = []
    
    for record in records:
        # Reconstruct features from dict
        features = QueryFeatures(**record["features"])
        X.append(features_to_vector(features))
        y.append(0 if record["label"] == "oltp" else 1)
        queries.append(record["query"])
    
    return np.array(X), np.array(y), queries


def train_classifier(
    dataset_path: str,
    output_path: str = "models/feature_router.pkl",
    test_size: float = 0.2,
):
    """Train a logistic regression classifier on the labeled dataset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\n{'='*60}")
    print("Training Feature-Based Query Router")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading dataset from {dataset_path}...")
    X, y, queries = load_labeled_dataset(dataset_path)
    print(f"Loaded {len(X)} samples")
    print(f"Class distribution: OLTP={sum(y==0)}, OLAP={sum(y==1)}")
    
    # Split data
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        X, y, queries, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Compensates for 76/24 imbalance
    )
    clf.fit(X_train_scaled, y_train)
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Test evaluation
    print("\nTest Set Evaluation:")
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['OLTP', 'OLAP']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Pred OLTP  Pred OLAP")
    print(f"True OLTP     {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"True OLAP     {cm[1][0]:>8}  {cm[1][1]:>8}")
    
    # Feature importance
    print("\nFeature Importance (Top 10):")
    importance = dict(zip(FEATURE_NAMES, clf.coef_[0]))
    sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_importance[:10]:
        direction = "-> OLAP" if coef > 0 else "-> OLTP"
        print(f"  {name:<25} {coef:>8.3f} {direction}")
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'importance': importance,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to: {output_path}")
    
    return clf, scaler


# =============================================================================
# Inference
# =============================================================================

class FeatureRouter:
    """Feature-based query router for OLTP/OLAP classification."""
    
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.importance = model_data['importance']
    
    def classify(self, query: str) -> str:
        """Classify a query as 'oltp' or 'olap'."""
        features = extract_features(query)
        X = features_to_vector(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        pred = self.classifier.predict(X_scaled)[0]
        return "oltp" if pred == 0 else "olap"
    
    def classify_with_confidence(self, query: str) -> Tuple[str, float]:
        """Classify with confidence score."""
        features = extract_features(query)
        X = features_to_vector(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X_scaled)[0]
        pred = "oltp" if proba[0] > proba[1] else "olap"
        confidence = max(proba)
        return pred, confidence
    
    def explain(self, query: str) -> Dict:
        """Explain classification decision."""
        features = extract_features(query)
        X = features_to_vector(features)
        
        # Find contributing features
        contributions = {}
        for i, name in enumerate(self.feature_names):
            coef = self.importance[name]
            value = X[i]
            contribution = coef * value
            if abs(contribution) > 0.1:  # Only significant contributions
                contributions[name] = {
                    'value': float(value),
                    'coefficient': coef,
                    'contribution': contribution,
                    'direction': 'OLAP' if contribution > 0 else 'OLTP'
                }
        
        label, confidence = self.classify_with_confidence(query)
        
        return {
            'query': query,
            'prediction': label,
            'confidence': confidence,
            'top_contributors': dict(sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]['contribution']), 
                reverse=True
            )[:5])
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature-Based Query Router")
    parser.add_argument("--create-dataset", action="store_true", help="Create labeled dataset")
    parser.add_argument("--train", action="store_true", help="Train classifier")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--classify", type=str, help="Classify a single query")
    
    parser.add_argument("--num-queries", type=int, default=500, help="Number of queries for dataset")
    parser.add_argument("--dataset", type=str, default="data/labeled_queries.jsonl")
    parser.add_argument("--model", type=str, default="models/feature_router.pkl")
    parser.add_argument("--llm", type=str, default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--api-key", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.create_dataset:
        create_labeled_dataset(
            num_queries=args.num_queries,
            output_path=args.dataset,
            llm_provider=args.llm,
            api_key=args.api_key,
        )
    
    elif args.train:
        train_classifier(
            dataset_path=args.dataset,
            output_path=args.model,
        )
    
    elif args.evaluate:
        print("Loading model and running evaluation...")
        router = FeatureRouter(args.model)
        
        test_queries = [
            ("What is the capital of France?", "oltp"),
            ("Who founded Microsoft?", "oltp"),
            ("When was Apple founded?", "oltp"),
            ("Compare Apple and Microsoft's AI strategies", "olap"),
            ("Why did the 2008 financial crisis happen?", "olap"),
            ("What factors contribute to climate change?", "olap"),
            ("What is inflation?", "olap"),  # Edge case
            ("Is Python faster than Java?", "oltp"),  # Edge case
        ]
        
        print("\nTest Predictions:")
        print("-" * 80)
        correct = 0
        for query, expected in test_queries:
            pred, conf = router.classify_with_confidence(query)
            status = "[OK]" if pred == expected else "[X]"
            correct += 1 if pred == expected else 0
            print(f"{status} [{pred.upper():4}] ({conf:.2f}) {query[:50]}...")
        
        print(f"\nAccuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.0f}%)")
        
        # Show explanation for one query
        print("\nExample Explanation:")
        explanation = router.explain("Compare the economic policies of US and China")
        print(json.dumps(explanation, indent=2))
    
    elif args.classify:
        router = FeatureRouter(args.model)
        explanation = router.explain(args.classify)
        print(json.dumps(explanation, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
