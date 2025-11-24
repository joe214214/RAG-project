from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from ml.classifier.classifier_inference import classify_query
from ml.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from pipelines.olap_pipeline import OlapPipeline, run_olap_pipeline, configure_olap_pipeline
from pipelines.oltp_pipeline import OltpPipeline, configure_oltp_pipeline, run_oltp_pipeline
from pipelines.policy import PolicyDecision, PolicyEngine
from pipelines.llm_client import LLMClient
from preprocess.chunking import ChunkMetadata, create_hierarchical_chunks
from preprocess.pdf_reader import extract_pdfs_to_text
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hierarchical_retriever import HierarchicalRetriever
from retrievers.hybrid_retriever import HybridRetriever
from vector_stores.finedb import FineChunkVectorStore, build_finedb_from_embeddings
from vector_stores.superdb import SuperChunkVectorStore, build_superdb_from_embeddings


CONFIG_PATH = Path("configs/base_config.yaml")
POLICY_ENGINE: PolicyEngine | None = None
EMBEDDER: SentenceTransformerEmbedder | None = None


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_demo_texts(text_dir: Path) -> None:
    """
    Populate `data/text` with demo documents if the directory is empty.
    """
    if any(text_dir.glob("*.txt")):
        return
    text_dir.mkdir(parents=True, exist_ok=True)
    demo_docs = {
        "apple": "Apple Inc. was founded in 1976 in Cupertino, California. The company focuses on consumer electronics.",
        "microsoft": "Microsoft Corporation was founded in 1975 in Albuquerque. It develops software and cloud infrastructure.",
        "ai_investments": (
            "In recent years, Big Tech companies have accelerated AI investments. "
            "Apple, Microsoft, and Google spend billions acquiring startups and training frontier models."
        ),
    }
    for name, text in demo_docs.items():
        (text_dir / f"{name}.txt").write_text(text, encoding="utf-8")


def load_chunk_metadata(metadata_path: Path) -> Dict[str, ChunkMetadata]:
    if not metadata_path.exists():
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {item["chunk_id"]: ChunkMetadata(**item) for item in data}


def load_parent_mapping(mapping_path: Path) -> Dict[str, List[str]]:
    if not mapping_path.exists():
        return {}
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def embed_chunks(
    metadata: Dict[str, ChunkMetadata],
    embedder: SentenceTransformerEmbedder,
    output_dir: Path,
) -> tuple[np.ndarray, List[str]]:
    ordered_items = sorted(metadata.items(), key=lambda kv: kv[0])
    texts = [Path(item.path).read_text(encoding="utf-8") for _, item in ordered_items]
    ids = [chunk_id for chunk_id, _ in ordered_items]
    embeddings = embedder.embed_texts(texts)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    return embeddings, ids


def build_vector_stores(
    config: dict,
    embedder: SentenceTransformerEmbedder,
    *,
    force_rebuild: bool = False,
) -> tuple[SuperChunkVectorStore, FineChunkVectorStore, Dict[str, ChunkMetadata], Dict[str, ChunkMetadata], Dict[str, List[str]]]:
    paths = config["paths"]
    text_dir = Path(paths["text_dir"])
    super_dir = Path(paths["super_chunk_dir"])
    fine_dir = Path(paths["fine_chunk_dir"])

    ensure_demo_texts(text_dir)
    if force_rebuild or not (super_dir / "metadata.json").exists():
        create_hierarchical_chunks(
            text_dir,
            super_dir,
            fine_dir,
        )

    super_metadata = load_chunk_metadata(super_dir / "metadata.json")
    fine_metadata = load_chunk_metadata(fine_dir / "metadata.json")
    parent_mapping = load_parent_mapping(super_dir / "parent_mapping.json")

    super_embeddings, super_ids = embed_chunks(super_metadata, embedder, Path(paths["super_embedding_dir"]))
    fine_embeddings, fine_ids = embed_chunks(fine_metadata, embedder, Path(paths["fine_embedding_dir"]))

    super_store = build_superdb_from_embeddings(super_embeddings, super_ids)
    fine_store = build_finedb_from_embeddings(fine_embeddings, fine_ids)
    return super_store, fine_store, super_metadata, fine_metadata, parent_mapping


def build_bm25_lookup(fine_metadata: Dict[str, ChunkMetadata]) -> BM25Retriever:
    id_to_text = {chunk_id: Path(meta.path).read_text(encoding="utf-8") for chunk_id, meta in fine_metadata.items()}
    return BM25Retriever.from_id_to_text(id_to_text)


def build_chunk_lookup(fine_metadata: Dict[str, ChunkMetadata]) -> Dict[str, str]:
    return {chunk_id: Path(meta.path).read_text(encoding="utf-8") for chunk_id, meta in fine_metadata.items()}


def initialize_system(config_path: Path = CONFIG_PATH, demo_mode: bool = False) -> None:
    global POLICY_ENGINE, EMBEDDER
    config = load_yaml_config(config_path)
    embed_config = EmbeddingConfig(
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"]["batch_size"],
    )
    EMBEDDER = SentenceTransformerEmbedder(embed_config)

    super_store, fine_store, super_metadata, fine_metadata, parent_mapping = build_vector_stores(
        config,
        EMBEDDER,
        force_rebuild=demo_mode,
    )

    bm25 = build_bm25_lookup(fine_metadata)
    chunk_lookup = build_chunk_lookup(fine_metadata)
    hierarchical_retriever = HierarchicalRetriever(
        super_store=super_store,
        fine_store=fine_store,
        embedder=EMBEDDER,
        super_chunk_paths={chunk_id: meta.path for chunk_id, meta in super_metadata.items()},
        fine_chunk_paths={chunk_id: meta.path for chunk_id, meta in fine_metadata.items()},
        super_to_fine=parent_mapping,
    )
    hybrid_retriever = HybridRetriever(
        bm25=bm25,
        vector_store=fine_store,
        embedder=EMBEDDER,
    )

    llm_client = LLMClient()
    oltp_pipeline = OltpPipeline(
        hybrid_retriever=hybrid_retriever,
        chunk_lookup=chunk_lookup,
        hierarchical_retriever=hierarchical_retriever,
        llm_client=llm_client,
    )
    olap_pipeline = OlapPipeline(
        hierarchical_retriever=hierarchical_retriever,
        llm_client=llm_client,
    )
    configure_oltp_pipeline(oltp_pipeline)
    configure_olap_pipeline(olap_pipeline)
    POLICY_ENGINE = PolicyEngine()


def rag_answer(
    query: str,
    system_load: float = 0.2,
    *,
    policy_override: PolicyDecision | None = None,
    force_query_type: str | None = None,
) -> dict:
    if POLICY_ENGINE is None:
        initialize_system()

    if force_query_type:
        query_type = force_query_type
    else:
        query_type = classify_query(query)

    if policy_override is not None:
        policy = policy_override
    else:
        policy = POLICY_ENGINE.decide_policy(query_type, system_load)

    if query_type == "oltp":
        result = run_oltp_pipeline(query, policy)
        metadata = result.metadata
    else:
        result = run_olap_pipeline(query, policy)
        metadata = {"community_summary": result.community_summary}

    return {
        "query": query,
        "query_type": query_type,
        "policy": asdict(policy),
        "answer": result.answer,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query-aware RAG system entrypoint.")
    parser.add_argument("--query", default="Summarize AI investments by Apple and Microsoft.")
    parser.add_argument("--system-load", type=float, default=0.3)
    parser.add_argument("--demo", action="store_true", help="Rebuild demo corpus before answering.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        initialize_system(demo_mode=True)
    response = rag_answer(args.query, system_load=args.system_load)
    print(json.dumps(response, indent=2))
