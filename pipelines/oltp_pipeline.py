from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pipelines.llm_client import LLMClient
from pipelines.policy import PolicyDecision
from retrievers.hierarchical_retriever import HierarchicalRetriever, RetrievedChunk
from retrievers.hybrid_retriever import HybridRetriever


@dataclass
class OltpPipelineResult:
    answer: str
    contexts: List[RetrievedChunk]
    metadata: Dict[str, object]


class OltpPipeline:
    """
    Fast-path pipeline for factoid queries.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        chunk_lookup: Dict[str, str],
        *,
        hierarchical_retriever: HierarchicalRetriever | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.chunk_lookup = chunk_lookup
        self.hierarchical_retriever = hierarchical_retriever
        self.llm_client = llm_client or LLMClient()

    def run(self, query: str, policy: PolicyDecision) -> OltpPipelineResult:
        if policy.use_hierarchical and self.hierarchical_retriever:
            retrieval = self.hierarchical_retriever.retrieve(
                query,
                top_k_super=policy.top_k_super,
                top_k_fine=policy.top_k_fine,
                use_hierarchical=True,
            )
            contexts = retrieval["fine_chunks"]
        else:
            hybrid_results = self.hybrid_retriever.retrieve(
                query,
                top_k=policy.top_k_fine,
                alpha=policy.hybrid_alpha,
            )
            contexts = [
                RetrievedChunk(
                    chunk_id=result.doc_id,
                    score=result.hybrid_score,
                    text=self.chunk_lookup.get(result.doc_id, ""),
                    level="fine",
                )
                for result in hybrid_results
            ]

        answer = self.llm_client.generate(query, [chunk.text for chunk in contexts])
        metadata = {
            "num_contexts": len(contexts),
            "retrieval_mode": "hierarchical" if policy.use_hierarchical else "hybrid",
        }
        return OltpPipelineResult(answer=answer, contexts=contexts, metadata=metadata)


OLTP_PIPELINE_SINGLETON: OltpPipeline | None = None


def configure_oltp_pipeline(pipeline: OltpPipeline) -> None:
    global OLTP_PIPELINE_SINGLETON
    OLTP_PIPELINE_SINGLETON = pipeline


def run_oltp_pipeline(query: str, policy: PolicyDecision | None = None) -> OltpPipelineResult:
    if OLTP_PIPELINE_SINGLETON is None:
        raise RuntimeError("OLTP pipeline has not been configured. Call configure_oltp_pipeline().")
    if policy is None:
        policy = PolicyDecision(use_hierarchical=False)
    return OLTP_PIPELINE_SINGLETON.run(query, policy)
