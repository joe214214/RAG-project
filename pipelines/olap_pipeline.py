from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from graph.build_graph import build_entity_graph, extract_entities
from graph.community_detection import CommunityResult, run_leiden
from graph.summarization import (
    summarize_community_level,
    summarize_global,
    summarize_paragraph_level,
)
from pipelines.llm_client import LLMClient
from pipelines.policy import PolicyDecision
from retrievers.hierarchical_retriever import HierarchicalRetriever, RetrievedChunk


@dataclass
class OlapPipelineResult:
    answer: str
    super_chunks: List[RetrievedChunk]
    fine_chunks: List[RetrievedChunk]
    community_summary: Dict[int, str]


class OlapPipeline:
    """
    Graph-based pipeline for complex OLAP queries.
    """

    def __init__(self, hierarchical_retriever: HierarchicalRetriever, llm_client: LLMClient | None = None) -> None:
        self.hierarchical_retriever = hierarchical_retriever
        self.llm_client = llm_client or LLMClient()
        self._nlp = None

    def _ensure_nlp(self):
        if self._nlp is None:
            from graph.build_graph import load_spacy_model

            self._nlp = load_spacy_model()
        return self._nlp

    def _assign_chunks_to_communities(
        self,
        fine_chunks: List[RetrievedChunk],
        communities: CommunityResult,
    ) -> Dict[int, List[str]]:
        community_to_chunks: Dict[int, List[str]] = defaultdict(list)
        for chunk in fine_chunks:
            entities = extract_entities(chunk.text, nlp=self._ensure_nlp())
            if not entities:
                community_to_chunks[0].append(chunk.text)
                continue
            entity_names = [entity.text for entity in entities]
            community_votes: Dict[int, int] = defaultdict(int)
            for name in entity_names:
                community_id = communities.node_to_community.get(name, 0)
                community_votes[community_id] += 1
            dominant = max(community_votes, key=community_votes.get)
            community_to_chunks[dominant].append(chunk.text)
        return community_to_chunks

    def run(self, query: str, policy: PolicyDecision) -> OlapPipelineResult:
        retrieval = self.hierarchical_retriever.retrieve(
            query,
            top_k_super=policy.top_k_super,
            top_k_fine=policy.top_k_fine,
            use_hierarchical=policy.use_hierarchical,
        )
        super_chunks = retrieval["super_chunks"]
        fine_chunks = retrieval["fine_chunks"]

        paragraph_summaries = summarize_paragraph_level([chunk.text for chunk in fine_chunks])
        graph = build_entity_graph(paragraph_summaries)
        community_result = run_leiden(graph) if policy.activate_graph else CommunityResult({}, 0)

        if policy.activate_graph and graph.number_of_nodes() > 0:
            community_to_chunks = self._assign_chunks_to_communities(fine_chunks, community_result)
            community_summaries = summarize_community_level(community_to_chunks)
        else:
            community_summaries = {0: "\n".join(paragraph_summaries)}

        global_summary = summarize_global(community_summaries)
        answer = self.llm_client.generate(query, [global_summary])

        return OlapPipelineResult(
            answer=answer,
            super_chunks=super_chunks,
            fine_chunks=fine_chunks,
            community_summary=community_summaries,
        )


OLAP_PIPELINE_SINGLETON: OlapPipeline | None = None


def configure_olap_pipeline(pipeline: OlapPipeline) -> None:
    global OLAP_PIPELINE_SINGLETON
    OLAP_PIPELINE_SINGLETON = pipeline


def run_olap_pipeline(query: str, policy: PolicyDecision | None = None) -> OlapPipelineResult:
    if OLAP_PIPELINE_SINGLETON is None:
        raise RuntimeError("OLAP pipeline has not been configured. Call configure_olap_pipeline().")
    if policy is None:
        policy = PolicyDecision(use_hierarchical=True, activate_graph=True)
    return OLAP_PIPELINE_SINGLETON.run(query, policy)
