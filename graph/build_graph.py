from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import spacy


@dataclass(frozen=True)
class Entity:
    text: str
    label: str


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - requires external download
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install via: python -m spacy download {model_name}"
        ) from exc


def extract_entities(text: str, nlp=None) -> List[Entity]:
    nlp = nlp or load_spacy_model()
    doc = nlp(text)
    return [Entity(ent.text, ent.label_) for ent in doc.ents]


def build_entity_graph(texts: Iterable[str], model_name: str = "en_core_web_sm") -> nx.Graph:
    """
    Build an undirected graph where nodes are entities and edges denote
    co-occurrence within the same chunk.
    """
    nlp = load_spacy_model(model_name)
    graph = nx.Graph()
    entity_counter: Counter[Entity] = Counter()
    co_occurrence: Dict[Tuple[Entity, Entity], int] = defaultdict(int)

    for text in texts:
        entities = extract_entities(text, nlp=nlp)
        entity_counter.update(entities)
        for i, entity_i in enumerate(entities):
            for entity_j in entities[i + 1 :]:
                pair = tuple(sorted((entity_i, entity_j), key=lambda e: e.text))
                co_occurrence[pair] += 1

    for entity, freq in entity_counter.items():
        graph.add_node(entity.text, label=entity.label, frequency=freq)

    for (entity_i, entity_j), weight in co_occurrence.items():
        graph.add_edge(entity_i.text, entity_j.text, weight=weight)

    return graph

