from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from main import initialize_system, rag_answer
from pipelines.policy import PolicyDecision


DEFAULT_QUERIES: List[str] = [
    "What year was Apple founded?",
    "Summarize AI investments by Apple and Microsoft.",
    "Where was Microsoft founded?",
]

EXPECTED_KEYWORDS: Dict[str, List[str]] = {
    "What year was Apple founded?": ["1976", "apple"],
    "Summarize AI investments by Apple and Microsoft.": ["apple", "microsoft", "ai"],
    "Where was Microsoft founded?": ["albuquerque", "microsoft"],
}

_SYSTEM_READY = False


def ensure_system_ready(demo_mode: bool = True) -> None:
    global _SYSTEM_READY
    if not _SYSTEM_READY:
        initialize_system(demo_mode=demo_mode)
        _SYSTEM_READY = True


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)


def measure_queries(
    queries: Sequence[str],
    repetitions: int = 1,
    *,
    policy_override: PolicyDecision | None = None,
    force_query_type: str | None = None,
) -> Dict[str, object]:
    ensure_system_ready()
    latencies: List[float] = []
    detailed: List[Dict[str, object]] = []
    for query in queries:
        for _ in range(repetitions):
            start = time.perf_counter()
            result = rag_answer(
                query,
                policy_override=policy_override,
                force_query_type=force_query_type,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            detailed.append({"query": query, "latency_ms": elapsed_ms, "result": result})
    return {"latencies_ms": latencies, "results": detailed}


def summarize_latencies(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "avg": mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
    }


def estimate_throughput(latencies_ms: Sequence[float]) -> float:
    if not latencies_ms:
        return 0.0
    total_time = sum(latencies_ms) / 1000.0
    if total_time == 0:
        return 0.0
    return len(latencies_ms) / total_time


def evaluate_quality(result_items: Sequence[Dict[str, object]]) -> float:
    if not result_items:
        return 0.0
    hits = 0
    total = 0
    for item in result_items:
        query = item["query"]
        result = item["result"]
        keywords = EXPECTED_KEYWORDS.get(query)
        if not keywords:
            continue
        answer = result["answer"].lower()
        total += 1
        if all(keyword in answer for keyword in keywords):
            hits += 1
    if total == 0:
        return 0.0
    return hits / total


def estimate_cost(result_items: Sequence[Dict[str, object]]) -> float:
    if not result_items:
        return 0.0
    costs: List[float] = []
    for item in result_items:
        metadata = item["result"].get("metadata", {})
        if "num_contexts" in metadata:
            costs.append(float(metadata["num_contexts"]))
        elif "community_summary" in metadata:
            costs.append(float(len(metadata["community_summary"])))
        else:
            costs.append(0.0)
    return mean(costs)


def save_measurements(path: Path | str, data: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

