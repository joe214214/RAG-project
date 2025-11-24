from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# Ensure project root is on sys.path so imports work when running
# this file directly: `python benchmarks/ablation.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.measurement_utils import (
    DEFAULT_QUERIES,
    evaluate_quality,
    estimate_cost,
    measure_queries,
    summarize_latencies,
)
from pipelines.policy import PolicyDecision


def _build_policy(hierarchical: bool, graph: bool, *, oracle: bool = False) -> PolicyDecision:
    """
    Construct a PolicyDecision with the requested knobs.

    For the oracle variant we use slightly larger retrieval depths and
    context size to approximate an upper bound on quality.
    """
    policy = PolicyDecision(
        use_hierarchical=hierarchical,
        activate_graph=graph,
    )
    if oracle:
        policy.top_k_super = 8
        policy.top_k_fine = 16
        policy.max_context_tokens = 2048
    return policy


def run_ablation(output_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    """
    Run real end-to-end experiments for different routing / hierarchy / graph
    settings and plot quality and cost based on actual RAG outputs.
    """
    scenarios = [
        {
            "name": "baseline",
            "routing": False,
            "hierarchical": False,
            "graph": False,
        },
        {
            "name": "routing_only",
            "routing": True,
            "hierarchical": False,
            "graph": False,
        },
        {
            "name": "routing_hierarchical",
            "routing": True,
            "hierarchical": True,
            "graph": False,
        },
        {
            "name": "full_system",
            "routing": True,
            "hierarchical": True,
            "graph": True,
        },
        {
            "name": "oracle",
            "routing": True,
            "hierarchical": True,
            "graph": True,
            "oracle": True,
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_results: List[Dict[str, object]] = []
    for scenario in scenarios:
        routing = bool(scenario["routing"])
        hierarchical = bool(scenario["hierarchical"])
        graph = bool(scenario["graph"])
        is_oracle = bool(scenario.get("oracle", False))

        # When routing is disabled, force OLTP so the classifier is bypassed.
        force_query_type = None if routing else "oltp"
        policy = _build_policy(hierarchical, graph, oracle=is_oracle)

        measurements = measure_queries(
            DEFAULT_QUERIES,
            repetitions=3,
            policy_override=policy,
            force_query_type=force_query_type,
        )

        quality = evaluate_quality(measurements["results"])
        cost = estimate_cost(measurements["results"])
        latency_summary = summarize_latencies(measurements["latencies_ms"])

        scenario_results.append(
            {
                "name": scenario["name"],
                "routing": routing,
                "hierarchical": hierarchical,
                "graph": graph,
                "oracle": is_oracle,
                "quality": quality,
                "cost": cost,
                "latency": latency_summary,
            }
        )

    # Persist full numeric results for further analysis.
    (output_dir / "ablation.json").write_text(
        json.dumps({"scenarios": scenario_results}, indent=2),
        encoding="utf-8",
    )

    # Routing ablation bar chart (quality vs. scenario)
    plt.figure(figsize=(6, 4))
    names = [r["name"] for r in scenario_results]
    qualities = [float(r["quality"]) for r in scenario_results]
    plt.bar(names, qualities, color="steelblue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Quality (keyword hit rate)")
    plt.title("Routing / Hierarchical / Graph Ablation (Real Measurements)")
    plt.tight_layout()
    plt.savefig(output_dir / "routing_ablation.png")
    plt.close()

    # Hierarchical vs flat comparison using measured qualities.
    flat_quality = next(
        r["quality"] for r in scenario_results if r["name"] == "routing_only"
    )
    hierarchical_quality = next(
        r["quality"] for r in scenario_results if r["name"] == "routing_hierarchical"
    )

    plt.figure(figsize=(5, 4))
    modes = ["flat", "hierarchical"]
    values = [float(flat_quality), float(hierarchical_quality)]
    plt.bar(modes, values, color=["tomato", "seagreen"])
    plt.ylabel("Answer accuracy")
    plt.title("Hierarchical vs. Flat Retrieval (Real Measurements)")
    plt.tight_layout()
    plt.savefig(output_dir / "hierarchical_vs_flat.png")
    plt.close()

    return {"scenarios": scenario_results}


if __name__ == "__main__":
    run_ablation(Path("results"))
