from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def simulate_latency_distribution(qps: int) -> Dict[str, float]:
    base = 150
    multiplier = 1 + (qps / 100) * 1.5
    p50 = base * multiplier
    p95 = p50 * 1.5
    p99 = p50 * 2.1
    error_rate = min(0.3, 0.01 * (qps / 5))
    return {"p50": p50, "p95": p95, "p99": p99, "error_rate": error_rate}


def run_experiment(output_dir: Path) -> List[Dict[str, float]]:
    qps_levels = [1, 5, 10, 20, 50, 100]
    measurements: List[Dict[str, float]] = []
    for qps in tqdm(qps_levels, desc="QPS scaling"):
        measurement = {"qps": qps, **simulate_latency_distribution(qps)}
        measurements.append(measurement)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qps_scale.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    plt.plot([m["qps"] for m in measurements], [m["p50"] for m in measurements], label="P50")
    plt.plot([m["qps"] for m in measurements], [m["p95"] for m in measurements], label="P95")
    plt.plot([m["qps"] for m in measurements], [m["p99"] for m in measurements], label="P99")
    plt.xlabel("Queries per second")
    plt.ylabel("Latency (ms)")
    plt.title("Tail latency vs QPS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "qps_tail_latency.png")
    plt.close()
    return measurements


if __name__ == "__main__":
    run_experiment(Path("results"))

