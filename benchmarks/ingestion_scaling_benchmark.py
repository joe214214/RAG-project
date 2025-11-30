#!/usr/bin/env python3
"""
Ingestion Scaling Benchmark

Measures MPI ingestion performance across different worker counts.
Run this script multiple times with different MPI configurations to compare scaling.

Usage:
    # 1 worker (baseline)
    python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000 --use-gpu --output-stats results/ingest_1w.json
    
    # 2 workers
    mpirun -np 2 python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000 --use-gpu --output-stats results/ingest_2w.json
    
    # 4 workers
    mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000 --use-gpu --output-stats results/ingest_4w.json
    
    # Then analyze results
    python benchmarks/ingestion_scaling_benchmark.py --results-dir results/
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict
import numpy as np

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_stats(results_dir: str) -> List[Dict]:
    """Load all ingestion stats JSON files."""
    results_path = Path(results_dir)
    stats_files = list(results_path.glob("ingest_*.json"))
    
    stats = []
    for f in stats_files:
        with open(f, "r") as file:
            data = json.load(file)
            stats.append(data)
    
    return sorted(stats, key=lambda x: x["workers"])


def analyze_scaling(stats: List[Dict]) -> Dict:
    """Analyze scaling efficiency."""
    if not stats:
        return {}
    
    baseline = stats[0]  # Assume first is 1 worker
    baseline_time = baseline["timings"]["total"]
    
    analysis = {
        "baseline": {
            "workers": baseline["workers"],
            "time": baseline_time,
            "throughput": baseline["throughput"]["chunks_per_sec"],
        },
        "scaling": [],
    }
    
    for s in stats[1:]:
        speedup = baseline_time / s["timings"]["total"]
        efficiency = speedup / s["workers"] * 100  # Percentage of ideal scaling
        analysis["scaling"].append({
            "workers": s["workers"],
            "time": s["timings"]["total"],
            "speedup": speedup,
            "efficiency": efficiency,
            "throughput": s["throughput"]["chunks_per_sec"],
        })
    
    return analysis


def print_report(analysis: Dict):
    """Print scaling analysis report."""
    print("\n" + "=" * 70)
    print("Ingestion Scaling Analysis")
    print("=" * 70)
    
    baseline = analysis["baseline"]
    print(f"\nBaseline (1 worker):")
    print(f"  Time: {baseline['time']:.2f}s")
    print(f"  Throughput: {baseline['throughput']:.1f} chunks/sec")
    
    print(f"\nScaling Results:")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Throughput':<15}")
    print("-" * 70)
    
    for s in analysis["scaling"]:
        print(f"{s['workers']:<10} {s['time']:<12.2f} {s['speedup']:<12.2f}x {s['efficiency']:<11.1f}% {s['throughput']:<15.1f}")
    
    # Calculate average efficiency
    if analysis["scaling"]:
        avg_efficiency = sum(s["efficiency"] for s in analysis["scaling"]) / len(analysis["scaling"])
        print(f"\nAverage Scaling Efficiency: {avg_efficiency:.1f}%")
        print(f"(100% = perfect linear scaling)")


def plot_scaling(stats: List[Dict], output_file: str = None):
    """Plot scaling performance."""
    workers = [s["workers"] for s in stats]
    times = [s["timings"]["total"] for s in stats]
    throughputs = [s["throughput"]["chunks_per_sec"] for s in stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Time vs Workers
    ax1.plot(workers, times, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Workers", fontsize=12)
    ax1.set_ylabel("Total Time (seconds)", fontsize=12)
    ax1.set_title("Ingestion Time vs Workers", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(workers)
    
    # Ideal scaling line (baseline time / workers)
    if len(workers) > 0:
        baseline_time = times[0]
        ideal_times = [baseline_time / w for w in workers]
        ax1.plot(workers, ideal_times, '--', color='gray', alpha=0.5, label='Ideal scaling')
        ax1.legend()
    
    # Plot 2: Throughput vs Workers
    ax2.plot(workers, throughputs, 's-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Workers", fontsize=12)
    ax2.set_ylabel("Throughput (chunks/sec)", fontsize=12)
    ax2.set_title("Throughput vs Workers", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(workers)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze ingestion scaling performance")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing ingest_*.json files")
    parser.add_argument("--plot", type=str, default=None, help="Save scaling plot to file (e.g., results/scaling.png)")
    parser.add_argument("--output", type=str, default=None, help="Save analysis to JSON")
    
    args = parser.parse_args()
    
    # Load stats
    stats = load_stats(args.results_dir)
    
    if not stats:
        print(f"No ingestion stats found in {args.results_dir}")
        print("Run ingestion with --output-stats to generate stats files first.")
        return
    
    print(f"Loaded {len(stats)} stats files")
    
    # Analyze
    analysis = analyze_scaling(stats)
    
    # Print report
    print_report(analysis)
    
    # Plot
    if args.plot:
        if HAS_MATPLOTLIB:
            plot_scaling(stats, args.plot)
        else:
            print("\n⚠ matplotlib not available, skipping plot")
            print("   Install with: pip install matplotlib")
    
    # Save analysis
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "stats": stats,
                "analysis": analysis,
            }, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()

