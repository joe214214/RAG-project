#!/usr/bin/env python3
"""
Plot load test results comparing dense-only vs hybrid retrieval modes.

Usage:
    python scripts/utils/plot_load_test_results.py \
        --dense-results results/parallel_load_test.json \
        --hybrid-results results/parallel_load_test_hybrid.json \
        --output results/plots/load_test_comparison.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def load_results(json_path: Path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_metrics(results_data):
    """Extract metrics from results data."""
    concurrency = []
    qps = []
    p50 = []
    p95 = []
    p99 = []
    
    for result in results_data['results']:
        concurrency.append(result['concurrency'])
        qps.append(result['qps'])
        p50.append(result['p50_ms'])
        p95.append(result['p95_ms'])
        p99.append(result['p99_ms'])
    
    return {
        'concurrency': concurrency,
        'qps': qps,
        'p50': p50,
        'p95': p95,
        'p99': p99,
    }


def plot_load_test_comparison(dense_results_path, hybrid_results_path, output_path):
    """Create comprehensive load test comparison plots."""
    
    # Load data
    dense_data = load_results(dense_results_path)
    hybrid_data = load_results(hybrid_results_path)
    
    # Extract metrics
    dense_metrics = extract_metrics(dense_data)
    hybrid_metrics = extract_metrics(hybrid_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Load Test Results: Dense-Only vs Hybrid Retrieval', fontsize=16, fontweight='bold')
    
    # Plot 1: QPS vs Concurrency
    ax1 = axes[0, 0]
    ax1.plot(dense_metrics['concurrency'], dense_metrics['qps'], 
             marker='o', linewidth=2, markersize=8, label='Dense-Only', color='#2E86AB')
    ax1.plot(hybrid_metrics['concurrency'], hybrid_metrics['qps'], 
             marker='s', linewidth=2, markersize=8, label='Hybrid', color='#A23B72')
    ax1.set_xlabel('Concurrency Level', fontsize=12)
    ax1.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax1.set_title('Throughput (QPS) vs Concurrency', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sorted(set(dense_metrics['concurrency'] + hybrid_metrics['concurrency'])))
    
    # Add peak QPS annotations
    dense_max_idx = np.argmax(dense_metrics['qps'])
    hybrid_max_idx = np.argmax(hybrid_metrics['qps'])
    ax1.annotate(f'Peak: {dense_metrics["qps"][dense_max_idx]:.1f} QPS\n@ concurrency={dense_metrics["concurrency"][dense_max_idx]}',
                xy=(dense_metrics['concurrency'][dense_max_idx], dense_metrics['qps'][dense_max_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.annotate(f'Peak: {hybrid_metrics["qps"][hybrid_max_idx]:.1f} QPS\n@ concurrency={hybrid_metrics["concurrency"][hybrid_max_idx]}',
                xy=(hybrid_metrics['concurrency'][hybrid_max_idx], hybrid_metrics['qps'][hybrid_max_idx]),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightpink', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: P50 Latency vs Concurrency
    ax2 = axes[0, 1]
    ax2.plot(dense_metrics['concurrency'], dense_metrics['p50'], 
             marker='o', linewidth=2, markersize=8, label='Dense-Only (P50)', color='#2E86AB')
    ax2.plot(hybrid_metrics['concurrency'], hybrid_metrics['p50'], 
             marker='s', linewidth=2, markersize=8, label='Hybrid (P50)', color='#A23B72')
    ax2.set_xlabel('Concurrency Level', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('P50 Latency vs Concurrency', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sorted(set(dense_metrics['concurrency'] + hybrid_metrics['concurrency'])))
    ax2.set_yscale('log')  # Log scale for better visualization
    
    # Plot 3: P95 Latency vs Concurrency
    ax3 = axes[1, 0]
    ax3.plot(dense_metrics['concurrency'], dense_metrics['p95'], 
             marker='o', linewidth=2, markersize=8, label='Dense-Only (P95)', color='#2E86AB')
    ax3.plot(hybrid_metrics['concurrency'], hybrid_metrics['p95'], 
             marker='s', linewidth=2, markersize=8, label='Hybrid (P95)', color='#A23B72')
    ax3.set_xlabel('Concurrency Level', fontsize=12)
    ax3.set_ylabel('Latency (ms)', fontsize=12)
    ax3.set_title('P95 Latency vs Concurrency', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(sorted(set(dense_metrics['concurrency'] + hybrid_metrics['concurrency'])))
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Plot 4: Latency Comparison (P50, P95, P99) at different concurrency levels
    ax4 = axes[1, 1]
    
    # Select common concurrency levels
    common_levels = sorted(set(dense_metrics['concurrency']) & set(hybrid_metrics['concurrency']))
    
    dense_p50_at_common = [dense_metrics['p50'][dense_metrics['concurrency'].index(c)] for c in common_levels]
    dense_p95_at_common = [dense_metrics['p95'][dense_metrics['concurrency'].index(c)] for c in common_levels]
    hybrid_p50_at_common = [hybrid_metrics['p50'][hybrid_metrics['concurrency'].index(c)] for c in common_levels]
    hybrid_p95_at_common = [hybrid_metrics['p95'][hybrid_metrics['concurrency'].index(c)] for c in common_levels]
    
    x = np.arange(len(common_levels))
    width = 0.35
    
    ax4.bar(x - width/2, dense_p50_at_common, width, label='Dense-Only P50', color='#2E86AB', alpha=0.7)
    ax4.bar(x - width/2, dense_p95_at_common, width, bottom=dense_p50_at_common, 
            label='Dense-Only P95', color='#2E86AB', alpha=0.5)
    ax4.bar(x + width/2, hybrid_p50_at_common, width, label='Hybrid P50', color='#A23B72', alpha=0.7)
    ax4.bar(x + width/2, hybrid_p95_at_common, width, bottom=hybrid_p50_at_common,
            label='Hybrid P95', color='#A23B72', alpha=0.5)
    
    ax4.set_xlabel('Concurrency Level', fontsize=12)
    ax4.set_ylabel('Latency (ms)', fontsize=12)
    ax4.set_title('Latency Comparison (P50 + P95)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(common_levels)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LOAD TEST SUMMARY")
    print("="*60)
    print(f"\nDense-Only Mode:")
    print(f"  Peak QPS: {max(dense_metrics['qps']):.2f} (concurrency={dense_metrics['concurrency'][np.argmax(dense_metrics['qps'])]})")
    print(f"  P50 Latency (concurrency=1): {dense_metrics['p50'][0]:.2f}ms")
    print(f"  P95 Latency (concurrency=1): {dense_metrics['p95'][0]:.2f}ms")
    print(f"\nHybrid Mode:")
    print(f"  Peak QPS: {max(hybrid_metrics['qps']):.2f} (concurrency={hybrid_metrics['concurrency'][np.argmax(hybrid_metrics['qps'])]})")
    print(f"  P50 Latency (concurrency=1): {hybrid_metrics['p50'][0]:.2f}ms")
    print(f"  P95 Latency (concurrency=1): {hybrid_metrics['p95'][0]:.2f}ms")
    print(f"\nPerformance Ratio:")
    print(f"  QPS: {max(dense_metrics['qps'])/max(hybrid_metrics['qps']):.1f}x faster (dense-only)")
    print(f"  P50 Latency: {hybrid_metrics['p50'][0]/dense_metrics['p50'][0]:.1f}x slower (hybrid)")
    print(f"  P95 Latency: {hybrid_metrics['p95'][0]/dense_metrics['p95'][0]:.1f}x slower (hybrid)")


def main():
    parser = argparse.ArgumentParser(description="Plot load test comparison results")
    parser.add_argument("--dense-results", type=str, required=True,
                       help="Path to dense-only load test results JSON")
    parser.add_argument("--hybrid-results", type=str, required=True,
                       help="Path to hybrid load test results JSON")
    parser.add_argument("--output", type=str, default="results/plots/load_test_comparison.png",
                       help="Output plot path")
    
    args = parser.parse_args()
    
    plot_load_test_comparison(
        Path(args.dense_results),
        Path(args.hybrid_results),
        args.output
    )


if __name__ == "__main__":
    main()

