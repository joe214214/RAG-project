#!/usr/bin/env python3
"""
Plot load test results from terminal output data.

Creates comprehensive comparison plots for dense-only vs hybrid retrieval.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

# Data from terminal output
dense_data = {
    'concurrency': [1, 5, 10, 20, 50],
    'qps': [29.83, 40.91, 39.70, 37.57, 35.85],
    'p50': [32.91, 118.12, 252.11, 529.71, 1386.76],
    'p95': [38.07, 164.45, 277.42, 580.06, 1535.14],
    'p99': [39.90, 190.30, 284.64, 599.00, 1604.99],
}

hybrid_data = {
    'concurrency': [1, 5, 10],
    'qps': [4.25, 4.38, 4.28],
    'p50': [259.96, 1223.97, 2170.89],
    'p95': [382.22, 1593.08, 2975.07],
    'p99': [399.30, 1597.36, 3046.62],
}

def plot_load_test_comparison(output_path="results/plots/load_test_comparison.png"):
    """Create comprehensive load test comparison plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Load Test Results: Dense-Only vs Hybrid Retrieval', fontsize=16, fontweight='bold')
    
    # Plot 1: QPS vs Concurrency
    ax1 = axes[0, 0]
    ax1.plot(dense_data['concurrency'], dense_data['qps'], 
             marker='o', linewidth=2.5, markersize=10, label='Dense-Only', color='#2E86AB', zorder=3)
    ax1.plot(hybrid_data['concurrency'], hybrid_data['qps'], 
             marker='s', linewidth=2.5, markersize=10, label='Hybrid', color='#A23B72', zorder=3)
    ax1.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Queries Per Second (QPS)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput (QPS) vs Concurrency', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.set_xticks(sorted(set(dense_data['concurrency'] + hybrid_data['concurrency'])))
    
    # Add peak QPS annotations
    dense_max_idx = np.argmax(dense_data['qps'])
    hybrid_max_idx = np.argmax(hybrid_data['qps'])
    ax1.annotate(f'Peak: {dense_data["qps"][dense_max_idx]:.1f} QPS\n@ concurrency={dense_data["concurrency"][dense_max_idx]}',
                xy=(dense_data['concurrency'][dense_max_idx], dense_data['qps'][dense_max_idx]),
                xytext=(15, 15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.8, edgecolor='#2E86AB'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#2E86AB', lw=1.5),
                fontsize=10, fontweight='bold')
    ax1.annotate(f'Peak: {hybrid_data["qps"][hybrid_max_idx]:.1f} QPS\n@ concurrency={hybrid_data["concurrency"][hybrid_max_idx]}',
                xy=(hybrid_data['concurrency'][hybrid_max_idx], hybrid_data['qps'][hybrid_max_idx]),
                xytext=(15, -35), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='lightpink', alpha=0.8, edgecolor='#A23B72'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#A23B72', lw=1.5),
                fontsize=10, fontweight='bold')
    
    # Plot 2: P50 Latency vs Concurrency
    ax2 = axes[0, 1]
    ax2.plot(dense_data['concurrency'], dense_data['p50'], 
             marker='o', linewidth=2.5, markersize=10, label='Dense-Only (P50)', color='#2E86AB', zorder=3)
    ax2.plot(hybrid_data['concurrency'], hybrid_data['p50'], 
             marker='s', linewidth=2.5, markersize=10, label='Hybrid (P50)', color='#A23B72', zorder=3)
    ax2.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('P50 Latency vs Concurrency', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.set_xticks(sorted(set(dense_data['concurrency'] + hybrid_data['concurrency'])))
    ax2.set_yscale('log')  # Log scale for better visualization
    
    # Plot 3: P95 Latency vs Concurrency
    ax3 = axes[1, 0]
    ax3.plot(dense_data['concurrency'], dense_data['p95'], 
             marker='o', linewidth=2.5, markersize=10, label='Dense-Only (P95)', color='#2E86AB', zorder=3)
    ax3.plot(hybrid_data['concurrency'], hybrid_data['p95'], 
             marker='s', linewidth=2.5, markersize=10, label='Hybrid (P95)', color='#A23B72', zorder=3)
    ax3.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('P95 Latency vs Concurrency', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3, zorder=0)
    ax3.set_xticks(sorted(set(dense_data['concurrency'] + hybrid_data['concurrency'])))
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Plot 4: Latency Comparison (P50, P95) at different concurrency levels
    ax4 = axes[1, 1]
    
    # Select common concurrency levels
    common_levels = sorted(set(dense_data['concurrency']) & set(hybrid_data['concurrency']))
    
    dense_p50_at_common = [dense_data['p50'][dense_data['concurrency'].index(c)] for c in common_levels]
    dense_p95_at_common = [dense_data['p95'][dense_data['concurrency'].index(c)] for c in common_levels]
    hybrid_p50_at_common = [hybrid_data['p50'][hybrid_data['concurrency'].index(c)] for c in common_levels]
    hybrid_p95_at_common = [hybrid_data['p95'][hybrid_data['concurrency'].index(c)] for c in common_levels]
    
    x = np.arange(len(common_levels))
    width = 0.35
    
    # Create stacked bars for P50 and P95
    ax4.bar(x - width/2, dense_p50_at_common, width, label='Dense-Only P50', color='#2E86AB', alpha=0.8, zorder=2)
    ax4.bar(x - width/2, [dense_p95_at_common[i] - dense_p50_at_common[i] for i in range(len(common_levels))], 
            width, bottom=dense_p50_at_common, label='Dense-Only P95-P50', color='#2E86AB', alpha=0.5, zorder=2)
    ax4.bar(x + width/2, hybrid_p50_at_common, width, label='Hybrid P50', color='#A23B72', alpha=0.8, zorder=2)
    ax4.bar(x + width/2, [hybrid_p95_at_common[i] - hybrid_p50_at_common[i] for i in range(len(common_levels))],
            width, bottom=hybrid_p50_at_common, label='Hybrid P95-P50', color='#A23B72', alpha=0.5, zorder=2)
    
    ax4.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Latency Comparison (P50 + P95)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(common_levels)
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y', zorder=0)
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
    print(f"  Peak QPS: {max(dense_data['qps']):.2f} (concurrency={dense_data['concurrency'][np.argmax(dense_data['qps'])]})")
    print(f"  P50 Latency (concurrency=1): {dense_data['p50'][0]:.2f}ms")
    print(f"  P95 Latency (concurrency=1): {dense_data['p95'][0]:.2f}ms")
    print(f"\nHybrid Mode:")
    print(f"  Peak QPS: {max(hybrid_data['qps']):.2f} (concurrency={hybrid_data['concurrency'][np.argmax(hybrid_data['qps'])]})")
    print(f"  P50 Latency (concurrency=1): {hybrid_data['p50'][0]:.2f}ms")
    print(f"  P95 Latency (concurrency=1): {hybrid_data['p95'][0]:.2f}ms")
    print(f"\nPerformance Ratio:")
    print(f"  QPS: {max(dense_data['qps'])/max(hybrid_data['qps']):.1f}x faster (dense-only)")
    print(f"  P50 Latency: {hybrid_data['p50'][0]/dense_data['p50'][0]:.1f}x slower (hybrid)")
    print(f"  P95 Latency: {hybrid_data['p95'][0]/dense_data['p95'][0]:.1f}x slower (hybrid)")


if __name__ == "__main__":
    plot_load_test_comparison()

