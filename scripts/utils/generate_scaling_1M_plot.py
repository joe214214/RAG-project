#!/usr/bin/env python3
"""
Generate scaling plot for 1M chunk ingestion (1, 2, 4 workers).
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import seaborn as sns

# Set non-interactive backend
matplotlib.use('Agg')

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_scaling_1M():
    """Plot 1M chunk ingestion scaling: 1, 2, 4 workers."""
    # Load data
    data_1w = load_json(RESULTS_DIR / "ingest_1M_1w.json")
    data_2w = load_json(RESULTS_DIR / "ingest_1M_2w.json")
    data_4w = load_json(RESULTS_DIR / "ingest_1M_4w.json")
    
    workers = [1, 2, 4]
    total_times = [
        data_1w['timings']['total'],
        data_2w['timings']['total'],
        data_4w['timings']['total']
    ]
    embed_times = [
        data_1w['timings']['embed'],
        data_2w['timings']['embed'],
        data_4w['timings']['embed']
    ]
    store_times = [
        data_1w['timings']['store'],
        data_2w['timings']['store'],
        data_4w['timings']['store']
    ]
    load_chunk_times = [
        data_1w['timings']['load'] + data_1w['timings']['chunk'],
        data_2w['timings']['load'] + data_2w['timings']['chunk'],
        data_4w['timings']['load'] + data_4w['timings']['chunk']
    ]
    
    throughputs = [
        data_1w['throughput']['chunks_per_sec'],
        data_2w['throughput']['chunks_per_sec'],
        data_4w['throughput']['chunks_per_sec']
    ]
    
    # Calculate speedup and efficiency
    baseline_time = total_times[0]
    speedups = [baseline_time / t for t in total_times]
    efficiencies = [s / w * 100 for s, w in zip(speedups, workers)]
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Time breakdown (stacked bar)
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(workers))
    width = 0.6
    
    p1 = ax1.bar(x, embed_times, width, label='Embedding', color='#3498db')
    p2 = ax1.bar(x, store_times, width, bottom=embed_times, label='Storage', color='#e74c3c')
    p3 = ax1.bar(x, load_chunk_times, width, bottom=np.array(embed_times) + np.array(store_times), 
                 label='Load+Chunk', color='#95a5a6')
    
    ax1.set_xlabel('Number of Workers', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Time Breakdown by Phase', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(workers)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels for total time
    for i, tot in enumerate(total_times):
        ax1.text(i, tot + 30, f'{tot:.0f}s', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Speedup and Efficiency
    ax2 = plt.subplot(2, 2, 2)
    x2 = np.arange(len(workers))
    width2 = 0.35
    
    bars1 = ax2.bar(x2 - width2/2, speedups, width2, label='Speedup', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(x2 + width2/2, efficiencies, width2, label='Efficiency (%)', color='#f39c12', alpha=0.8)
    
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Speedup / Efficiency (%)', fontsize=12)
    ax2.set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(workers)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Ideal (100%)')
    
    # Add value labels
    for i, (sp, eff) in enumerate(zip(speedups, efficiencies)):
        ax2.text(i - width2/2, sp + 0.05, f'{sp:.2f}x', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width2/2, eff + 2, f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Throughput
    ax3 = plt.subplot(2, 2, 3)
    bars3 = ax3.bar(x, throughputs, width, color='#9b59b6', alpha=0.8)
    
    ax3.set_xlabel('Number of Workers', fontsize=12)
    ax3.set_ylabel('Throughput (chunks/sec)', fontsize=12)
    ax3.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(workers)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, thr in enumerate(throughputs):
        ax3.text(i, thr + 50, f'{thr:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Total Time (line plot with ideal scaling)
    ax4 = plt.subplot(2, 2, 4)
    ideal_times = [baseline_time / w for w in workers]
    
    ax4.plot(workers, total_times, 'o-', label='Actual', linewidth=2.5, markersize=10, color='#e74c3c')
    ax4.plot(workers, ideal_times, 's--', label='Ideal (linear)', linewidth=2, markersize=8, color='#2ecc71', alpha=0.7)
    
    ax4.set_xlabel('Number of Workers', fontsize=12)
    ax4.set_ylabel('Total Time (seconds)', fontsize=12)
    ax4.set_title('Total Time vs Ideal Scaling', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Add value labels
    for i, (w, t) in enumerate(zip(workers, total_times)):
        ax4.text(w, t + 30, f'{t:.0f}s', ha='center', va='bottom', fontweight='bold')
    
    # Add annotation for 4-worker result
    ax4.annotate('1.87x speedup\n(46.7% efficiency)', 
                 xy=(4, total_times[2]), 
                 xytext=(3.5, total_times[2] + 100),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('1M Chunk Ingestion Scaling: 1 → 2 → 4 Workers', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(PLOTS_DIR / "scaling_1M_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'scaling_1M_plot.png'}")
    
    # Print summary
    print("\nScaling Summary:")
    print(f"  Workers | Time (s) | Speedup | Efficiency | Throughput (chunks/s)")
    print(f"  --------|----------|---------|------------|---------------------")
    for w, t, s, e, thr in zip(workers, total_times, speedups, efficiencies, throughputs):
        print(f"  {w:8d} | {t:8.1f} | {s:7.2f}x | {e:10.1f}% | {thr:20.1f}")


if __name__ == "__main__":
    plot_scaling_1M()

