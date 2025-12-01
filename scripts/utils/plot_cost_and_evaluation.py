#!/usr/bin/env python3
"""
Generate plots for cost evaluation and comprehensive analysis.

1. Cost Evaluation Plot: Baseline vs Best configuration comparison
2. Comprehensive Evaluation Plot: Quality metrics (MRR, Recall@10, NDCG) across configurations
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Color scheme
BASELINE_COLOR = '#95A5A6'  # Gray
BEST_COLOR = '#FF6B35'  # Orange/red
OTHER_COLOR = '#4A90E2'  # Blue
HIGHLIGHT_COLOR = '#2ECC71'  # Green


def load_json(filepath: Path):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_cost_evaluation():
    """Plot cost evaluation: Baseline vs Best configuration."""
    print("Generating cost evaluation plot...")
    
    cost_data = load_json(RESULTS_DIR / "cost_evaluation.json")
    
    # Extract baseline and best configs
    baseline = None
    best = None
    
    for config in cost_data:
        if isinstance(config, dict):
            if config.get('is_baseline'):
                baseline = config
            elif config.get('is_best'):
                best = config
    
    if not baseline or not best:
        print("⚠️  Could not find baseline or best config in cost_evaluation.json")
        return
    
    # Extract metrics
    baseline_cost = baseline['avg_cost_per_query_usd']
    best_cost = best['avg_cost_per_query_usd']
    baseline_latency = baseline['avg_retrieval_latency_ms']
    best_latency = best['avg_retrieval_latency_ms']
    
    # Calculate cost reduction
    cost_reduction_pct = ((baseline_cost - best_cost) / baseline_cost) * 100
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Cost comparison (bar chart)
    configs = ['Baseline\n(Feature Dense)', 'Best\n(Transformer Hybrid+Rerank)']
    costs = [baseline_cost * 1000, best_cost * 1000]  # Convert to millicents for readability
    colors = [BASELINE_COLOR, BEST_COLOR]
    
    bars = ax1.bar(configs, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.3f}\nper 1K queries',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add cost reduction annotation
    ax1.annotate(f'{cost_reduction_pct:.1f}% Cost Reduction',
                xy=(1, best_cost * 1000), xytext=(0.5, best_cost * 1000 + 0.05),
                arrowprops=dict(arrowstyle='->', color=BEST_COLOR, lw=2),
                fontsize=12, fontweight='bold', color=BEST_COLOR,
                ha='center')
    
    ax1.set_ylabel('Cost per 1,000 Queries (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Cost Comparison: Baseline vs Best Configuration', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(costs) * 1.2)
    
    # Subplot 2: Cost vs Latency trade-off (scatter)
    ax2.scatter(baseline_latency, baseline_cost * 1000, 
               s=300, color=BASELINE_COLOR, alpha=0.7, 
               edgecolor='black', linewidth=2, label='Baseline', zorder=3)
    ax2.scatter(best_latency, best_cost * 1000,
               s=300, color=BEST_COLOR, alpha=0.7,
               edgecolor='black', linewidth=2, label='Best Config', zorder=3)
    
    # Add labels
    ax2.annotate('Baseline', 
                xy=(baseline_latency, baseline_cost * 1000),
                xytext=(baseline_latency + 50, baseline_cost * 1000 + 0.01),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BASELINE_COLOR, alpha=0.3))
    
    ax2.annotate('Best Config', 
                xy=(best_latency, best_cost * 1000),
                xytext=(best_latency + 100, best_cost * 1000 - 0.01),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BEST_COLOR, alpha=0.3))
    
    # Draw arrow showing trade-off
    ax2.annotate('', xy=(best_latency, best_cost * 1000),
                xytext=(baseline_latency, baseline_cost * 1000),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.5))
    
    ax2.set_xlabel('Average Retrieval Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost per 1,000 Queries (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost vs Latency Trade-off', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / 'cost_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {PLOTS_DIR / 'cost_evaluation.png'}")


def plot_comprehensive_evaluation():
    """Plot comprehensive evaluation metrics across configurations."""
    print("Generating comprehensive evaluation plot...")
    
    eval_data = load_json(RESULTS_DIR / "comprehensive_eval_improved.json")
    
    # Extract config metrics
    configs = []
    mrr_values = []
    recall_10_values = []
    ndcg_10_values = []
    latency_values = []
    config_names = []
    
    for config_data in eval_data.get('configs', []):
        config_name = config_data['config']['name']
        queries = config_data.get('queries', [])
        
        if not queries:
            continue
        
        # Calculate aggregated metrics
        mrr_list = [q.get('mrr', 0) for q in queries if 'mrr' in q]
        recall_10_list = [q.get('recall_at_10', 0) for q in queries if 'recall_at_10' in q]
        ndcg_10_list = [q.get('ndcg_at_10', 0) for q in queries if 'ndcg_at_10' in q]
        latency_list = [q.get('latency_ms', 0) for q in queries if 'latency_ms' in q]
        
        if mrr_list:
            configs.append(config_name)
            config_names.append(config_name.replace('_', ' ').title())
            mrr_values.append(np.mean(mrr_list))
            recall_10_values.append(np.mean(recall_10_list))
            ndcg_10_values.append(np.mean(ndcg_10_list))
            latency_values.append(np.mean(latency_list))
    
    if not configs:
        print("⚠️  No configuration data found in comprehensive_eval_improved.json")
        return
    
    # Identify best and baseline configs
    best_idx = np.argmax(mrr_values)
    baseline_idx = None
    for i, name in enumerate(configs):
        if 'baseline' in name.lower() or 'feature_dense' in name:
            baseline_idx = i
            break
    
    if baseline_idx is None:
        baseline_idx = 0  # Use first as baseline
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Evaluation: Quality Metrics Across Configurations', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color mapping
    colors = []
    for i, name in enumerate(configs):
        if i == best_idx:
            colors.append(BEST_COLOR)
        elif i == baseline_idx:
            colors.append(BASELINE_COLOR)
        else:
            colors.append(OTHER_COLOR)
    
    # Subplot 1: MRR comparison
    ax1 = axes[0, 0]
    bars1 = ax1.barh(config_names, mrr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Highlight best
    if best_idx is not None:
        bars1[best_idx].set_edgecolor('red')
        bars1[best_idx].set_linewidth(3)
        ax1.text(mrr_values[best_idx] + 0.02, best_idx,
                f'*** BEST ***\nMRR: {mrr_values[best_idx]:.3f}',
                va='center', fontsize=10, fontweight='bold', color=BEST_COLOR)
    
    ax1.set_xlabel('Mean Reciprocal Rank (MRR)', fontsize=12, fontweight='bold')
    ax1.set_title('MRR Comparison', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, max(mrr_values) * 1.3)
    ax1.grid(axis='x', alpha=0.3)
    
    # Subplot 2: Recall@10 comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(config_names, recall_10_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    if best_idx is not None:
        bars2[best_idx].set_edgecolor('red')
        bars2[best_idx].set_linewidth(3)
        ax2.text(recall_10_values[best_idx] + 0.02, best_idx,
                f'*** BEST ***\nRecall@10: {recall_10_values[best_idx]:.3f}',
                va='center', fontsize=10, fontweight='bold', color=BEST_COLOR)
    
    ax2.set_xlabel('Recall@10', fontsize=12, fontweight='bold')
    ax2.set_title('Recall@10 Comparison', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, max(recall_10_values) * 1.3)
    ax2.grid(axis='x', alpha=0.3)
    
    # Subplot 3: NDCG@10 comparison
    ax3 = axes[1, 0]
    bars3 = ax3.barh(config_names, ndcg_10_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    if best_idx is not None:
        bars3[best_idx].set_edgecolor('red')
        bars3[best_idx].set_linewidth(3)
        ax3.text(ndcg_10_values[best_idx] + 0.02, best_idx,
                f'*** BEST ***\nNDCG@10: {ndcg_10_values[best_idx]:.3f}',
                va='center', fontsize=10, fontweight='bold', color=BEST_COLOR)
    
    ax3.set_xlabel('NDCG@10', fontsize=12, fontweight='bold')
    ax3.set_title('NDCG@10 Comparison', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, max(ndcg_10_values) * 1.3)
    ax3.grid(axis='x', alpha=0.3)
    
    # Subplot 4: Quality vs Latency trade-off
    ax4 = axes[1, 1]
    
    # Scatter plot
    for i, (lat, mrr) in enumerate(zip(latency_values, mrr_values)):
        if i == best_idx:
            ax4.scatter(lat, mrr, s=400, color=BEST_COLOR, alpha=0.8,
                       edgecolor='red', linewidth=3, zorder=3, label='Best Config')
        elif i == baseline_idx:
            ax4.scatter(lat, mrr, s=300, color=BASELINE_COLOR, alpha=0.7,
                       edgecolor='black', linewidth=2, zorder=2, label='Baseline')
        else:
            ax4.scatter(lat, mrr, s=200, color=OTHER_COLOR, alpha=0.6,
                       edgecolor='black', linewidth=1, zorder=1)
    
    # Add labels
    for i, name in enumerate(config_names):
        if i == best_idx or i == baseline_idx:
            ax4.annotate(name.replace(' ', '\n'),
                        xy=(latency_values[i], mrr_values[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=colors[i], alpha=0.3))
    
    ax4.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('MRR', fontsize=12, fontweight='bold')
    ax4.set_title('Quality vs Latency Trade-off', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / 'comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {PLOTS_DIR / 'comprehensive_evaluation.png'}")


def main():
    """Generate all plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating Cost Evaluation and Comprehensive Analysis Plots")
    print("=" * 70)
    
    try:
        plot_cost_evaluation()
    except Exception as e:
        print(f"❌ Error generating cost evaluation plot: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        plot_comprehensive_evaluation()
    except Exception as e:
        print(f"❌ Error generating comprehensive evaluation plot: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Plot generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

