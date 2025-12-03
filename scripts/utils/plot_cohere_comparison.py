#!/usr/bin/env python3
"""
Generate comparison plot: Cohere API vs Local Models (ID-Based Evaluation)
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Patch

# Set style for publication-quality plots
matplotlib.use('Agg')  # Non-interactive backend
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots" / "best"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_config_metrics(data):
    """Extract metrics for each configuration."""
    configs = {}
    for config_data in data.get('configs', []):
        config_name = config_data['config']['name']
        configs[config_name] = {
            'mrr': config_data.get('mrr', 0),
            'recall_at_10': config_data.get('recall_at_10', 0),
            'ndcg_at_10': config_data.get('ndcg_at_10', 0),
            'latency_ms': config_data.get('avg_latency_ms', 0),
        }
    return configs


def plot_cohere_vs_local_comparison():
    """Plot comparison: Cohere API vs Local Models."""
    print("Generating Cohere API vs Local Models comparison plot...")
    
    # Load data
    try:
        cohere_data = load_json(RESULTS_DIR / "comprehensive_eval_cohere_id_based.json")
        local_data = load_json(RESULTS_DIR / "comprehensive_eval_improved.json")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Extract metrics
    cohere_configs = extract_config_metrics(cohere_data)
    local_configs = extract_config_metrics(local_data)
    
    # Key configurations to compare
    key_configs = [
        "transformer_hybrid_rerank",
        "transformer_dense",
        "transformer_hybrid",
        "baseline_feature_dense",
        "feature_hybrid",
    ]
    
    # Filter to only configs that exist in both
    configs_to_plot = [c for c in key_configs if c in cohere_configs and c in local_configs]
    
    if not configs_to_plot:
        print("No matching configurations found!")
        return
    
    # Prepare data
    config_names_short = {
        "transformer_hybrid_rerank": "Transformer\nHybrid+Rerank",
        "transformer_dense": "Transformer\nDense",
        "transformer_hybrid": "Transformer\nHybrid",
        "baseline_feature_dense": "Baseline\nFeature+Dense",
        "feature_hybrid": "Feature\nHybrid",
    }
    
    x = np.arange(len(configs_to_plot))
    width = 0.35
    
    # Extract metrics
    cohere_mrr = [cohere_configs[c]['mrr'] for c in configs_to_plot]
    local_mrr = [local_configs[c]['mrr'] for c in configs_to_plot]
    
    cohere_recall = [cohere_configs[c]['recall_at_10'] for c in configs_to_plot]
    local_recall = [local_configs[c]['recall_at_10'] for c in configs_to_plot]
    
    cohere_ndcg = [cohere_configs[c]['ndcg_at_10'] for c in configs_to_plot]
    local_ndcg = [local_configs[c]['ndcg_at_10'] for c in configs_to_plot]
    
    cohere_latency = [cohere_configs[c]['latency_ms'] for c in configs_to_plot]
    local_latency = [local_configs[c]['latency_ms'] for c in configs_to_plot]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cohere API vs Local Models: ID-Based Evaluation Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Colors
    COHERE_COLOR = '#2ECC71'  # Green
    LOCAL_COLOR = '#3498DB'   # Blue
    
    # Plot 1: MRR Comparison
    ax1 = axes[0, 0]
    bars1a = ax1.bar(x - width/2, local_mrr, width, label='Local Models', 
                     color=LOCAL_COLOR, alpha=0.8)
    bars1b = ax1.bar(x + width/2, cohere_mrr, width, label='Cohere API', 
                     color=COHERE_COLOR, alpha=0.8)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('MRR')
    ax1.set_title('Mean Reciprocal Rank (MRR)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([config_names_short.get(c, c) for c in configs_to_plot], 
                        rotation=0, ha='center')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(max(cohere_mrr), max(local_mrr)) * 1.15])
    
    # Add value labels
    for i, (local_val, cohere_val) in enumerate(zip(local_mrr, cohere_mrr)):
        ax1.text(i - width/2, local_val + 0.01, f'{local_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, cohere_val + 0.01, f'{cohere_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        # Add improvement percentage
        if cohere_val > local_val:
            improvement = ((cohere_val - local_val) / local_val) * 100
            ax1.text(i, max(local_val, cohere_val) + 0.05, 
                    f'+{improvement:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color=COHERE_COLOR)
    
    # Plot 2: Recall@10 Comparison
    ax2 = axes[0, 1]
    bars2a = ax2.bar(x - width/2, local_recall, width, label='Local Models', 
                     color=LOCAL_COLOR, alpha=0.8)
    bars2b = ax2.bar(x + width/2, cohere_recall, width, label='Cohere API', 
                     color=COHERE_COLOR, alpha=0.8)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Recall@10')
    ax2.set_xticks(x)
    ax2.set_xticklabels([config_names_short.get(c, c) for c in configs_to_plot], 
                        rotation=0, ha='center')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(max(cohere_recall), max(local_recall)) * 1.15])
    
    # Add value labels
    for i, (local_val, cohere_val) in enumerate(zip(local_recall, cohere_recall)):
        ax2.text(i - width/2, local_val + 0.01, f'{local_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, cohere_val + 0.01, f'{cohere_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        # Add improvement percentage
        if cohere_val > local_val:
            improvement = ((cohere_val - local_val) / local_val) * 100
            ax2.text(i, max(local_val, cohere_val) + 0.05, 
                    f'+{improvement:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color=COHERE_COLOR)
    
    # Plot 3: NDCG@10 Comparison
    ax3 = axes[1, 0]
    bars3a = ax3.bar(x - width/2, local_ndcg, width, label='Local Models', 
                     color=LOCAL_COLOR, alpha=0.8)
    bars3b = ax3.bar(x + width/2, cohere_ndcg, width, label='Cohere API', 
                     color=COHERE_COLOR, alpha=0.8)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('NDCG@10')
    ax3.set_title('Normalized Discounted Cumulative Gain (NDCG@10)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([config_names_short.get(c, c) for c in configs_to_plot], 
                        rotation=0, ha='center')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, max(max(cohere_ndcg), max(local_ndcg)) * 1.15])
    
    # Add value labels
    for i, (local_val, cohere_val) in enumerate(zip(local_ndcg, cohere_ndcg)):
        ax3.text(i - width/2, local_val + 0.01, f'{local_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        ax3.text(i + width/2, cohere_val + 0.01, f'{cohere_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        # Add improvement percentage
        if cohere_val > local_val:
            improvement = ((cohere_val - local_val) / local_val) * 100
            ax3.text(i, max(local_val, cohere_val) + 0.05, 
                    f'+{improvement:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color=COHERE_COLOR)
    
    # Plot 4: Latency Comparison
    ax4 = axes[1, 1]
    bars4a = ax4.bar(x - width/2, local_latency, width, label='Local Models', 
                     color=LOCAL_COLOR, alpha=0.8)
    bars4b = ax4.bar(x + width/2, cohere_latency, width, label='Cohere API', 
                     color=COHERE_COLOR, alpha=0.8)
    
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Average Latency')
    ax4.set_xticks(x)
    ax4.set_xticklabels([config_names_short.get(c, c) for c in configs_to_plot], 
                        rotation=0, ha='center')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (local_val, cohere_val) in enumerate(zip(local_latency, cohere_latency)):
        ax4.text(i - width/2, local_val + max(local_latency) * 0.02, 
                f'{local_val:.1f}ms', ha='center', va='bottom', fontsize=8)
        ax4.text(i + width/2, cohere_val + max(cohere_latency) * 0.02, 
                f'{cohere_val:.1f}ms', ha='center', va='bottom', fontsize=8)
        # Add improvement percentage (lower is better for latency)
        if cohere_val < local_val:
            improvement = ((local_val - cohere_val) / local_val) * 100
            ax4.text(i, max(local_val, cohere_val) + max(cohere_latency) * 0.08, 
                    f'-{improvement:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color=COHERE_COLOR)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "cohere_vs_local_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path}")


def main():
    """Generate comparison plot."""
    plot_cohere_vs_local_comparison()


if __name__ == "__main__":
    main()


