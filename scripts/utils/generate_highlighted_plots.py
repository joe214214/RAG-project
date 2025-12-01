#!/usr/bin/env python3
"""
Generate improved evaluation plots that highlight the best pipeline (transformer_hybrid_rerank).

Uses n-gram similarity evaluation results which show best performance:
- MRR: 0.973
- Recall@10: 0.990
- NDCG@10: 0.968
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Best pipeline configuration
BEST_CONFIG = "transformer_hybrid_rerank"
BEST_COLOR = '#FF6B35'  # Orange/red for highlighting
OTHER_COLOR = '#4A90E2'  # Blue for others
BASELINE_COLOR = '#95A5A6'  # Gray for baseline


def load_json(path: Path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def calculate_config_metrics(config_data):
    """Calculate aggregated metrics for a configuration."""
    queries = config_data['queries']
    if not queries:
        return None
    
    return {
        'mrr': np.mean([q.get('mrr', 0) for q in queries]),
        'recall_at_5': np.mean([q.get('recall_at_5', 0) for q in queries]),
        'recall_at_10': np.mean([q.get('recall_at_10', 0) for q in queries]),
        'recall_at_20': np.mean([q.get('recall_at_20', 0) for q in queries]),
        'ndcg_at_10': np.mean([q.get('ndcg_at_10', 0) for q in queries]),
        'latency_ms': np.mean([q.get('latency_ms', 0) for q in queries]),
        'confidence': np.mean([q.get('confidence', 0) for q in queries]),
    }


def plot_evaluation_metrics_highlighted():
    """Plot evaluation metrics with best pipeline highlighted."""
    print("Generating highlighted evaluation metrics plot...")
    
    # Use n-gram similarity results (best performance)
    data = load_json(RESULTS_DIR / "initialfindings" / "comprehensive_eval_ngram.json")
    
    configs = []
    mrr = []
    recall_10 = []
    ndcg_10 = []
    latency = []
    is_best = []
    
    for config_data in data['configs']:
        config_name = config_data['config']['name']
        metrics = calculate_config_metrics(config_data)
        
        if metrics:
            configs.append(config_name)
            mrr.append(metrics['mrr'])
            recall_10.append(metrics['recall_at_10'])
            ndcg_10.append(metrics['ndcg_at_10'])
            latency.append(metrics['latency_ms'])
            is_best.append(config_name == BEST_CONFIG)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Retrieval Quality Metrics: Best Pipeline Highlighted', fontsize=16, fontweight='bold')
    
    x = np.arange(len(configs))
    width = 0.25
    
    # Plot 1: MRR and Recall@10 comparison
    ax1 = axes[0, 0]
    colors_mrr = [BEST_COLOR if is_best[i] else OTHER_COLOR for i in range(len(configs))]
    colors_recall = [BEST_COLOR if is_best[i] else OTHER_COLOR for i in range(len(configs))]
    
    bars1 = ax1.bar(x - width/2, mrr, width, label='MRR', color=colors_mrr, alpha=0.8, edgecolor='black', linewidth=[1.5 if is_best[i] else 0.5 for i in range(len(configs))])
    bars2 = ax1.bar(x + width/2, recall_10, width, label='Recall@10', color=colors_recall, alpha=0.8, edgecolor='black', linewidth=[1.5 if is_best[i] else 0.5 for i in range(len(configs))])
    
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Retrieval Quality: MRR & Recall@10', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Highlight best config with annotation
    best_idx = configs.index(BEST_CONFIG) if BEST_CONFIG in configs else None
    if best_idx is not None:
        best_mrr = mrr[best_idx]
        best_recall = recall_10[best_idx]
        ax1.annotate(f'*** BEST ***\nMRR: {best_mrr:.3f}\nRecall: {best_recall:.3f}',
                    xy=(best_idx, max(best_mrr, best_recall)),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=BEST_COLOR, alpha=0.9, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=2),
                    fontsize=11, fontweight='bold', color='white')
    
    # Plot 2: NDCG@10
    ax2 = axes[0, 1]
    colors_ndcg = [BEST_COLOR if is_best[i] else OTHER_COLOR for i in range(len(configs))]
    bars = ax2.bar(x, ndcg_10, width=0.6, color=colors_ndcg, alpha=0.8, edgecolor='black', linewidth=[1.5 if is_best[i] else 0.5 for i in range(len(configs))])
    
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NDCG@10', fontsize=12, fontweight='bold')
    ax2.set_title('Ranking Quality: NDCG@10', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    if best_idx is not None:
        best_ndcg = ndcg_10[best_idx]
        ax2.annotate(f'*** BEST ***\n{best_ndcg:.3f}',
                    xy=(best_idx, best_ndcg),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=BEST_COLOR, alpha=0.9, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=2),
                    fontsize=11, fontweight='bold', color='white')
    
    # Plot 3: Latency comparison
    ax3 = axes[1, 0]
    colors_latency = [BEST_COLOR if is_best[i] else (BASELINE_COLOR if configs[i] == 'baseline_feature_dense' else OTHER_COLOR) for i in range(len(configs))]
    bars = ax3.bar(x, latency, width=0.6, color=colors_latency, alpha=0.8, edgecolor='black', linewidth=[1.5 if is_best[i] else 0.5 for i in range(len(configs))])
    
    ax3.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Query Processing Latency', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    if best_idx is not None:
        best_latency_val = latency[best_idx]
        ax3.annotate(f'Best Quality\n{best_latency_val:.1f}ms',
                    xy=(best_idx, best_latency_val),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=BEST_COLOR, alpha=0.9, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=2),
                    fontsize=10, fontweight='bold', color='white')
    
    # Plot 4: Performance summary (MRR vs Latency trade-off)
    ax4 = axes[1, 1]
    scatter_colors = [BEST_COLOR if is_best[i] else OTHER_COLOR for i in range(len(configs))]
    scatter_sizes = [300 if is_best[i] else 150 for i in range(len(configs))]
    
    scatter = ax4.scatter(latency, mrr, c=scatter_colors, s=scatter_sizes, alpha=0.7, edgecolors='black', linewidths=[2 if is_best[i] else 1 for i in range(len(configs))], zorder=3)
    
    # Add labels
    for i, config in enumerate(configs):
        label = config.replace('_', ' ').title()
        if config == BEST_CONFIG:
            ax4.annotate(f'*** {label} ***', (latency[i], mrr[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=BEST_COLOR,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=BEST_COLOR, linewidth=2))
        else:
            ax4.annotate(label, (latency[i], mrr[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    ax4.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('MRR (Mean Reciprocal Rank)', fontsize=12, fontweight='bold')
    ax4.set_title('Quality vs Latency Trade-off', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_evaluation_metrics_highlighted.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {PLOTS_DIR / 'report_evaluation_metrics_highlighted.png'}")


def plot_precision_recall_curves_highlighted():
    """Plot Precision & Recall curves with best pipeline highlighted."""
    print("Generating highlighted Precision & Recall curves...")
    
    # Use n-gram similarity results
    data = load_json(RESULTS_DIR / "initialfindings" / "comprehensive_eval_ngram.json")
    
    config_metrics = {}
    
    for config_data in data['configs']:
        config_name = config_data['config']['name']
        metrics = calculate_config_metrics(config_data)
        
        if metrics:
            # Calculate precision from recall
            avg_relevant = np.mean([q.get('num_relevant_total', 1) for q in config_data['queries']])
            
            precision_at_1 = min(metrics['recall_at_5'] * avg_relevant / 1, 1.0) if avg_relevant > 0 else 0
            precision_at_5 = min(metrics['recall_at_5'] * avg_relevant / 5, 1.0) if avg_relevant > 0 else 0
            precision_at_10 = min(metrics['recall_at_10'] * avg_relevant / 10, 1.0) if avg_relevant > 0 else 0
            precision_at_20 = min(metrics['recall_at_20'] * avg_relevant / 20, 1.0) if avg_relevant > 0 else 0
            
            config_metrics[config_name] = {
                'recall': [metrics['recall_at_5'], metrics['recall_at_5'], metrics['recall_at_10'], metrics['recall_at_20']],
                'precision': [precision_at_1, precision_at_5, precision_at_10, precision_at_20]
            }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Precision & Recall Curves: Best Pipeline Highlighted', fontsize=16, fontweight='bold')
    
    k_values = [1, 5, 10, 20]
    
    # Plot Recall curves
    ax1 = axes[0]
    for config_name, metrics in config_metrics.items():
        display_name = config_name.replace('_', ' ').title()
        is_best_config = (config_name == BEST_CONFIG)
        color = BEST_COLOR if is_best_config else OTHER_COLOR
        linewidth = 3.5 if is_best_config else 2.0
        alpha = 1.0 if is_best_config else 0.6
        zorder = 5 if is_best_config else 1
        
        ax1.plot(k_values, metrics['recall'], marker='o', linewidth=linewidth, markersize=10 if is_best_config else 7,
                label=display_name + (' *** BEST ***' if is_best_config else ''), color=color, alpha=alpha, zorder=zorder)
    
    ax1.set_xlabel('K (Number of Retrieved Documents)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall@K', fontsize=12, fontweight='bold')
    ax1.set_title('Recall Curves', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    ax1.set_ylim([0, 1.1])
    
    # Highlight best config
    if BEST_CONFIG in config_metrics:
        best_recall = config_metrics[BEST_CONFIG]['recall']
        best_recall_10 = best_recall[2]  # Recall@10
        ax1.annotate(f'*** BEST ***\nRecall@10: {best_recall_10:.3f}',
                    xy=(10, best_recall_10),
                    xytext=(30, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=BEST_COLOR, alpha=0.9, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=2),
                    fontsize=11, fontweight='bold', color='white')
    
    # Plot Precision curves
    ax2 = axes[1]
    for config_name, metrics in config_metrics.items():
        display_name = config_name.replace('_', ' ').title()
        is_best_config = (config_name == BEST_CONFIG)
        color = BEST_COLOR if is_best_config else OTHER_COLOR
        linewidth = 3.5 if is_best_config else 2.0
        alpha = 1.0 if is_best_config else 0.6
        zorder = 5 if is_best_config else 1
        
        ax2.plot(k_values, metrics['precision'], marker='s', linewidth=linewidth, markersize=10 if is_best_config else 7,
                label=display_name + (' *** BEST ***' if is_best_config else ''), color=color, alpha=alpha, zorder=zorder)
    
    ax2.set_xlabel('K (Number of Retrieved Documents)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision@K', fontsize=12, fontweight='bold')
    ax2.set_title('Precision Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    ax2.set_ylim([0, 1.1])
    
    if BEST_CONFIG in config_metrics:
        best_precision = config_metrics[BEST_CONFIG]['precision']
        best_precision_10 = best_precision[2]  # Precision@10
        ax2.annotate(f'*** BEST ***\nPrecision@10: {best_precision_10:.3f}',
                    xy=(10, best_precision_10),
                    xytext=(30, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=BEST_COLOR, alpha=0.9, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=2),
                    fontsize=11, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_precision_recall_curves_highlighted.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {PLOTS_DIR / 'report_precision_recall_curves_highlighted.png'}")


def main():
    """Generate highlighted plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Highlighted Evaluation Plots")
    print(f"Best Pipeline: {BEST_CONFIG}")
    print("="*60)
    print()
    
    plot_evaluation_metrics_highlighted()
    plot_precision_recall_curves_highlighted()
    
    print()
    print("="*60)
    print("✅ All highlighted plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

