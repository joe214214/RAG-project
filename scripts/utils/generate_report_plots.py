#!/usr/bin/env python3
"""
Generate plots for the comprehensive RAG system report.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
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


def plot_ingestion_scaling():
    """Plot ingestion scaling: 1M chunks (1 vs 2 workers)."""
    data_1w = load_json(RESULTS_DIR / "scaling" / "ingest_1M_1w.json")
    data_2w = load_json(RESULTS_DIR / "scaling" / "ingest_1M_2w.json")
    
    workers = [1, 2]
    total_times = [data_1w['timings']['total'], data_2w['timings']['total']]
    embed_times = [data_1w['timings']['embed'], data_2w['timings']['embed']]
    store_times = [data_1w['timings']['store'], data_2w['timings']['store']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(workers))
    width = 0.6
    
    p1 = ax.bar(x, embed_times, width, label='Embedding', color='#3498db')
    p2 = ax.bar(x, store_times, width, bottom=embed_times, label='Storage', color='#e74c3c')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Ingestion Scaling: 1M Chunks (1 vs 2 Workers)')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (tot, emb, st) in enumerate(zip(total_times, embed_times, store_times)):
        ax.text(i, tot + 20, f'{tot:.0f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_ingestion_scaling_1M.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_ingestion_scaling_1M.png'}")


def plot_multi_scale_scaling():
    """Plot multi-scale ingestion: 10K vs 1M chunks."""
    data_10k_1w = load_json(RESULTS_DIR / "scaling" / "ingest_1w.json")
    data_10k_2w = load_json(RESULTS_DIR / "scaling" / "ingest_2w.json")
    data_10k_4w = load_json(RESULTS_DIR / "scaling" / "ingest_4w.json")
    data_1M_1w = load_json(RESULTS_DIR / "scaling" / "ingest_1M_1w.json")
    data_1M_2w = load_json(RESULTS_DIR / "scaling" / "ingest_1M_2w.json")
    
    workers_10k = [1, 2, 4]
    times_10k = [
        data_10k_1w['timings']['total'],
        data_10k_2w['timings']['total'],
        data_10k_4w['timings']['total']
    ]
    
    workers_1M = [1, 2]
    times_1M = [
        data_1M_1w['timings']['total'],
        data_1M_2w['timings']['total']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(workers_10k, times_10k, 'o-', label='10K chunks', linewidth=2, markersize=8)
    ax.plot(workers_1M, times_1M, 's-', label='1M chunks', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Multi-Scale Ingestion: Dataset Size Dependency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_multi_scale_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_multi_scale_scaling.png'}")


def plot_throughput_scaling():
    """Plot throughput vs workers."""
    data_1M_1w = load_json(RESULTS_DIR / "scaling" / "ingest_1M_1w.json")
    data_1M_2w = load_json(RESULTS_DIR / "scaling" / "ingest_1M_2w.json")
    
    workers = [1, 2]
    throughput = [
        data_1M_1w['throughput']['chunks_per_sec'],
        data_1M_2w['throughput']['chunks_per_sec']
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(workers, throughput, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.axhline(y=throughput[0], color='gray', linestyle='--', alpha=0.5, label='Baseline throughput')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Throughput (chunks/sec)')
    ax.set_title('Throughput Scaling: 1M Chunks')
    ax.set_xticks(workers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for w, t in zip(workers, throughput):
        ax.text(w, t + 10, f'{t:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_throughput_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_throughput_scaling.png'}")


def plot_load_test_scaling():
    """Plot load test: QPS vs concurrency."""
    data = load_json(RESULTS_DIR / "evaluation" / "parallel_load_test.json")
    
    concurrency = [r['concurrency'] for r in data['results']]
    qps = [r['qps'] for r in data['results']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concurrency, qps, 'o-', linewidth=2, markersize=8, color='#3498db')
    
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Queries Per Second (QPS)')
    ax.set_title('Load Test: Baseline Dense-Only Retrieval')
    ax.grid(alpha=0.3)
    
    # Highlight peak
    max_idx = np.argmax(qps)
    ax.plot(concurrency[max_idx], qps[max_idx], 'ro', markersize=12, label=f'Peak: {qps[max_idx]:.1f} QPS')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_load_test_qps.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_load_test_qps.png'}")


def plot_latency_degradation():
    """Plot latency degradation with concurrency."""
    data = load_json(RESULTS_DIR / "evaluation" / "parallel_load_test.json")
    
    concurrency = [r['concurrency'] for r in data['results']]
    p95 = [r['p95_ms'] for r in data['results']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concurrency, p95, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_yscale('log')
    
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('P95 Latency (ms, log scale)')
    ax.set_title('Latency Degradation Under Load')
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_latency_degradation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_latency_degradation.png'}")


def plot_evaluation_metrics():
    """Plot evaluation metrics comparison."""
    data = load_json(RESULTS_DIR / "initialfindings" / "comprehensive_eval_ngram.json")
    
    configs = []
    mrr = []
    recall_10 = []
    latency = []
    
    for config_data in data['configs']:
        config_name = config_data['config']['name']
        queries = config_data['queries']
        
        if queries:
            avg_mrr = np.mean([q.get('mrr', 0) for q in queries])
            avg_recall = np.mean([q.get('recall_at_10', 0) for q in queries])
            avg_latency = np.mean([q.get('latency_ms', 0) for q in queries])
            
            configs.append(config_name.replace('_', ' ').title())
            mrr.append(avg_mrr)
            recall_10.append(avg_recall)
            latency.append(avg_latency)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    # MRR and Recall@10
    ax1.bar(x - width/2, mrr, width, label='MRR', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, recall_10, width, label='Recall@10', color='#2ecc71', alpha=0.8)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Score')
    ax1.set_title('Retrieval Quality Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Latency
    ax2.bar(x, latency, color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Retrieval Latency')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_evaluation_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_evaluation_metrics.png'}")


def plot_cost_vs_latency():
    """Plot cost vs latency trade-off."""
    data = load_json(RESULTS_DIR / "evaluation" / "cost_evaluation.json")
    
    configs = []
    costs = []
    latencies = []
    
    for config_data in data:
        if config_data.get('is_baseline') or config_data.get('is_best'):
            config_name = config_data['config']
            queries = config_data['queries']
            
            if queries:
                avg_cost = np.mean([q.get('answer_cost_usd', 0) for q in queries])
                avg_latency = np.mean([q.get('total_latency_ms', 0) for q in queries])
                
                configs.append(config_name)
                costs.append(avg_cost * 1000)  # Convert to millicents for readability
                latencies.append(avg_latency)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'baseline' in c.lower() else '#2ecc71' for c in configs]
    ax.scatter(latencies, costs, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, config in enumerate(configs):
        ax.annotate(config.replace('_', ' ').title(), 
                   (latencies[i], costs[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Total Latency (ms)')
    ax.set_ylabel('Cost per Query (millicents USD)')
    ax.set_title('Cost vs Latency Trade-off')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_cost_latency_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_cost_latency_tradeoff.png'}")


def plot_time_breakdown():
    """Plot time breakdown for 1M chunk ingestion."""
    data = load_json(RESULTS_DIR / "scaling" / "ingest_1M_1w.json")
    
    timings = data['timings']
    phases = ['Load', 'Chunk', 'Embed', 'Store']
    times = [
        timings['load'],
        timings['chunk'],
        timings['embed'],
        timings['store']
    ]
    colors = ['#95a5a6', '#f39c12', '#3498db', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(times, labels=phases, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Time Breakdown: 1M Chunk Ingestion (1 Worker)')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_time_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_time_breakdown.png'}")


def plot_worker_comparison():
    """Plot worker comparison showing load imbalance."""
    data = load_json(RESULTS_DIR / "scaling" / "ingest_1M_2w.json")
    
    workers = ['Worker 0', 'Worker 1']
    embed_times = [
        data['per_worker_timings'][0]['embed_time'],
        data['per_worker_timings'][1]['embed_time']
    ]
    store_times = [
        data['per_worker_timings'][0]['store_time'],
        data['per_worker_timings'][1]['store_time']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(workers))
    width = 0.35
    
    p1 = ax.bar(x - width/2, embed_times, width, label='Embedding', color='#3498db')
    p2 = ax.bar(x + width/2, store_times, width, label='Storage', color='#e74c3c')
    
    ax.set_xlabel('Worker')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Worker Load Imbalance: 1M Chunks (2 Workers)')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (emb, st) in enumerate(zip(embed_times, store_times)):
        ax.text(i - width/2, emb + 10, f'{emb:.0f}s', ha='center', va='bottom')
        ax.text(i + width/2, st + 10, f'{st:.0f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_worker_imbalance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_worker_imbalance.png'}")


def plot_scaling_1_to_4_workers():
    """Plot scaling from 1 to 4 workers showing time breakdown and speedup."""
    data_1w = load_json(RESULTS_DIR / "scaling" / "ingest_1w.json")
    data_2w = load_json(RESULTS_DIR / "scaling" / "ingest_2w.json")
    data_4w = load_json(RESULTS_DIR / "scaling" / "ingest_4w.json")
    
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
    
    # Calculate speedup (relative to 1 worker)
    speedup = [total_times[0] / t for t in total_times]
    efficiency = [s / w for s, w in zip(speedup, workers)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Stacked bar chart showing time breakdown
    x = np.arange(len(workers))
    width = 0.6
    
    p1 = ax1.bar(x, embed_times, width, label='Embedding', color='#3498db', alpha=0.8)
    p2 = ax1.bar(x, store_times, width, bottom=embed_times, label='Storage', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Number of Workers', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Ingestion Time Breakdown: 1→4 Workers\n(10K Documents)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workers)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (tot, emb, st) in enumerate(zip(total_times, embed_times, store_times)):
        ax1.text(i, tot + 0.5, f'{tot:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right subplot: Speedup and efficiency
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(workers, speedup, 'o-', linewidth=2.5, markersize=10, color='#2ecc71', label='Speedup')
    line2 = ax2_twin.plot(workers, efficiency, 's-', linewidth=2.5, markersize=10, color='#f39c12', label='Efficiency')
    
    # Ideal speedup line (linear scaling)
    ax2.plot(workers, workers, '--', linewidth=1.5, color='gray', alpha=0.5, label='Ideal Speedup')
    
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Speedup (relative to 1 worker)', fontsize=12, color='#2ecc71')
    ax2_twin.set_ylabel('Efficiency (%)', fontsize=12, color='#f39c12')
    ax2.set_title('Scaling Efficiency: 1→4 Workers', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(workers)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=10)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax2_twin.axhline(y=100, color='#f39c12', linestyle=':', alpha=0.3)
    
    # Add value labels
    for i, (w, s, e) in enumerate(zip(workers, speedup, efficiency)):
        ax2.text(w, s + 0.05, f'{s:.2f}x', ha='center', va='bottom', fontsize=9, color='#2ecc71', fontweight='bold')
        ax2_twin.text(w, e + 2, f'{e*100:.0f}%', ha='center', va='bottom', fontsize=9, color='#f39c12', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_scaling_1_to_4_workers.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_scaling_1_to_4_workers.png'}")


def plot_comprehensive_load_testing():
    """Plot comprehensive load testing: QPS and latency for Baseline, Reranked, and Hybrid configurations."""
    # Load data for all three configurations
    baseline_data = load_json(RESULTS_DIR / "evaluation" / "parallel_load_test.json")
    reranked_data = load_json(RESULTS_DIR / "evaluation" / "load_test_reranked.json")
    hybrid_data = load_json(RESULTS_DIR / "initialfindings" / "load_test_hybrid_fixed.json")
    
    # Extract concurrency levels and metrics
    def extract_metrics(data):
        concurrency = [r['concurrency'] for r in data['results']]
        qps = [r['qps'] for r in data['results']]
        p95 = [r['p95_ms'] for r in data['results']]
        return concurrency, qps, p95
    
    baseline_conc, baseline_qps, baseline_p95 = extract_metrics(baseline_data)
    reranked_conc, reranked_qps, reranked_p95 = extract_metrics(reranked_data)
    hybrid_conc, hybrid_qps, hybrid_p95 = extract_metrics(hybrid_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: QPS vs Concurrency
    ax1.plot(baseline_conc, baseline_qps, 'o-', linewidth=2.5, markersize=10, 
             color='#3498db', label='Baseline (Dense-Only)', alpha=0.9)
    ax1.plot(reranked_conc, reranked_qps, 's-', linewidth=2.5, markersize=10, 
             color='#2ecc71', label='Reranked (Dense + Reranker)', alpha=0.9)
    ax1.plot(hybrid_conc, hybrid_qps, '^-', linewidth=2.5, markersize=10, 
             color='#e74c3c', label='Hybrid (BM25 + Dense)', alpha=0.9)
    
    ax1.set_xlabel('Concurrency Level', fontsize=12)
    ax1.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax1.set_title('Load Testing: Throughput Comparison', fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    
    # Highlight peak QPS for baseline
    max_idx = np.argmax(baseline_qps)
    ax1.plot(baseline_conc[max_idx], baseline_qps[max_idx], 'o', 
             markersize=15, color='#3498db', markeredgecolor='black', 
             markeredgewidth=2, label=f'Peak Baseline: {baseline_qps[max_idx]:.1f} QPS')
    ax1.annotate(f'{baseline_qps[max_idx]:.1f} QPS', 
                xy=(baseline_conc[max_idx], baseline_qps[max_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Right subplot: P95 Latency vs Concurrency (log scale)
    ax2.plot(baseline_conc, baseline_p95, 'o-', linewidth=2.5, markersize=10, 
             color='#3498db', label='Baseline (Dense-Only)', alpha=0.9)
    ax2.plot(reranked_conc, reranked_p95, 's-', linewidth=2.5, markersize=10, 
             color='#2ecc71', label='Reranked (Dense + Reranker)', alpha=0.9)
    ax2.plot(hybrid_conc, hybrid_p95, '^-', linewidth=2.5, markersize=10, 
             color='#e74c3c', label='Hybrid (BM25 + Dense)', alpha=0.9)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Concurrency Level', fontsize=12)
    ax2.set_ylabel('P95 Latency (ms, log scale)', fontsize=12)
    ax2.set_title('Load Testing: Latency Degradation', fontsize=14)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3, which='both')
    
    # Add annotations for key latency points
    # Baseline at concurrency=1
    ax2.annotate(f'{baseline_p95[0]:.1f}ms', 
                xy=(baseline_conc[0], baseline_p95[0]),
                xytext=(5, -20), textcoords='offset points',
                fontsize=8, color='#3498db', fontweight='bold')
    
    # Hybrid at concurrency=100 (if exists)
    if len(hybrid_p95) > 0:
        last_idx = len(hybrid_p95) - 1
        ax2.annotate(f'{hybrid_p95[last_idx]:.0f}ms', 
                    xy=(hybrid_conc[last_idx], hybrid_p95[last_idx]),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "report_comprehensive_load_testing.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {PLOTS_DIR / 'report_comprehensive_load_testing.png'}")


def main():
    """Generate all plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots for comprehensive report...")
    
    plot_ingestion_scaling()
    plot_multi_scale_scaling()
    plot_throughput_scaling()
    plot_load_test_scaling()
    plot_latency_degradation()
    plot_evaluation_metrics()
    plot_cost_vs_latency()
    plot_time_breakdown()
    plot_worker_comparison()
    plot_scaling_1_to_4_workers()
    plot_comprehensive_load_testing()
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
