# Mermaid Diagrams for Presentation Slides

This file contains Mermaid code for all architecture and system diagrams needed for the presentation slides.

---

## 1. System Architecture Diagram (Slide 6)

```mermaid
graph TB
    Query[Query Input] --> Classifier[Query Classifier<br/>Feature-based or Transformer<br/>Confidence: 0.759-0.991]
    
    Classifier -->|OLTP<br/>50-80%| OLTP_Pipeline[OLTP Pipeline]
    Classifier -->|OLAP<br/>20-50%| OLAP_Pipeline[OLAP Pipeline]
    
    OLTP_Pipeline --> OLTP_Chunks[Fine-grained Chunks<br/>384-512 tokens<br/>Sentence-level]
    OLAP_Pipeline --> OLAP_Chunks[Coarse-grained Chunks<br/>Section-level<br/>Hierarchical]
    
    OLTP_Chunks --> Hybrid_OLTP[Hybrid Retrieval<br/>BM25 + Dense Vectors<br/>RRF Fusion]
    OLAP_Chunks --> Hybrid_OLAP[Hybrid Retrieval<br/>BM25 + Dense Vectors<br/>RRF Fusion]
    
    Hybrid_OLTP --> Embed_OLTP{Embedding<br/>Option}
    Hybrid_OLAP --> Embed_OLAP{Embedding<br/>Option}
    
    Embed_OLTP -->|Local| Local_Embed[Local Model<br/>all-MiniLM-L6-v2<br/>384-dim]
    Embed_OLTP -->|Cohere API| Cohere_Embed[Cohere API<br/>embed-english-light-v3.0<br/>384-dim]
    
    Embed_OLAP -->|Local| Local_Embed
    Embed_OLAP -->|Cohere API| Cohere_Embed
    
    Local_Embed --> Qdrant[Qdrant Vector Database<br/>HNSW Index<br/>ecetesla0]
    Cohere_Embed --> Qdrant
    
    Qdrant --> Rerank{Optional<br/>Reranking}
    
    Rerank -->|Local| Local_Rerank[Local Cross-encoders<br/>TinyBERT OLTP<br/>MiniLM OLAP]
    Rerank -->|Cohere API| Cohere_Rerank[Cohere API<br/>rerank-english-v3.0]
    Rerank -->|Skip| Output[Retrieved Chunks]
    
    Local_Rerank --> Output
    Cohere_Rerank --> Output
    
    Output --> LLM[LLM Answer Generation<br/>gpt-4o-mini]
    LLM --> Final[Final Answer]
    
    style Query fill:#e1f5ff
    style Classifier fill:#fff4e1
    style OLTP_Pipeline fill:#e8f5e9
    style OLAP_Pipeline fill:#fff3e0
    style Qdrant fill:#f3e5f5
    style Output fill:#e8f5e9
    style Final fill:#c8e6c9
```

---

## 2. Distributed Ingestion Architecture (Slide 12)

```mermaid
graph TB
    subgraph "Master/Orchestrator"
        Master[MPI Master Process<br/>Work Distribution<br/>Round-robin Assignment]
    end
    
    subgraph "Worker Node 0 (ecetesla1)"
        W0[Worker 0<br/>GPU-enabled]
        W0 --> W0_Load[Load Documents]
        W0_Load --> W0_Chunk[Chunk Documents]
        W0_Chunk --> W0_Embed[Embed Chunks<br/>Batch Size: 32<br/>GPU Accelerated<br/>98.6s ⚡]
        W0_Embed --> W0_Store[Store to Qdrant]
    end
    
    subgraph "Worker Node 1 (ecetesla2)"
        W1[Worker 1<br/>GPU-enabled<br/>⚠️ Bottleneck]
        W1 --> W1_Load[Load Documents]
        W1_Load --> W1_Chunk[Chunk Documents]
        W1_Chunk --> W1_Embed[Embed Chunks<br/>Batch Size: 32<br/>GPU Accelerated<br/>271.2s ⚠️ 2.75× slower]
        W1_Embed --> W1_Store[Store to Qdrant]
    end
    
    subgraph "Worker Node 2 (ecetesla4)"
        W2[Worker 2<br/>GPU-enabled]
        W2 --> W2_Load[Load Documents]
        W2_Load --> W2_Chunk[Chunk Documents]
        W2_Chunk --> W2_Embed[Embed Chunks<br/>Batch Size: 32<br/>GPU Accelerated<br/>186.6s]
        W2_Embed --> W2_Store[Store to Qdrant]
    end
    
    subgraph "Worker Node 3 (ecetesla4)"
        W3[Worker 3<br/>GPU-enabled<br/>Same node as Worker 2]
        W3 --> W3_Load[Load Documents]
        W3_Load --> W3_Chunk[Chunk Documents]
        W3_Chunk --> W3_Embed[Embed Chunks<br/>Batch Size: 32<br/>GPU Accelerated<br/>186.8s]
        W3_Embed --> W3_Store[Store to Qdrant]
    end
    
    subgraph "Vector Database"
        Qdrant[Qdrant<br/>ecetesla0<br/>HNSW Index<br/>Concurrent Writes]
    end
    
    Master -->|Distribute Work| W0
    Master -->|Distribute Work| W1
    Master -->|Distribute Work| W2
    Master -->|Distribute Work| W3
    
    W0_Store --> Qdrant
    W1_Store --> Qdrant
    W2_Store --> Qdrant
    W3_Store --> Qdrant
    
    W0_Embed -.->|Barrier Sync| Sync[Barrier Synchronization<br/>4 Workers Total]
    W1_Embed -.->|Barrier Sync| Sync
    W2_Embed -.->|Barrier Sync| Sync
    W3_Embed -.->|Barrier Sync| Sync
    
    Sync --> Final_Store[Final Storage<br/>813K chunks<br/>100% success rate<br/>1.87× speedup]
    
    style Master fill:#e1f5ff
    style W0 fill:#e8f5e9
    style W1 fill:#ffebee
    style W2 fill:#e8f5e9
    style W3 fill:#e8f5e9
    style Qdrant fill:#f3e5f5
    style Sync fill:#fff4e1
    style Final_Store fill:#c8e6c9
```

---

## 3. HNSW Graph Illustration (Slide 8)

```mermaid
graph TB
    subgraph "Layer 2 (Top)"
        L2_1[Node A]
        L2_2[Node B]
        L2_1 <--> L2_2
    end
    
    subgraph "Layer 1 (Middle)"
        L1_1[Node A]
        L1_2[Node B]
        L1_3[Node C]
        L1_4[Node D]
        L1_1 <--> L1_2
        L1_2 <--> L1_3
        L1_3 <--> L1_4
    end
    
    subgraph "Layer 0 (Bottom - All Items)"
        L0_1[Node A]
        L0_2[Node B]
        L0_3[Node C]
        L0_4[Node D]
        L0_5[Node E]
        L0_6[Node F]
        L0_7[Node G]
        L0_8[Node H]
        
        L0_1 <--> L0_2
        L0_2 <--> L0_3
        L0_3 <--> L0_4
        L0_4 <--> L0_5
        L0_5 <--> L0_6
        L0_6 <--> L0_7
        L0_7 <--> L0_8
        L0_8 <--> L0_1
    end
    
    L2_1 --> L1_1
    L2_2 --> L1_2
    
    L1_1 --> L0_1
    L1_2 --> L0_2
    L1_3 --> L0_3
    L1_4 --> L0_4
    
    Query[Query Vector] -->|Start at Top| L2_1
    L2_1 -->|Graph Walk| L2_2
    L2_2 -->|Move Down| L1_2
    L1_2 -->|Move Down| L0_2
    L0_2 -->|Find Nearest| Result[Nearest Neighbor<br/>O log n complexity]
    
    style Query fill:#e1f5ff
    style L2_1 fill:#fff4e1
    style L2_2 fill:#fff4e1
    style Result fill:#c8e6c9
```

---

## 4. Query Routing Flow Diagram (Slide 9)

```mermaid
graph TD
    Query[User Query] --> Classifier{Query Classifier}
    
    subgraph "Feature-Based Classifier"
        Feature[Random Forest<br/>Query Features<br/>Length, Keyword Density]
        Feature --> Feature_Conf[Confidence: 0.759<br/>Latency: <1ms]
        Feature_Conf --> Feature_Route{Routing Decision}
        Feature_Route -->|55%| Feature_OLTP[OLTP Pipeline]
        Feature_Route -->|45%| Feature_OLAP[OLAP Pipeline]
    end
    
    subgraph "Transformer Classifier"
        Transformer[MiniLM-L12-H384<br/>Sentence Transformers]
        Transformer --> Trans_Conf[Confidence: 0.991<br/>Latency: ~10ms]
        Trans_Conf --> Trans_Route{Routing Decision}
        Trans_Route -->|50%| Trans_OLTP[OLTP Pipeline]
        Trans_Route -->|50%| Trans_OLAP[OLAP Pipeline]
    end
    
    Classifier -->|Option 1| Feature
    Classifier -->|Option 2| Transformer
    
    Feature_OLTP --> OLTP_Retrieval[OLTP Retrieval<br/>Fine-grained chunks<br/>Fast HNSW search]
    Feature_OLAP --> OLAP_Retrieval[OLAP Retrieval<br/>Coarse-grained chunks<br/>Hierarchical search]
    
    Trans_OLTP --> OLTP_Retrieval
    Trans_OLAP --> OLAP_Retrieval
    
    OLTP_Retrieval --> Results[Retrieved Results]
    OLAP_Retrieval --> Results
    
    style Query fill:#e1f5ff
    style Classifier fill:#fff4e1
    style Feature fill:#e8f5e9
    style Transformer fill:#e8f5e9
    style Results fill:#c8e6c9
```

---

## 5. Hybrid Retrieval Architecture (Slide 11)

```mermaid
graph TB
    Query[User Query] --> Split{Hybrid Retrieval}
    
    Split -->|Sparse Path| BM25[BM25 Sparse Retrieval<br/>Rank-BM25 Library<br/>Full Corpus: 105K+ chunks<br/>Keyword Matching]
    
    Split -->|Dense Path| Embed{Embedding<br/>Option}
    
    Embed -->|Local| Local_Embed[Local Embedding<br/>sentence-transformers<br/>all-MiniLM-L6-v2<br/>384 dimensions]
    
    Embed -->|Cohere API| Cohere_Embed[Cohere API<br/>embed-english-light-v3.0<br/>384 dimensions<br/>Network Latency]
    
    Local_Embed --> HNSW[HNSW Index<br/>Qdrant<br/>ef_search: 200-400<br/>Semantic Similarity]
    Cohere_Embed --> HNSW
    
    BM25 --> BM25_Rank[BM25 Rankings<br/>Top-k Results]
    HNSW --> Dense_Rank[Dense Rankings<br/>Top-k Results]
    
    BM25_Rank --> RRF[Reciprocal Rank Fusion<br/>RRF Formula:<br/>score = 1/rank + 60<br/>Sum scores]
    Dense_Rank --> RRF
    
    RRF --> Combined[Combined Rankings<br/>Hybrid Results]
    
    Combined --> Rerank{Optional<br/>Reranking}
    
    Rerank -->|Skip| Final[Final Results]
    Rerank -->|Local| Local_Rerank[Local Cross-encoder<br/>TinyBERT/MiniLM<br/>+146ms overhead]
    Rerank -->|Cohere API| Cohere_Rerank[Cohere API<br/>rerank-english-v3.0<br/>Optimized latency]
    
    Local_Rerank --> Final
    Cohere_Rerank --> Final
    
    style Query fill:#e1f5ff
    style Split fill:#fff4e1
    style BM25 fill:#e8f5e9
    style HNSW fill:#f3e5f5
    style RRF fill:#fff3e0
    style Final fill:#c8e6c9
```

---

## 6. Cohere API vs Local Models Comparison (Slide 17B)

```mermaid
graph LR
    subgraph "Local Models"
        Local_Config[Configuration:<br/>transformer_hybrid_rerank]
        Local_MRR[MRR: 0.454]
        Local_Recall[Recall@10: 0.273]
        Local_NDCG[NDCG@10: 0.456]
        Local_Latency[Latency: 305.0ms]
        
        Local_Config --> Local_MRR
        Local_Config --> Local_Recall
        Local_Config --> Local_NDCG
        Local_Config --> Local_Latency
    end
    
    subgraph "Cohere API"
        Cohere_Config[Configuration:<br/>transformer_hybrid_rerank]
        Cohere_MRR[MRR: 0.482<br/>+6%]
        Cohere_Recall[Recall@10: 0.595<br/>+118% ⭐]
        Cohere_NDCG[NDCG@10: 0.520<br/>+14%]
        Cohere_Latency[Latency: 184.7ms<br/>-39% ⚡]
        
        Cohere_Config --> Cohere_MRR
        Cohere_Config --> Cohere_Recall
        Cohere_Config --> Cohere_NDCG
        Cohere_Config --> Cohere_Latency
    end
    
    Local_MRR -.->|Improvement| Cohere_MRR
    Local_Recall -.->|+118%| Cohere_Recall
    Local_NDCG -.->|+14%| Cohere_NDCG
    Local_Latency -.->|39% faster| Cohere_Latency
    
    style Local_Config fill:#e8f5e9
    style Cohere_Config fill:#fff3e0
    style Cohere_Recall fill:#c8e6c9
    style Cohere_Latency fill:#c8e6c9
```

---

## 7. Scaling Results Visualization (Slide 14)

```mermaid
graph TB
    subgraph "1 Worker"
        W1[Worker 0<br/>ecetesla1]
        W1 --> W1_Time[Time: 786.5s<br/>13.1 min]
        W1 --> W1_Throughput[Throughput: 1,034 chunks/s]
        W1 --> W1_Efficiency[Efficiency: 100%]
    end
    
    subgraph "2 Workers"
        W2_0[Worker 0<br/>ecetesla1]
        W2_1[Worker 1<br/>ecetesla2<br/>⚠️ Bottleneck]
        W2_0 --> W2_Time[Time: 795.6s<br/>13.3 min<br/>+1.2%]
        W2_1 --> W2_Time
        W2_Time --> W2_Throughput[Throughput: 1,022 chunks/s<br/>-1.1%]
        W2_Time --> W2_Efficiency[Efficiency: 49.5%<br/>Storage I/O Bottleneck]
    end
    
    subgraph "4 Workers"
        W4_0[Worker 0<br/>ecetesla1]
        W4_1[Worker 1<br/>ecetesla2<br/>⚠️ Bottleneck]
        W4_2[Worker 2<br/>ecetesla4]
        W4_3[Worker 3<br/>ecetesla4]
        W4_0 --> W4_Time[Time: 421.4s<br/>7.0 min<br/>-46.4% ⚡]
        W4_1 --> W4_Time
        W4_2 --> W4_Time
        W4_3 --> W4_Time
        W4_Time --> W4_Throughput[Throughput: 1,930 chunks/s<br/>+86.7% ⚡]
        W4_Time --> W4_Efficiency[Efficiency: 46.7%<br/>Speedup: 1.87×]
    end
    
    W1 -->|Minimal Improvement| W2_Time
    W2_Time -->|Significant Improvement| W4_Time
    
    style W1 fill:#e8f5e9
    style W2_1 fill:#ffebee
    style W4_1 fill:#ffebee
    style W4_Time fill:#c8e6c9
    style W4_Throughput fill:#c8e6c9
```

---

## Usage Instructions

1. **For Markdown/HTML**: Copy the Mermaid code blocks into your markdown file or HTML page with a Mermaid renderer.

2. **For PowerPoint/Google Slides**: 
   - Use online Mermaid editors (https://mermaid.live/) to generate SVG/PNG
   - Export and insert into slides

3. **For LaTeX/Beamer**: 
   - Use `mermaid` package or convert to TikZ
   - Or export as PNG and include as image

4. **Customization**: 
   - Adjust colors by modifying `style` blocks
   - Modify node labels and connections as needed
   - Add/remove nodes based on your specific needs

---

## Notes

- All diagrams use consistent color schemes:
  - Blue (#e1f5ff): Input/Query nodes
  - Yellow (#fff4e1): Decision/Classifier nodes
  - Green (#e8f5e9): Processing nodes
  - Purple (#f3e5f5): Database/Storage nodes
  - Light Green (#c8e6c9): Output/Final nodes
  - Red (#ffebee): Bottleneck/Warning nodes

- Diagrams are designed to be clear and readable in presentation format
- Adjust sizes and layouts as needed for your slide dimensions

