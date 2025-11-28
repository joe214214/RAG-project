## TL;DR

Use a pretrained DistilBERT and fine-tune it for OLTP vs OLAP when high precision or domain shift is expected; for quick prototypes or extremely low‑label regimes consider frozen-head or prompt/zero‑shot fallbacks. Fine-tuning should use progressive unfreezing, careful validation, and targeted augmentation for class imbalance.

----

## Fine-tune or use off-the-shelf

Fine-tuning is generally recommended when the query distribution or performance targets differ from the pretraining domain, because smaller fine-tuned models often outperform few‑shot LLM approaches on classification. When labels are very scarce or you need a quick baseline, freezing the backbone and training only the head or using few‑shot LLMs are viable fallbacks, but they tend to underperform tuned smaller models on many classification tasks [1] [2].

- **When to fine-tune** Use fine-tuning if the RAG query mix differs from general web text or you require high recall/precision tradeoffs; fine-tuning smaller models often beats few‑shot LLMs for text classification tasks [1].  
- **When off-the-shelf suffices** Use a frozen head, off‑the‑shelf checkpoint, or an LLM prompt when labels are extremely limited or latency/compute forbids training, noting potential accuracy gaps relative to fine-tuning [2] [1].  
- **Checkpoint choice** Start from a general DistilBERT base matching your text case (e.g., uncased for lowercased SQL/text) or prefer a domain-specific backbone if available (domain models outperform general models for specialized text) [3].

----

## Data preparation and labeling

High‑quality labels and preprocessing tailored to query types matter more than raw model size; label strategy and balanced validation enable reliable fine‑tuning. Use rule‑based bootstrapping plus human review for scale, augment the minority class thoughtfully, and align tokenization with the chosen checkpoint.

- **Labeling strategy** Combine rule‑based heuristics (syntactic cues, SQL keywords, aggregation/analytic patterns) with human verification to bootstrap labels and scale annotation efficiently, as rule‑based annotation has been used successfully to create training sets for transformer fine‑tuning [4].  
- **Preprocessing** Tokenize with the DistilBERT tokenizer matching your checkpoint (uncased vs cased) and maintain consistent normalization between training and inference [5].  
- **Class balance** Address imbalance proactively: apply targeted augmentation and loss techniques rather than only oversampling; focal loss and ensembling have been used to mitigate skewed classes in transformer fine‑tuning scenarios [6].  
- **Data augmentation** Use label‑preserving augmentation and synthetic negative generation where appropriate; LLMs can generate hard negatives or synthetic queries for retrieval/classification use cases [7].  
- **Minimum dataset size** Insufficient evidence in the supplied literature to state a hard minimum; published DistilBERT fine‑tuning examples range from moderate thousands to tens of thousands of examples for robust performance [8].

----

## Training protocol and hyperparameters

There is no single universal hyperparameter set in the supplied corpus; tune learning rate and batch size per task and use progressive unfreezing or parameter‑efficient methods when compute or label budget is constrained. Monitor validation metrics and use strategies that trade compute for stability.

- **Optimization strategy** Prioritize hyperparameter search over blind defaults because learning rate, batch size, and optimizer choices strongly affect results; empirical studies show sensitivity of accuracy and F1 to these settings [5] [9].  
- **Layer management** Consider freezing most backbone layers while training the classifier head and then progressively unfreezing with layer‑wise learning‑rate decay to stabilize training and reduce compute, a two‑step freeze‑then‑unfreeze approach has shown strong results in related classification tasks [2].  
- **Parameter‑efficient tuning** Use LoRA or adapter methods when you need to limit trainable parameters or store many task adapters; LoRA has been used successfully for domain adaptation of DistilBERT/BERT variants in practice, while full fine‑tuning can yield higher absolute performance if compute permits [10].  
- **Preventing overfitting** Apply validation‑driven early stopping, stratified splits or cross‑validation, targeted regularization (dropout in the classifier head), and consider focal loss or ensembling for imbalanced labels [6] [2].  
- **Hyperparameter specifics** The supplied corpus demonstrates the importance of tuning but does not provide a universally supported numeric recipe for learning rate, batch size, epochs, warmup steps, weight decay, or optimizer; practitioners should run small grid or Bayesian searches guided by validation behavior rather than rely on fixed defaults [5] [9].

----

## Evaluation, alternatives, and deployment

Measure with standard classification metrics, validate with stratified splits or cross‑validation, and prefer DistilBERT fine‑tuning for the best accuracy/cost tradeoff; deployment should emphasize model compression and latency profiling. Some deployment topics lack concrete details in the supplied literature.

- **Metrics to monitor** Track F1 score, precision, recall, and accuracy on a held‑out validation set (and class‑wise metrics) for binary OLTP/OLAP separation [9].  
- **Validation design** Use stratified validation or k‑fold cross‑validation to ensure stable estimates for both classes; progressive unfreezing experiments should be validated on the same splits used for early stopping [2] [5].  
- **Alternatives** Zero‑shot/few‑shot LLM prompting can be useful for low‑label regimes but often underperforms fine‑tuned DistilBERT classifiers on classification accuracy and cost‑efficiency [1] [11]. Lightweight classifiers that consume pretrained embeddings can be competitive when compute is extremely constrained, but fine‑tuned DistilBERT typically gives a better accuracy‑cost tradeoff for sentence/query classification tasks [11] [8].  
- **Adapters vs full fine‑tuning** LoRA/adapters reduce training footprint and enable many task heads; studies report successful LoRA usage, but full fine‑tuning can still outperform parameter‑efficient methods when compute and labeled data are sufficient [10].  
- **Deployment and inference** DistilBERT is materially smaller and faster than full BERT, providing better inference latency and lower memory use in production; the supplied corpus demonstrates reduced inference time as a practical advantage but does not provide step‑by‑step quantization or ONNX conversion guidance, so specific conversion/quantization recommendations are insufficiently supported by the included literature [8] [1].