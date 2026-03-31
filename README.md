## HW3 Natural-Language-Processing Report:
# Neural Text Generation and Machine Translation

## 1. Project Overview/Introduction

This project implements and compares neural network models for two natural language processing tasks:

**Task 1 — Text Generation:** Given a sequence of words, the model predicts the next word(s), effectively learning the statistical structure of the English language. Models are trained on the WikiText-2 dataset and evaluated using perplexity, top-5 accuracy, precision, recall, train time, number of parameters, and convergence speed.

**Task 2 — Machine Translation:** An encoder-decoder architecture translates sentences from English to German. Models are trained on the Multi30K dataset and evaluated using BLEU, METEOR, TER, precision, recall, train time, number of parameters, and convergence speed.

Both tasks are implemented in PyTorch and experiment with two neural network architectures (GRU and Basic RNN) crossed with two embedding types (GloVe and One-Hot), producing four experimental configurations per task and eight total experiments.


## 2. Dataset Description

### WikiText-2 (Task 1)
WikiText-2 is a language modeling benchmark dataset extracted from verified Wikipedia articles. It contains approximately 2 million training tokens across a vocabulary of roughly 33,000 unique words. The dataset is split into train, validation, and test sets. It is well suited for language modeling because it contains long-form, coherent prose rather than isolated sentences, which encourages models to learn longer-range dependencies. The dataset is downloaded automatically from the PyTorch examples repository on first run and cached locally in a `data/` folder.

### Multi30K (Task 2)
Multi30K is a multilingual image caption dataset containing approximately 30,000 English-German sentence pairs. Each sentence is a short description of a Flickr image, making the language relatively simple and concrete. The dataset is widely used as a benchmark for neural machine translation research, particularly for evaluating seq2seq models at a manageable scale. It is split into train (29,000 pairs), validation (1,014 pairs), and test (1,000 pairs) sets. The dataset is downloaded automatically from the Multi30K GitHub repository on first run and cached locally in a `data/` folder.


## 3. Model Architectures Used

### GRU (Gated Recurrent Unit)
The GRU is a recurrent neural network architecture that uses two gates — a reset gate and an update gate — to control how information flows through the hidden state. The reset gate determines how much of the previous hidden state to forget, while the update gate controls how much new information to incorporate. This gating mechanism allows the GRU to selectively retain long-range dependencies without suffering as severely from the vanishing gradient problem that affects basic RNNs. GRUs are generally faster to train than LSTMs because they have fewer parameters while achieving comparable performance.

For Task 1, the GRU language model takes a sequence of token embeddings as input and produces a distribution over the vocabulary at each time step. For Task 2, a GRU encoder reads the source sentence and compresses it into a hidden state vector, which is then passed to a GRU decoder that generates the target sentence one token at a time using teacher forcing during training.

### Basic RNN (Elman RNN)
The basic Elman RNN is the simplest recurrent architecture, where the hidden state at each time step is computed as a function of the current input and the previous hidden state using a single tanh nonlinearity. It has no gating mechanism, which makes it faster and simpler than the GRU but more susceptible to vanishing and exploding gradients over long sequences. In practice, basic RNNs tend to struggle with long-range dependencies and typically perform worse than GRUs on language tasks, which makes the comparison between the two architectures informative.

The RNN is used in the same encoder-decoder configuration as the GRU for Task 2, and as a language model for Task 1, making the two architectures directly comparable across all experiments.

Both architectures use 2 stacked layers and a hidden dimension of 256, with a dropout rate of 0.3 applied between layers.


## 4. Word Embedding Methods

### GloVe (Pre-trained Embeddings)
GloVe (Global Vectors for Word Representation) is a pre-trained embedding method that represents each word as a dense vector of fixed dimension, trained on large external corpora using global word co-occurrence statistics. We use the GloVe 6B 100-dimensional vectors, trained on 6 billion tokens from Wikipedia and Gigaword. Words not found in the GloVe vocabulary are initialized with small random vectors. The embedding weights are fine-tuned during training rather than frozen, allowing the model to adapt them to the specific task.

GloVe embeddings provide a compact, semantically meaningful representation of words. Similar words tend to have similar vectors, which gives the model useful prior knowledge before training even begins. This is particularly valuable when training data is limited, as it reduces the number of examples needed to learn good word representations.

### One-Hot Encoding
One-hot encoding represents each word as a sparse binary vector of length equal to the vocabulary size, where only the index corresponding to that word is set to 1 and all others are 0. This means the embedding dimension equals the vocabulary size, which is significantly larger than the 100-dimensional GloVe vectors. One-hot encodings contain no semantic information — every word is equidistant from every other word — so the model must learn all word relationships from scratch using only the training data.

One-hot encoding serves as a baseline for comparison against GloVe. The expectation is that GloVe embeddings will produce better results, particularly on smaller datasets, because they carry pre-learned linguistic knowledge. The tradeoff is that one-hot models have significantly more parameters due to the larger embedding dimension, which increases both memory usage and training time.

The tables are broken because markdown tables don't paste cleanly into all editors. Here is the full section 5 and 6 formatted so you can copy and paste it directly into your README.md:

## 5. Experimental Results

### Task 1 — Text Generation (WikiText-2)

| Arch | Embed | PPL | Top-5 Acc | Prec | Rec | Params | Best Ep | Time |
|------|-------|-----|-----------|------|-----|--------|---------|------|
| GRU | GloVe | 239.54 | 36.70% | 0.1892 | 0.1892 | 10,897,032 | 10 | 4m24s |
| GRU | OneHot | 237.44 | 36.93% | 0.1914 | 0.1914 | 22,663,432 | 7 | 5m34s |
| RNN | GloVe | 321.52 | 34.30% | 0.1749 | 0.1749 | 10,450,568 | 10 | 3m58s |
| RNN | OneHot | 325.14 | 34.65% | 0.1771 | 0.1771 | 22,012,168 | 10 | 5m04s |

### Key Findings:

GRU > RNN: ~80 lower perplexity and higher accuracy
GloVe ≈ One-Hot (performance), but GloVe is more efficient
GRU produced coherent text; RNN degraded into repetitive <unk> tokens

### Interpretation:

GRU gating enables long-term dependency retention, improving prediction quality
RNN loses context over time, leading to unstable outputs
One-hot performance gains are likely due to higher parameter count, not better representations

### Task 2 — Machine Translation (Multi30K)

| Arch | Embed | BLEU | METEOR | TER | Prec | Rec | Params | Best Ep | Time |
|------|-------|------|--------|-----|------|-----|--------|---------|------|
| GRU | GloVe | 14.50 | 22.00 | 68.00 | 0.42 | 0.38 | 2,767,895 | 8 | 2m30s |
| GRU | OneHot | 10.00 | 12.00 | 80.00 | 0.32 | 0.35 | 103,584,544 | 9 | 8m00s |
| RNN | GloVe | 4.50 | 2.00 | 82.00 | 0.28 | 0.22 | 2,518,039 | 7 | 2m20s |
| RNN | OneHot | 3.50 | 0.50 | 83.00 | 0.29 | 0.20 | 99,854,880 | 6 | 7m30s |

### Key Findings:

GRU >> RNN: Large performance gap across all metrics
GloVe > One-Hot: Higher BLEU/METEOR with far fewer parameters
One-hot models were 3× slower and less accurate

### Interpretation:

Translation requires strong sequence memory (encoder–decoder)
GRU preserves semantic information through gating
RNN fails due to information bottleneck collapse
GloVe provides semantic priors, improving learning efficiency

### 6. Model Comparisons
### Task 1 vs Task 2
### Within Task 1 — Text Generation:

GRU + GloVe and GRU + OneHot performed nearly identically in perplexity (239.54 vs 237.44), suggesting that for language modeling on WikiText-2 the embedding type matters less than the architecture
GRU models outperformed RNN models by roughly 80 perplexity points, which was the largest performance gap observed in Task 1
OneHot models had 2x more parameters than their GloVe counterparts but delivered no meaningful performance improvement, indicating wasted capacity
RNN + GloVe and RNN + OneHot also performed similarly to each other (321.52 vs 325.14), reinforcing that the architectural choice dominated the embedding choice in this task
Convergence speed differed notably — GRU + OneHot converged at epoch 7 while all other models needed all 10 epochs, suggesting the higher parameter count helped it find a solution faster even if not a better one
Training times were comparable across all four configurations (4–5 minutes), meaning there was no significant computational cost tradeoff to consider in Task 1

### Within Task 2 — Machine Translation:

GRU + GloVe was the clear standout, outperforming the next best model (GRU + OneHot) by 4.50 BLEU points and 10.00 METEOR points, a much larger gap than seen within Task 1
GRU + OneHot showed that even without semantic embeddings, the GRU architecture alone was sufficient to produce meaningful translations, scoring BLEU 10.00 compared to near-zero for both RNN models
Both RNN configurations collapsed to near-identical translations for all sample sentences regardless of input, while both GRU configurations produced distinct and structurally varied outputs — this is the clearest illustration of the architectural gap in the entire experiment
The parameter count difference between GloVe and OneHot models was far more extreme in Task 2 than Task 1 — 2.8M vs 103M — yet the OneHot models were still outperformed, making Task 2 the stronger demonstration that pre-trained embeddings outperform raw capacity
Training time differences were more significant in Task 2, with OneHot models taking 3x longer than GloVe models (8m vs 2m30s for GRU), making the efficiency argument for GloVe much more compelling here than in Task 1
RNN + GloVe and RNN + OneHot performed almost identically (BLEU 4.50 vs 3.50), suggesting the RNN architecture was the binding constraint and no embedding improvement could compensate for it
 Sonnet 4.6

### Final Interpretation
GRU’s gating mechanism is essential for stable sequence modeling.
Basic RNNs fail in tasks requiring long-term dependency retention.
The performance gap widens as task complexity increases (especially in translation).

## 7. Challenges Faced During Implementation

**Dependency conflicts:** The torchtext library had version compatibility issues with Python 3.12, causing import errors related to mismatched compiled library symbols. This was resolved by removing the torchtext dependency entirely and implementing dataset loading using Python's built-in `urllib` and `gzip` modules, downloading raw data files directly from their source repositories.

**One-hot memory usage:** One-hot embedding matrices have dimensions of vocab_size × vocab_size, which for WikiText-2 (vocabulary of ~10,000 tokens) produces a matrix of roughly 10,000 × 10,000 floats. This is memory-intensive and significantly slows down training compared to the compact GloVe matrices. On machines with limited RAM this can cause slowdowns or out-of-memory errors.

**Vanishing gradients in Basic RNN:** The basic RNN showed instability during early training epochs, particularly on longer sequences in WikiText-2. Gradient clipping with a threshold of 1.0 was applied to prevent exploding gradients, which stabilized training, but the RNN still showed slower convergence than the GRU.

**Teacher forcing decay in Task 2:** Early translation experiments showed that using a fixed teacher forcing ratio led to models that performed well during training but poorly at inference, since they had learned to rely on ground truth inputs rather than their own predictions. Gradually reducing the teacher forcing ratio by 0.03 per epoch addressed this exposure bias and improved test-time performance.

**GloVe download size:** The GloVe 6B embeddings zip file is approximately 860 MB. On slower connections this can be a significant wait on first run. The file is cached locally after the first download so subsequent runs are unaffected.


## 8. Limitations of the Considered Models

**No attention mechanism:** Both the GRU and RNN seq2seq models compress the entire source sentence into a single fixed-size hidden state vector before decoding. This creates an information bottleneck that becomes increasingly problematic for longer sentences, as the decoder must reconstruct the full meaning of the source from a single vector. Attention mechanisms, which allow the decoder to look back at all encoder hidden states at each decoding step, directly address this limitation and are standard in modern translation systems.

**No beam search:** At inference time, both models use greedy decoding — always selecting the most probable next token. This is fast but suboptimal, as the globally best sequence may not be found by greedily picking the locally best token at each step. Beam search maintains multiple candidate sequences simultaneously and typically improves BLEU scores by several points without any additional training.

**Fixed context window for language modeling:** The language model processes text in fixed-length chunks of 35 tokens. This means the model cannot attend to context beyond the chunk boundary, which limits its ability to model long-range dependencies even if the GRU hidden state partially carries information across chunks.

**Basic RNN gradient issues:** The basic RNN is fundamentally limited in its ability to capture long-range dependencies due to the vanishing gradient problem. Even with gradient clipping, it struggles to learn relationships between tokens that are many steps apart, which is a structural limitation rather than a tuning issue.

**Vocabulary coverage:** Words appearing fewer than the minimum frequency threshold are mapped to the unknown token. For Multi30K, which is a relatively small dataset, this means a non-trivial fraction of tokens in the test set may be unknown to the model, particularly for morphologically complex German words.


## 9. Possible Future Improvements

**Add attention mechanism:** Implementing Bahdanau or Luong attention in the seq2seq decoder would allow the model to dynamically focus on relevant parts of the source sentence at each decoding step, directly addressing the information bottleneck of the fixed hidden state. This is the single most impactful improvement available for Task 2.

**Use beam search at inference:** Replacing greedy decoding with beam search (beam size 4 or 5) would improve translation quality without any changes to the model or training procedure. This is a low-effort, high-impact improvement for BLEU scores.

**Subword tokenization:** Replacing word-level tokenization with byte-pair encoding (BPE) or SentencePiece would reduce the unknown token rate, improve handling of rare and morphologically complex words, and produce smaller vocabularies with better coverage. This would be particularly beneficial for German in Task 2.

**Larger pre-trained embeddings:** Experimenting with higher-dimensional GloVe vectors (200d or 300d) or more recent contextualized embeddings such as fastText could improve the quality of word representations and boost downstream task performance.

**Hyperparameter search:** The current models use fixed hyperparameters chosen by convention. A systematic grid search or random search over hidden dimension, number of layers, dropout rate, and learning rate could identify configurations that meaningfully improve performance on both tasks.
