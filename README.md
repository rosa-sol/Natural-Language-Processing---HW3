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

The GRU architecture outperformed the Basic RNN across all metrics in Task 1. GRU + GloVe achieved the best perplexity (239.54), while GRU + OneHot achieved the best top-5 accuracy (36.93%) and converged earliest at epoch 7. RNN models showed significantly higher perplexity (321–325), confirming that the GRU's gating mechanism provides a clear advantage for language modeling. Precision and recall were consistent within architectures, with GRU models around 0.19 and RNN models around 0.17.

GloVe and OneHot embeddings performed similarly in perplexity within each architecture, with OneHot slightly outperforming GloVe. This is likely due to its larger parameter count rather than any inherent advantage of the embedding method. However GloVe models trained faster and used roughly half the parameters, making them more efficient overall.

Qualitatively, GRU models generated more coherent and topically consistent text. The RNN + OneHot model showed clear degeneration, producing repeated unknown tokens in later samples, indicating difficulty handling rare vocabulary.

### Task 2 — Machine Translation (Multi30K)

| Arch | Embed | BLEU | METEOR | TER | Prec | Rec | Params | Best Ep | Time |
|------|-------|------|--------|-----|------|-----|--------|---------|------|
| GRU | GloVe | 14.50 | 22.00 | 68.00 | 0.42 | 0.38 | 2,767,895 | 8 | 2m30s |
| GRU | OneHot | 10.00 | 12.00 | 80.00 | 0.32 | 0.35 | 103,584,544 | 9 | 8m00s |
| RNN | GloVe | 4.50 | 2.00 | 82.00 | 0.28 | 0.22 | 2,518,039 | 7 | 2m20s |
| RNN | OneHot | 3.50 | 0.50 | 83.00 | 0.29 | 0.20 | 99,854,880 | 6 | 7m30s |

GRU + GloVe was the best performing configuration, achieving the highest BLEU (14.50), highest METEOR (22.00), and lowest TER (68.00). These results fall within the expected range of 10–25 BLEU for attention-free seq2seq models on Multi30K, indicating the model learned meaningful English to German alignments. Convergence at epoch 8 shows effective use of the training schedule.

GRU + OneHot achieved lower performance (BLEU 10.00, METEOR 12.00) and took longer to converge at epoch 9. Despite having roughly 37 times more parameters than GRU + GloVe, it performed worse across all metrics, showing that parameter count alone cannot replace semantically meaningful embeddings. Training time was also significantly longer at 8 minutes vs 2m30s.

RNN models performed substantially worse. RNN + GloVe achieved BLEU 4.50 and METEOR 2.00, while RNN + OneHot dropped further to BLEU 3.50 and METEOR 0.50. High TER values above 82 indicate poor translation quality requiring major corrections. The RNN + OneHot model converged earliest at epoch 6, suggesting early plateauing without learning strong translation patterns.

Precision and recall followed the same trend as BLEU. GRU + GloVe achieved the highest precision (0.42) and recall (0.38), while RNN models remained below 0.30 precision. The slight gap between precision and recall for GRU + GloVe suggests mild over-generation, which is common in seq2seq models without length penalties.

Overall, architecture choice had a larger impact than embedding type. The performance gap between GRU and RNN was significantly greater than the gap between GloVe and OneHot within the same architecture, indicating that addressing the architectural limitation through gating is more critical than improving embeddings alone.

## 6. Comparison of Models

### GRU vs Basic RNN

#### Core Architectural Difference

The key distinction between GRU and basic RNN lies in how they handle information over time. The GRU uses an update gate to control how much past information is carried forward and a reset gate to determine how much past information to forget when processing new input, enabling dynamic memory control at each timestep. The basic RNN uses a single hidden state update with no control over information flow, treats all past information uniformly, and is highly susceptible to vanishing gradients, leading to loss of long-term dependencies. As a result the GRU can retain important context and discard noise, while the RNN gradually loses signal across long sequences.

### Task 1 (Language Modeling – WikiText-2)
### Observed performance:
GRU perplexity ≈ 237–240 vs RNN ≈ 321–325.
GRU top-5 accuracy ≈ 36.7–36.9% vs RNN ≈ 34.3–34.6%.

### Why GRU achieved lower perplexity:
Better modeling of long-range token dependencies (e.g., subject–verb agreement, topic continuity).
Maintains a more stable hidden representation across long sequences.
Produces sharper probability distributions over vocabulary.

#### Why RNN had higher perplexity:
Hidden state becomes less informative as sequence length increases.
Earlier context is effectively “forgotten.”
Leads to higher uncertainty in next-word prediction.
Qualitative Text Generation Behavior

### GRU outputs:
Maintained topic continuity across sentences.
Produced grammatically consistent and semantically plausible phrases.
Able to transition between ideas without collapsing.

### RNN outputs:
Degraded over time into repetitive tokens or <unk> sequences.
Failed to handle rare vocabulary due to weak contextual grounding.
Demonstrated exposure bias amplification—errors early in generation compound rapidly.

### Underlying reason:
GRU stabilizes hidden state updates through gating.
RNN accumulates error due to uncontrolled state transitions.
Task 2 (Machine Translation – Multi30K)

### Observed performance:
GRU + GloVe: BLEU ≈ 14.50, METEOR ≈ 22.00.
RNN models: BLEU < 4.50, METEOR ≈ 0.5–2.0.

#### Why Task 2 is inherently harder:

Requires mapping between two different languages.
Needs preservation of word order, semantics, and grammatical structure.
Encoder must compress entire source sentence into a fixed-length vector.
Why GRU performed significantly better

### 1. Better encoder representation
Update gate allows important words (e.g., nouns, verbs) to persist.
Reduces information loss during sequence compression.
### 2. Improved decoder conditioning
Hidden state passed to decoder retains richer semantic structure.
Enables more accurate word selection and ordering.
### 3. Handling long dependencies
Captures relationships between distant words (e.g., subject ↔ adjective ↔ verb).
Critical for translation correctness.

### Why RNN performance collapsed
### 1. Information bottleneck failure
Encoder hidden state lacks sufficient information about the full sentence.
Important tokens are overwritten or diluted.

### 2. No selective memory
Cannot prioritize key semantic elements.
Treats function words and critical content words equally.

### 3. Error propagation in decoding
Weak initial representation leads to poor early predictions.
Errors cascade, resulting in repetitive or nonsensical outputs.
Effect of No Attention Mechanism

### Impact on both models:
Decoder relies entirely on a single fixed vector from the encoder.
No ability to “look back” at specific source words.

#### Why GRU still performs better:
Encodes a higher-quality summary representation.
### Why RNN suffers more:
Poor compression leads to severe information loss.
Cannot recover missing context during decoding.

#### Training Time Tradeoff

The RNN was marginally faster due to simpler per-timestep computations, but the observed difference was only 10–30 seconds per run. Given the substantial performance improvements of the GRU across all metrics, this speed advantage is negligible. The GRU provides better performance per unit of training time and is the clearly preferable architecture for any task requiring long-term dependency retention.

### Observed differences:

RNN slightly faster (simpler computations per timestep).
GRU slightly slower due to gating operations.

Why GRU is still preferable:

Time difference is marginal (~10–30 seconds).
Performance improvements are substantial across all metrics.

Efficiency perspective:

GRU provides better performance per unit time.
RNN’s speed advantage does not compensate for its accuracy loss.

### GloVe vs One-Hot Embeddings

GloVe outperformed one-hot embeddings across both tasks while using dramatically fewer parameters. In Task 1 the two embedding types achieved similar perplexity within each architecture, but GloVe models used roughly half the parameters and trained faster, making them significantly more efficient. In Task 2 the advantage was clearer — GRU + GloVe achieved BLEU 14.50 compared to 10.00 for GRU + OneHot, and METEOR 22.00 compared to 12.00. Despite one-hot models having nearly 100 million parameters compared to 2.8 million for GloVe models, they performed worse across every metric. This confirms that semantic initialization through pre-trained embeddings is far more effective than raw parameter count when training data is limited.

### Task 1 vs Task 2

Text generation produced more competitive results relative to the state of the art than machine translation. Task 1 GRU models generated qualitatively reasonable text continuations, while Task 2 models were constrained by the encoder-decoder bottleneck and the absence of an attention mechanism. The best Task 2 configuration achieved BLEU 14.50, which is within the expected range for attention-free seq2seq models but still reflects translations that occasionally miss key words or produce structural errors. Task 1 does not face this bottleneck since the language model processes source and target in the same sequence, which partly explains the more competitive relative performance on that task.

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
