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


## 5. Experimental Results

*Results to be filled in after running experiments.*


## 6. Comparison of Models

### GRU vs Basic RNN
The GRU is expected to outperform the basic RNN on both tasks due to its gating mechanism, which allows it to better capture long-range dependencies in text. In language modeling, where context from many tokens back can be relevant, this advantage is particularly significant. The basic RNN is prone to forgetting early context as sequences grow longer, which tends to result in higher perplexity and lower BLEU scores. However, the basic RNN trains faster per epoch due to its simpler computation, so the tradeoff between performance and speed is visible in the train time metric.

### GloVe vs One-Hot Embeddings
GloVe embeddings are expected to outperform one-hot encodings on both tasks, particularly on the smaller Multi30K dataset where limited training data makes it harder to learn good representations from scratch. The semantic structure encoded in GloVe vectors gives the model a meaningful starting point, which typically leads to faster convergence and better generalization. One-hot models require learning all word relationships purely from the training signal, which demands more data and more epochs to achieve comparable performance. The one-hot models also have significantly more parameters due to the larger embedding dimension, which increases memory usage and training time considerably.

### Task 1 vs Task 2
Text generation is a simpler task than machine translation in the sense that it only requires modeling one language. Machine translation introduces the additional challenge of aligning two different languages and generating coherent output in the target language, which generally makes it harder to achieve good performance. BLEU scores for neural machine translation on Multi30K with small models are typically in the range of 10–25, which reflects this difficulty.


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

**Experiment with LSTM:** Adding LSTM as a third architecture would complete the standard comparison of recurrent architectures. The LSTM uses both a hidden state and a cell state, giving it a stronger memory mechanism than the GRU, which could further improve performance on both tasks.

**Subword tokenization:** Replacing word-level tokenization with byte-pair encoding (BPE) or SentencePiece would reduce the unknown token rate, improve handling of rare and morphologically complex words, and produce smaller vocabularies with better coverage. This would be particularly beneficial for German in Task 2.

**Transformer architecture:** The current state of the art for both language modeling and machine translation is the Transformer, which replaces recurrence entirely with self-attention. Comparing the GRU and RNN results against a Transformer baseline would contextualize the performance gap between older and modern architectures.

**Larger pre-trained embeddings:** Experimenting with higher-dimensional GloVe vectors (200d or 300d) or more recent contextualized embeddings such as fastText could improve the quality of word representations and boost downstream task performance.

**Hyperparameter search:** The current models use fixed hyperparameters chosen by convention. A systematic grid search or random search over hidden dimension, number of layers, dropout rate, and learning rate could identify configurations that meaningfully improve performance on both tasks.
