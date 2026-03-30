"""
t1_main.py
Entry point for Task 1: Text Generation on WikiText-2.

Runs all four experiment combinations:
  1. GRU  + GloVe embeddings
  2. GRU  + One-Hot embeddings
  3. RNN  + GloVe embeddings
  4. RNN  + One-Hot embeddings

Results (perplexity + generated text) are printed at the end of each run
and summarised in a comparison table at the very end.
"""

import math
import torch

from t1_dataset import (
    load_data,
    build_glove_matrix,
    build_onehot_matrix,
    GLOVE_DIM,
)
from t1_architecture import GRULanguageModel, RNNLanguageModel
from t1_train import run_training
from t1_eval import compute_perplexity, generate_text, print_eval_report


# ── hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_EPOCHS   = 10
LR           = 1e-3
BATCH_SIZE   = 32
CLIP         = 1.0

# seed prompts used for qualitative evaluation
SEED_PROMPTS = [
    "the president of the united states",
    "scientists have recently discovered",
    "in the early nineteenth century",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── load data ─────────────────────────────────────────────────────────────────
vocab, train_data, val_data, test_data = load_data(batch_size=BATCH_SIZE)
VOCAB_SIZE = len(vocab)

# move data tensors to device
train_data = train_data.to(DEVICE)
val_data   = val_data.to(DEVICE)
test_data  = test_data.to(DEVICE)


# ── build embedding matrices ──────────────────────────────────────────────────
print("\nBuilding GloVe embedding matrix …")
glove_matrix = build_glove_matrix(vocab, glove_dim=GLOVE_DIM)

print("Building One-Hot embedding matrix …")
onehot_matrix = build_onehot_matrix(VOCAB_SIZE)


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(model_class, embed_matrix, embed_type: str, arch_name: str):
    """
    Instantiate, train, and evaluate one model configuration.

    Returns
    -------
    test_ppl : float
    """
    embed_dim = embed_matrix.size(1)
    save_path = f"best_{arch_name}_{embed_type}.pt"

    print(f"\n{'#'*65}")
    print(f"  Experiment: {arch_name} + {embed_type}  "
          f"(embed_dim={embed_dim}, hidden={HIDDEN_DIM})")
    print(f"{'#'*65}")

    model = model_class(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_emb=embed_matrix.clone(),
        freeze_emb=False,         # fine-tune embeddings during training
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # train
    train_losses, val_losses = run_training(
        model, train_data, val_data, DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        clip=CLIP,
        save_path=save_path,
    )

    # load best checkpoint for evaluation
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    # test perplexity
    test_ppl = compute_perplexity(model, test_data, DEVICE)

    # qualitative generation
    samples = [
        generate_text(model, vocab, prompt, DEVICE, num_words=30, temperature=0.8, top_k=40)
        for prompt in SEED_PROMPTS
    ]

    print_eval_report(arch_name, embed_type, train_losses, val_losses, test_ppl, samples)

    return test_ppl


# ── run all four combinations ─────────────────────────────────────────────────

results = {}

results[("GRU", "GloVe")]  = run_experiment(GRULanguageModel, glove_matrix,  "GloVe",  "GRU")
results[("GRU", "OneHot")] = run_experiment(GRULanguageModel, onehot_matrix, "OneHot", "GRU")
results[("RNN", "GloVe")]  = run_experiment(RNNLanguageModel, glove_matrix,  "GloVe",  "RNN")
results[("RNN", "OneHot")] = run_experiment(RNNLanguageModel, onehot_matrix, "OneHot", "RNN")


# ── final comparison table ────────────────────────────────────────────────────

print("\n" + "="*55)
print("  FINAL COMPARISON — Test Perplexity (lower is better)")
print("="*55)
print(f"  {'Architecture':<12} {'Embedding':<10} {'Test PPL':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*10}")
for (arch, emb), ppl in results.items():
    print(f"  {arch:<12} {emb:<10} {ppl:>10.2f}")
print("="*55)

