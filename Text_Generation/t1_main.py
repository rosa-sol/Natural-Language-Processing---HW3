"""
t1_main.py
Entry point for Task 1: Text Generation on WikiText-2.

Runs all four experiment combinations:
  1. GRU  + GloVe embeddings
  2. GRU  + One-Hot embeddings
  3. RNN  + GloVe embeddings
  4. RNN  + One-Hot embeddings

Metrics reported:
  - Perplexity
  - Top-5 Accuracy
  - Precision & Recall
  - Train Time
  - Number of Parameters
  - Convergence Speed
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
from t1_eval import (
    compute_perplexity,
    compute_topk_accuracy,
    compute_precision_recall,
    compute_convergence_speed,
    count_parameters,
    generate_text,
    print_eval_report,
)

# ── hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_EPOCHS  = 10
LR          = 1e-3
BATCH_SIZE  = 32
CLIP        = 1.0
TOPK        = 5

# one-hot projected dimension — keeps matrix manageable
ONE_HOT_DIM = 500

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

train_data = train_data.to(DEVICE)
val_data   = val_data.to(DEVICE)
test_data  = test_data.to(DEVICE)


# ── build embedding matrices ──────────────────────────────────────────────────
print("\nBuilding GloVe embedding matrix ...")
glove_matrix = build_glove_matrix(vocab, glove_dim=GLOVE_DIM)

print("Building One-Hot (projected) embedding matrix ...")
# project one-hot down to ONE_HOT_DIM so training is feasible
onehot_matrix = torch.randn(VOCAB_SIZE, ONE_HOT_DIM) * 0.01


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(model_class, embed_matrix, embed_type: str, arch_name: str):
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
        freeze_emb=False,
    ).to(DEVICE)

    num_params = count_parameters(model)
    print(f"  Trainable parameters: {num_params:,}")

    # train
    train_losses, val_losses, train_time = run_training(
        model, train_data, val_data, DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        clip=CLIP,
        save_path=save_path,
    )

    # load best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    # metrics
    test_ppl          = compute_perplexity(model, test_data, DEVICE)
    topk_acc          = compute_topk_accuracy(model, test_data, DEVICE, k=TOPK)
    precision, recall = compute_precision_recall(model, test_data, DEVICE)
    best_epoch        = compute_convergence_speed(val_losses)

    # qualitative generation
    samples = [
        generate_text(model, vocab, prompt, DEVICE, num_words=30,
                      temperature=0.8, top_k=40)
        for prompt in SEED_PROMPTS
    ]

    print_eval_report(
        arch_name, embed_type, train_losses, val_losses,
        test_ppl, topk_acc, precision, recall,
        train_time, num_params, samples,
    )

    return {
        "ppl":        test_ppl,
        "topk_acc":   topk_acc,
        "precision":  precision,
        "recall":     recall,
        "train_time": train_time,
        "params":     num_params,
        "best_epoch": best_epoch,
    }


# ── run all four combinations ─────────────────────────────────────────────────

results = {}
results[("GRU", "GloVe")]  = run_experiment(GRULanguageModel, glove_matrix,  "GloVe",  "GRU")
results[("GRU", "OneHot")] = run_experiment(GRULanguageModel, onehot_matrix, "OneHot", "GRU")
results[("RNN", "GloVe")]  = run_experiment(RNNLanguageModel, glove_matrix,  "GloVe",  "RNN")
results[("RNN", "OneHot")] = run_experiment(RNNLanguageModel, onehot_matrix, "OneHot", "RNN")


# ── final comparison table ────────────────────────────────────────────────────

print("\n" + "="*85)
print("  FINAL COMPARISON -- Task 1 Text Generation (lower PPL is better)")
print("="*85)
print(f"  {'Arch':<6} {'Embed':<8} {'PPL':>7} {'Top5Acc':>8} "
      f"{'Prec':>7} {'Rec':>7} {'Params':>12} {'BestEp':>7} {'Time':>10}")
print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*12} {'-'*7} {'-'*10}")

for (arch, emb), r in results.items():
    mins, secs = divmod(int(r["train_time"]), 60)
    print(
        f"  {arch:<6} {emb:<8} {r['ppl']:>7.2f} {r['topk_acc']*100:>7.2f}% "
        f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['params']:>12,} "
        f"{r['best_epoch']:>7} {mins:>4}m{secs:02d}s"
    )
print("="*85)
