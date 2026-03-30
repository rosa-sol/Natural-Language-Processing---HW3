"""
t2_main.py
Entry point for Task 2: English -> German translation on Multi30K.

Runs all four experiment combinations:
  1. GRU  + GloVe embeddings
  2. GRU  + One-Hot embeddings
  3. RNN  + GloVe embeddings
  4. RNN  + One-Hot embeddings

Metrics reported:
  - BLEU
  - METEOR
  - TER
  - Precision & Recall
  - Train Time
  - Number of Parameters
  - Convergence Speed
"""

import torch

from t2_dataset import (
    build_vocabs, get_dataloaders,
    build_glove_matrix, build_onehot_matrix, GLOVE_DIM,
)
from t2_architecture import build_gru_seq2seq, build_rnn_seq2seq
from t2_train import run_training
from t2_eval import (
    compute_all_metrics, compute_convergence_speed,
    count_parameters, translate_sentence, print_eval_report
)


# ── hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN_DIM       = 256
NUM_LAYERS       = 2
DROPOUT          = 0.3
NUM_EPOCHS       = 15
LR               = 1e-3
BATCH_SIZE       = 128
CLIP             = 1.0
TEACHER_FORCING  = 0.5

SAMPLE_SENTENCES = [
    "a dog is running in the park .",
    "two children are playing outside .",
    "the man is wearing a red shirt .",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── load data ─────────────────────────────────────────────────────────────────
src_vocab, tgt_vocab = build_vocabs()
train_loader, val_loader, test_loader = get_dataloaders(
    src_vocab, tgt_vocab, batch_size=BATCH_SIZE
)

SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)

# ── embeddings ────────────────────────────────────────────────────────────────
print("\nBuilding GloVe matrices ...")
src_glove = build_glove_matrix(src_vocab, GLOVE_DIM)
tgt_glove = build_glove_matrix(tgt_vocab, GLOVE_DIM)

print("Building One-Hot matrices ...")
src_onehot = build_onehot_matrix(SRC_VOCAB_SIZE)
tgt_onehot = build_onehot_matrix(TGT_VOCAB_SIZE)


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(builder_fn, src_emb, tgt_emb, embed_type, arch_name):
    src_embed_dim = src_emb.size(1)
    tgt_embed_dim = tgt_emb.size(1)
    save_path = f"best_t2_{arch_name}_{embed_type}.pt"

    print(f"\n{'#'*65}")
    print(f"  Experiment: {arch_name} + {embed_type}  "
          f"(src_emb={src_embed_dim}, tgt_emb={tgt_embed_dim}, hidden={HIDDEN_DIM})")
    print(f"{'#'*65}")

    model = builder_fn(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        src_embed_dim=src_embed_dim,
        tgt_embed_dim=tgt_embed_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=DEVICE,
        src_pretrained=src_emb.clone(),
        tgt_pretrained=tgt_emb.clone(),
    )

    num_params = count_parameters(model)
    print(f"  Trainable parameters: {num_params:,}")

    train_losses, val_losses, train_time = run_training(
        model, train_loader, val_loader, DEVICE,
        num_epochs=NUM_EPOCHS, lr=LR, clip=CLIP,
        teacher_forcing_ratio=TEACHER_FORCING,
        save_path=save_path,
    )

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    bleu, meteor, ter, precision, recall = compute_all_metrics(
        model, test_loader, src_vocab, tgt_vocab, DEVICE
    )
    best_epoch = compute_convergence_speed(val_losses)

    examples = []
    for sent in SAMPLE_SENTENCES:
        hyp = translate_sentence(model, sent, src_vocab, tgt_vocab, DEVICE)
        examples.append((sent, "--", " ".join(hyp)))

    print_eval_report(
        arch_name, embed_type, train_losses, val_losses,
        bleu, meteor, ter, precision, recall,
        train_time, num_params, examples
    )

    return {
        "bleu":       bleu,
        "meteor":     meteor,
        "ter":        ter,
        "precision":  precision,
        "recall":     recall,
        "train_time": train_time,
        "params":     num_params,
        "best_epoch": best_epoch,
    }


# ── run all four combinations ─────────────────────────────────────────────────

results = {}
results[("GRU", "GloVe")]  = run_experiment(build_gsru_seq2seq, src_glove,  tgt_glove,  "GloVe",  "GRU")
results[("GRU", "OneHot")] = run_experiment(build_gru_seq2seq, src_onehot, tgt_onehot, "OneHot", "GRU")
results[("RNN", "GloVe")]  = run_experiment(build_rnn_seq2seq, src_glove,  tgt_glove,  "GloVe",  "RNN")
results[("RNN", "OneHot")] = run_experiment(build_rnn_seq2seq, src_onehot, tgt_onehot, "OneHot", "RNN")


# ── final comparison table ────────────────────────────────────────────────────

print("\n" + "="*95)
print("  FINAL COMPARISON -- Task 2 Machine Translation")
print("="*95)
print(f"  {'Arch':<6} {'Embed':<8} {'BLEU':>6} {'METEOR':>7} {'TER':>6} "
      f"{'Prec':>7} {'Rec':>7} {'Params':>10} {'BestEp':>7} {'Time':>10}")
print(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6} "
      f"{'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*10}")

for (arch, emb), r in results.items():
    mins, secs = divmod(int(r["train_time"]), 60)
    print(f"  {arch:<6} {emb:<8} {r['bleu']:>6.2f} {r['meteor']:>7.2f} {r['ter']:>6.2f} "
          f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['params']:>10,} "
          f"{r['best_epoch']:>7} {mins:>4}m{secs:02d}s")
print("="*95)
