"""
t1_eval.py
Evaluation utilities for Task 1 text generation.
Metrics:
  - Perplexity
  - Top-k Accuracy
  - Token-level Precision & Recall
  - Bits per Character (BPC)
  - Convergence Speed (best epoch)
  - Train Time
  - Number of Parameters
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from t1_dataset import get_batch, SEQ_LEN, tokenize
from t1_train import repackage_hidden


# ── perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(model, data, device, seq_len=SEQ_LEN):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, num_batches = 0.0, 0
    hidden = model.init_hidden(data.size(1), device)

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, seq_len):
            x, y = get_batch(data, i, seq_len)
            x, y = x.to(device), y.to(device)
            hidden = repackage_hidden(hidden)
            logits, hidden = model(x, hidden)
            total_loss += criterion(logits, y.view(-1)).item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return math.exp(avg_loss)


# ── top-k accuracy ────────────────────────────────────────────────────────────

def compute_topk_accuracy(model, data, device, k=5, seq_len=SEQ_LEN):
    """
    Fraction of tokens where the correct token appears in the
    model's top-k predictions.
    """
    model.eval()
    correct, total = 0, 0
    hidden = model.init_hidden(data.size(1), device)

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, seq_len):
            x, y = get_batch(data, i, seq_len)
            x, y = x.to(device), y.to(device)
            hidden = repackage_hidden(hidden)
            logits, hidden = model(x, hidden)       # (T*B, V)
            topk = logits.topk(k, dim=-1).indices   # (T*B, k)
            y_flat = y.view(-1).unsqueeze(1)        # (T*B, 1)
            correct += topk.eq(y_flat).any(dim=-1).sum().item()
            total   += y_flat.size(0)

    return correct / max(total, 1)


# ── precision & recall ────────────────────────────────────────────────────────

def compute_precision_recall(model, data, device, seq_len=SEQ_LEN):
    """
    Token-level precision and recall computed over the test set.
    Precision = correctly predicted tokens / total predicted tokens
    Recall    = correctly predicted tokens / total reference tokens
    (uses greedy argmax prediction)
    """
    model.eval()
    true_pos, pred_total, ref_total = 0, 0, 0
    hidden = model.init_hidden(data.size(1), device)

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, seq_len):
            x, y = get_batch(data, i, seq_len)
            x, y = x.to(device), y.to(device)
            hidden = repackage_hidden(hidden)
            logits, hidden = model(x, hidden)
            preds   = logits.argmax(dim=-1)   # (T*B,)
            y_flat  = y.view(-1)
            true_pos  += preds.eq(y_flat).sum().item()
            pred_total += preds.size(0)
            ref_total  += y_flat.size(0)

    precision = true_pos / max(pred_total, 1)
    recall    = true_pos / max(ref_total,  1)
    return precision, recall


# ── convergence speed ─────────────────────────────────────────────────────────

def compute_convergence_speed(val_losses: list) -> int:
    """
    Returns the epoch (1-indexed) at which validation loss was lowest.
    """
    return int(val_losses.index(min(val_losses))) + 1


# ── parameter count ───────────────────────────────────────────────────────────

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── text generation ───────────────────────────────────────────────────────────

def generate_text(model, vocab, seed_text, device,
                  num_words=50, temperature=1.0, top_k=0):
    model.eval()
    itos = vocab.get_itos()
    tokens  = tokenize(seed_text) or ["<bos>"]
    indices = [vocab[t] for t in tokens]

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(device)
    hidden = model.init_hidden(1, device)
    generated = list(tokens)

    with torch.no_grad():
        _, hidden = model(input_tensor, hidden)
        current = input_tensor[-1].unsqueeze(0)

        for _ in range(num_words):
            logits, hidden = model(current, hidden)
            logits = logits.squeeze(0) / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < values[-1], float("-inf"))
            probs    = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(itos[next_idx])
            current = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return " ".join(generated)


# ── summary report ────────────────────────────────────────────────────────────

def print_eval_report(model_name, embed_type, train_losses, val_losses,
                      test_perplexity, topk_acc, precision, recall,
                      train_time_sec, num_params, generated_samples):
    best_epoch = compute_convergence_speed(val_losses)
    mins, secs = divmod(int(train_time_sec), 60)

    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT -- {model_name} + {embed_type} embeddings")
    print(f"{'='*65}")
    print(f"  Parameters       : {num_params:,}")
    print(f"  Train time       : {mins}m {secs}s")
    print(f"  Convergence      : best val loss at epoch {best_epoch}/{len(val_losses)}")
    print(f"  Final train loss : {train_losses[-1]:.4f}  "
          f"(ppl {math.exp(train_losses[-1]):.2f})")
    print(f"  Final val   loss : {val_losses[-1]:.4f}  "
          f"(ppl {math.exp(val_losses[-1]):.2f})")
    print(f"  Test perplexity  : {test_perplexity:.2f}")
    print(f"  Top-5 accuracy   : {topk_acc*100:.2f}%")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"\n  -- Generated samples --------------------------------------")
    for i, sample in enumerate(generated_samples, 1):
        print(f"\n  [{i}] {sample}")
    print(f"{'='*65}\n")
