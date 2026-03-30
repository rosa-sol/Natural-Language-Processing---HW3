"""
t1_eval.py
Evaluation utilities for Task 1:
  - perplexity on a held-out split
  - qualitative text generation given a seed prompt
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from t1_dataset import get_batch, SEQ_LEN, tokenize
from t1_train import repackage_hidden


# ── perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    seq_len: int = SEQ_LEN,
) -> float:
    """
    Compute perplexity = exp(average cross-entropy loss) on the given data.

    Lower perplexity → better model.

    Returns
    -------
    perplexity : float
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(data.size(1), device)
    num_batches = 0

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, seq_len):
            x, y = get_batch(data, i, seq_len)
            x, y = x.to(device), y.to(device)
            hidden = repackage_hidden(hidden)
            logits, hidden = model(x, hidden)
            loss = criterion(logits, y.view(-1))
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(avg_loss)
    return perplexity


# ── text generation ───────────────────────────────────────────────────────────

def generate_text(
    model: nn.Module,
    vocab,
    seed_text: str,
    device: torch.device,
    num_words: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    """
    Given a seed sentence, generate num_words additional tokens.

    Parameters
    ----------
    model       : trained language model
    vocab       : torchtext Vocab object
    seed_text   : str  - starting prompt
    device      : torch.device
    num_words   : int  - number of tokens to generate
    temperature : float - >1 → more random, <1 → more conservative
    top_k       : int   - if >0, restrict sampling to the top-k logits

    Returns
    -------
    generated : str - seed + generated tokens joined by spaces
    """
    model.eval()
    itos = vocab.get_itos()   # index → token string

    # encode seed
    tokens = tokenize(seed_text)
    if not tokens:
        tokens = ["<bos>"]
    indices = [vocab[t] for t in tokens]

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(device)
    hidden = model.init_hidden(1, device)

    generated_tokens = list(tokens)

    with torch.no_grad():
        _, hidden = model(input_tensor, hidden)
        current = input_tensor[-1].unsqueeze(0)

        for _ in range(num_words):
            logits, hidden = model(current, hidden)
            logits = logits.squeeze(0) / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                threshold = values[-1]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            generated_tokens.append(itos[next_idx])
            current = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return " ".join(generated_tokens)


# ── summary report ────────────────────────────────────────────────────────────

def print_eval_report(
    model_name: str,
    embed_type: str,
    train_losses: list,
    val_losses: list,
    test_perplexity: float,
    generated_samples: list,
):
    """
    Print a formatted evaluation summary.
    """
    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT — {model_name} + {embed_type} embeddings")
    print(f"{'='*65}")
    print(f"  Final train loss : {train_losses[-1]:.4f}  "
          f"(ppl {math.exp(train_losses[-1]):.2f})")
    print(f"  Final val   loss : {val_losses[-1]:.4f}  "
          f"(ppl {math.exp(val_losses[-1]):.2f})")
    print(f"  Test perplexity  : {test_perplexity:.2f}")
    print(f"\n  ── Generated samples ──────────────────────────────────")
    for i, sample in enumerate(generated_samples, 1):
        print(f"\n  [{i}] {sample}")
    print(f"{'='*65}\n")
