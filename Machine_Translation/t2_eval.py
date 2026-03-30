"""
t2_eval.py
Evaluation utilities for Task 2 machine translation.
Metrics:
  - BLEU (corpus-level)
  - METEOR
  - TER (Translation Edit Rate)
  - Token-level Precision & Recall
  - Convergence Speed
  - Train Time
  - Number of Parameters
"""

import math
import torch
from collections import Counter
from t2_dataset import PAD_IDX, BOS_IDX, EOS_IDX, tokenize


# ── greedy translation ────────────────────────────────────────────────────────

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    itos    = tgt_vocab.get_itos()
    tokens  = tokenize(sentence)
    src_ids = [BOS_IDX] + [src_vocab[t] for t in tokens] + [EOS_IDX]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device)

    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)

    translated = []
    dec_input  = torch.tensor([[BOS_IDX]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model.decoder(dec_input, hidden)
            next_idx = logits.argmax(dim=-1).item()
            if next_idx == EOS_IDX:
                break
            translated.append(itos[next_idx])
            dec_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return translated


# ── BLEU ──────────────────────────────────────────────────────────────────────

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(candidates, references, max_n=4):
    clipped = [0] * max_n
    total   = [0] * max_n
    cand_len = ref_len = 0

    for cand, ref in zip(candidates, references):
        cand_len += len(cand)
        ref_len  += len(ref)
        for n in range(1, max_n + 1):
            cng = _ngram_counts(cand, n)
            rng = _ngram_counts(ref,  n)
            for ng, cnt in cng.items():
                clipped[n-1] += min(cnt, rng.get(ng, 0))
            total[n-1] += max(len(cand) - n + 1, 0)

    if cand_len == 0:
        return 0.0
    bp = 1.0 if cand_len >= ref_len else math.exp(1 - ref_len / cand_len)
    log_avg = 0.0
    for n in range(max_n):
        if clipped[n] == 0 or total[n] == 0:
            return 0.0
        log_avg += math.log(clipped[n] / total[n])
    return bp * math.exp(log_avg / max_n) * 100


# ── METEOR ────────────────────────────────────────────────────────────────────

def _meteor_sentence(cand, ref):
    """
    Unigram METEOR (no stemming/synonyms for simplicity).
    F_mean = 10*P*R / (9*P + R), score = F_mean * (1 - penalty)
    """
    if not cand or not ref:
        return 0.0
    ref_counts = Counter(ref)
    matched    = 0
    for token in cand:
        if ref_counts.get(token, 0) > 0:
            matched += 1
            ref_counts[token] -= 1
    if matched == 0:
        return 0.0
    P = matched / len(cand)
    R = matched / len(ref)
    F = 10 * P * R / (9 * P + R + 1e-12)
    # chunk penalty
    chunks, in_chunk = 1, False
    ref_set = set(ref)
    for token in cand:
        if token in ref_set and not in_chunk:
            chunks += 1
            in_chunk = True
        elif token not in ref_set:
            in_chunk = False
    penalty = 0.5 * (chunks / max(matched, 1)) ** 3
    return F * (1 - penalty)


def compute_meteor(candidates, references):
    scores = [_meteor_sentence(c, r) for c, r in zip(candidates, references)]
    return sum(scores) / max(len(scores), 1) * 100


# ── TER ───────────────────────────────────────────────────────────────────────

def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def compute_ter(candidates, references):
    """
    TER = total edits / total reference length  (lower is better)
    Reported as a percentage.
    """
    total_edits = total_ref = 0
    for cand, ref in zip(candidates, references):
        total_edits += _edit_distance(cand, ref)
        total_ref   += max(len(ref), 1)
    return (total_edits / max(total_ref, 1)) * 100


# ── Precision & Recall ────────────────────────────────────────────────────────

def compute_precision_recall(candidates, references):
    """Token-level unigram precision and recall across the corpus."""
    tp = pred_total = ref_total = 0
    for cand, ref in zip(candidates, references):
        ref_counts  = Counter(ref)
        cand_counts = Counter(cand)
        for token, cnt in cand_counts.items():
            tp += min(cnt, ref_counts.get(token, 0))
        pred_total += len(cand)
        ref_total  += len(ref)
    precision = tp / max(pred_total, 1)
    recall    = tp / max(ref_total,  1)
    return precision, recall


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_convergence_speed(val_losses):
    return int(val_losses.index(min(val_losses))) + 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _collect_translations(model, loader, src_vocab, tgt_vocab, device, max_len=50):
    itos_src = src_vocab.get_itos()
    itos_tgt = tgt_vocab.get_itos()
    candidates, references = [], []
    model.eval()
    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            B = src_batch.size(1)
            for b in range(B):
                src_tokens = [itos_src[i.item()] for i in src_batch[:, b]
                              if i.item() not in (PAD_IDX, BOS_IDX, EOS_IDX)]
                pred = translate_sentence(
                    model, " ".join(src_tokens),
                    src_vocab, tgt_vocab, device, max_len
                )
                ref = [itos_tgt[i.item()] for i in tgt_batch[:, b]
                       if i.item() not in (PAD_IDX, BOS_IDX, EOS_IDX)]
                candidates.append(pred)
                references.append(ref)
    return candidates, references


def compute_all_metrics(model, loader, src_vocab, tgt_vocab, device, max_len=50):
    """Compute BLEU, METEOR, TER, Precision, Recall in one pass."""
    cands, refs = _collect_translations(model, loader, src_vocab, tgt_vocab, device, max_len)
    bleu      = _corpus_bleu(cands, refs)
    meteor    = compute_meteor(cands, refs)
    ter       = compute_ter(cands, refs)
    prec, rec = compute_precision_recall(cands, refs)
    return bleu, meteor, ter, prec, rec


# ── report ────────────────────────────────────────────────────────────────────

def print_eval_report(model_name, embed_type, train_losses, val_losses,
                      bleu, meteor, ter, precision, recall,
                      train_time_sec, num_params, example_translations):
    best_epoch = compute_convergence_speed(val_losses)
    mins, secs = divmod(int(train_time_sec), 60)

    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT -- {model_name} + {embed_type} embeddings")
    print(f"{'='*65}")
    print(f"  Parameters       : {num_params:,}")
    print(f"  Train time       : {mins}m {secs}s")
    print(f"  Convergence      : best val loss at epoch {best_epoch}/{len(val_losses)}")
    print(f"  Final train loss : {train_losses[-1]:.4f}")
    print(f"  Final val   loss : {val_losses[-1]:.4f}")
    print(f"  BLEU             : {bleu:.2f}")
    print(f"  METEOR           : {meteor:.2f}")
    print(f"  TER              : {ter:.2f}  (lower is better)")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"\n  -- Translation Examples -----------------------------------")
    for src, ref, hyp in example_translations:
        print(f"\n  SRC : {src}")
        print(f"  REF : {ref}")
        print(f"  HYP : {hyp}")
    print(f"{'='*65}\n")
