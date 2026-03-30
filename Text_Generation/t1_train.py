"""
t1_train.py
Training loop for Task 1 language models (GRU / RNN).
Now returns train time and tracks convergence speed.
"""

import time
import math
import torch
import torch.nn as nn
from t1_dataset import get_batch, SEQ_LEN


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)


def train_one_epoch(model, train_data, optimizer, criterion, device,
                    clip=1.0, seq_len=SEQ_LEN, log_interval=200):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    hidden = model.init_hidden(train_data.size(1), device)
    num_batches = (train_data.size(0) - 1) // seq_len

    for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
        x, y = get_batch(train_data, i, seq_len)
        x, y = x.to(device), y.to(device)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits, y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(f"  batch {batch_idx+1:>5}/{num_batches} | "
                  f"loss {avg:.4f} | ppl {math.exp(avg):>8.2f} | {elapsed:.1f}s")

    return total_loss / num_batches


def evaluate(model, data, criterion, device, seq_len=SEQ_LEN):
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

    return total_loss / max(num_batches, 1)


def run_training(model, train_data, val_data, device,
                 num_epochs=10, lr=1e-3, clip=1.0,
                 scheduler_patience=2, save_path="best_model.pt"):
    """
    Returns
    -------
    train_losses, val_losses, total_train_time_seconds
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=scheduler_patience, factor=0.5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    total_start   = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}  |  lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        t_loss = train_one_epoch(model, train_data, optimizer, criterion, device, clip)
        v_loss = evaluate(model, val_data, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"\n  > Train loss: {t_loss:.4f}  ppl: {math.exp(t_loss):.2f}"
              f"  |  Val loss: {v_loss:.4f}  ppl: {math.exp(v_loss):.2f}")

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Best model saved -> {save_path}")

    total_time = time.time() - total_start
    return train_losses, val_losses, total_time
