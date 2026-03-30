"""
t2_train.py
Training loop for Task 2 Seq2Seq translation models.
Now returns total training time.
"""

import time
import torch
import torch.nn as nn
from t2_dataset import PAD_IDX


def train_one_epoch(model, loader, optimizer, criterion, device,
                    clip=1.0, teacher_forcing_ratio=0.5, log_interval=100):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        outputs  = model(src, tgt, teacher_forcing_ratio)
        out_flat = outputs[1:].reshape(-1, outputs.shape[2])
        tgt_flat = tgt[1:].reshape(-1)
        loss = criterion(out_flat, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"  batch {batch_idx+1:>4}/{len(loader)} | "
                  f"loss {avg:.4f} | {time.time()-start_time:.1f}s")

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs  = model(src, tgt, teacher_forcing_ratio=0.0)
            out_flat = outputs[1:].reshape(-1, outputs.shape[2])
            tgt_flat = tgt[1:].reshape(-1)
            total_loss += criterion(out_flat, tgt_flat).item()
    return total_loss / len(loader)


def run_training(model, train_loader, val_loader, device,
                 num_epochs=15, lr=1e-3, clip=1.0,
                 teacher_forcing_ratio=0.5, save_path="best_t2_model.pt"):
    """
    Returns
    -------
    train_losses, val_losses, total_train_time_seconds
    """
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, verbose=True
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    total_start   = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}  |  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"|  TF ratio={teacher_forcing_ratio:.2f}")
        print(f"{'='*60}")

        t_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                 device, clip, teacher_forcing_ratio)
        v_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"\n  > Train loss: {t_loss:.4f}  |  Val loss: {v_loss:.4f}")
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Best model saved -> {save_path}")

        teacher_forcing_ratio = max(0.0, teacher_forcing_ratio - 0.03)

    total_time = time.time() - total_start
    return train_losses, val_losses, total_time
