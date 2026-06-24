"""
Classifier training utilities.

BUG-013: Early stopping for the unified 1D-CNN classifier (notebook 13).
BUG-014: FD001 fine-tuning pass to recover single-domain accuracy after
         unified training.

Usage in notebook 13:

    import sys; sys.path.insert(0, "..")
    from src.utils.classifier_training import (
        ClassifierEarlyStopping,
        finetune_fd001,
    )

    # --- training loop -------------------------------------------------------
    stopper = ClassifierEarlyStopping(
        model=model,
        val_loader=val_loader,   # hold-out FD001 windows
        ckpt_path="/content/checkpoints/best_unified.pt",
        patience=5,
        eval_interval=5,
        device=DEVICE,
    )

    for epoch in range(MAX_EPOCHS):
        # ... one epoch of training on the unified loader ...
        val_f1 = compute_val_f1(model, val_loader, DEVICE)
        if stopper.step(epoch, val_f1):
            break

    stopper.restore_best()

    # --- FD001 fine-tuning (BUG-014) -----------------------------------------
    finetune_fd001(
        model=model,
        fd001_loader=fd001_loader,
        device=DEVICE,
        epochs=10,
        lr=1e-4,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_val_f1(
    model: nn.Module,
    loader,
    device,
    average: str = "macro",
) -> float:
    """Evaluate macro-F1 on a DataLoader without gradient tracking."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
    model.train()
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return float(f1_score(y_true, y_pred, average=average))


# ---------------------------------------------------------------------------
# BUG-013: Early stopping for the unified classifier
# ---------------------------------------------------------------------------

class ClassifierEarlyStopping:
    """
    BUG-013: Patience-based early stopping for the unified 1D-CNN classifier.

    Evaluates macro-F1 on a validation loader every `eval_interval` epochs
    and stops training when F1 has not improved for `patience` consecutive
    evaluations.  The best model checkpoint is saved automatically.

    Args:
        model         : the 1D-CNN model being trained.
        val_loader    : DataLoader for validation data (e.g. held-out FD001).
        ckpt_path     : path where the best model state dict is saved.
        patience      : consecutive non-improving evaluations before stopping.
        eval_interval : evaluate every this many epochs.
        device        : torch device.
        min_delta     : minimum F1 improvement to count as improvement.
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader,
        ckpt_path,
        patience: int = 5,
        eval_interval: int = 5,
        device=None,
        min_delta: float = 1e-4,
    ):
        self.model = model
        self.val_loader = val_loader
        self.ckpt_path = Path(ckpt_path)
        self.patience = patience
        self.eval_interval = eval_interval
        self.device = device or torch.device("cpu")
        self.min_delta = min_delta

        self._best_f1 = -1.0
        self._no_improve = 0

    def step(self, epoch: int, val_f1: Optional[float] = None) -> bool:
        """
        Call at the end of each epoch.

        If `val_f1` is already computed (e.g. from the training loop),
        it is used directly; otherwise the validation set is evaluated here.

        Returns True when training should stop.
        """
        if (epoch + 1) % self.eval_interval != 0:
            return False

        if val_f1 is None:
            val_f1 = compute_val_f1(self.model, self.val_loader, self.device)

        print(
            f"  [EarlyStopping] epoch {epoch + 1}: val_macro_f1={val_f1:.4f} "
            f"  best={self._best_f1:.4f}"
        )

        if val_f1 >= self._best_f1 + self.min_delta:
            self._best_f1 = val_f1
            self._no_improve = 0
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_path)
            print(f"    -> New best; checkpoint saved to {self.ckpt_path}")
        else:
            self._no_improve += 1
            print(
                f"    -> No improvement ({self._no_improve}/{self.patience})"
            )
            if self._no_improve >= self.patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                return True

        return False

    def restore_best(self):
        """Load the best checkpoint back into the model."""
        if self.ckpt_path.exists():
            self.model.load_state_dict(
                torch.load(self.ckpt_path, map_location=self.device)
            )
            print(f"Restored best classifier weights from {self.ckpt_path}")
        else:
            print(f"[Warning] No best checkpoint found at {self.ckpt_path}")


# ---------------------------------------------------------------------------
# BUG-014: FD001 fine-tuning pass
# ---------------------------------------------------------------------------

def finetune_fd001(
    model: nn.Module,
    fd001_loader,
    device,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    ckpt_path: Optional[Path] = None,
) -> dict:
    """
    BUG-014: Fine-tune the unified classifier on FD001 data.

    The unified classifier generalises over all four subsets but loses some
    FD001-specific precision.  This function performs a short additional
    training pass on FD001 data with a low learning rate to recover
    single-domain accuracy without forgetting multi-subset knowledge.

    Args:
        model        : the already-trained unified 1D-CNN.
        fd001_loader : DataLoader of FD001 balanced training windows.
        device       : torch device.
        epochs       : number of fine-tuning epochs (keep small: 5-15).
        lr           : fine-tuning learning rate (default 1e-4, much smaller
                       than the original 1e-3 to avoid catastrophic forgetting).
        weight_decay : L2 regularisation.
        ckpt_path    : if provided, save the fine-tuned model here.

    Returns:
        History dict with ``train_loss`` and ``train_acc`` lists.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": []}

    print(
        f"Fine-tuning on FD001 for {epochs} epochs  "
        f"(lr={lr}, BUG-014 fix) ..."
    )

    for epoch in range(epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for x_batch, y_batch in fd001_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(y_batch)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"  FT Epoch {epoch + 1:>3}/{epochs}  "
                f"loss={avg_loss:.4f}  acc={avg_acc:.4f}"
            )

    print("Fine-tuning complete.")

    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Fine-tuned model saved -> {ckpt_path}")

    return history
