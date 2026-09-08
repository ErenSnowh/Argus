"""
train.py
---------
Training script for the World Model Transformer.

Trains on either synthetic temporal data or real CIC-IDS-2018/CTU-13 datasets,
saves model weights and training metrics, and supports early stopping.

Usage:
    python -m ml.world_model.train                      # synthetic data
    python -m ml.world_model.train --dataset cicids2018 --path /path/to/csvs
    python -m ml.world_model.train --epochs 100 --lr 0.001
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ml.world_model.features import (
    ATTACK_LABELS,
    ATTACK_STAGE_NAMES,
    NUM_FEATURES,
    WORLD_MODEL_FEATURES,
)
from ml.world_model.model import (
    WORLD_MODEL_METRICS_PATH,
    WORLD_MODEL_PATH,
    WorldModelLoss,
    WorldModelTransformer,
    _check_torch,
)
from ml.world_model.dataset_loader import load_dataset, train_test_split_temporal

# PyTorch import (guarded)
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    pass


def _compute_grad_norm(module: torch.nn.Module) -> float:
    """Compute L2 norm of gradients across all parameters of a module."""
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_world_model(
    data_source: str = "synthetic",
    data_path: str | None = None,
    n_sequences: int = 500,
    seq_len: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 8,
    device: str = "auto",
    seed: int = 42,
    validate_supervision: bool = False,
    max_rows: int | None = None,
) -> dict:
    """Train the World Model Transformer.

    Parameters
    ----------
    data_source : str
        'synthetic' (default), 'cicids2018', or 'ctu13'
    data_path : str or None
        Path to real dataset (required for non-synthetic)
    n_sequences : int
        Number of training sequences (synthetic only)
    seq_len : int
        Time steps per sequence
    epochs : int
        Maximum training epochs (overridden to 5 if validate_supervision=True)
    batch_size : int
        Batch size
    lr : float
        Learning rate
    patience : int
        Early stopping patience (epochs without improvement)
    device : str
        'auto', 'cpu', or 'cuda'
    seed : int
        Random seed
    validate_supervision : bool
        If True, runs a 5-epoch sanity check to verify all prediction heads learn.
    max_rows : int or None
        Maximum rows to load for real datasets

    Returns
    -------
    dict with training metrics and model info
    """
    _check_torch()

    if validate_supervision:
        epochs = 5
        print("\n[Mode] Running Multi-Task Supervision Verification (5 epochs)...")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"Training World Model on {device}...")
    print(f"  Data source: {data_source}")
    print(f"  Sequences: {n_sequences}, Seq length: {seq_len}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Load data
    t0 = time.time()
    X, y_labels, y_infiltration = load_dataset(
        source=data_source,
        path=data_path,
        n_sequences=n_sequences,
        seq_len=seq_len,
        seed=seed,
        max_rows=max_rows,
    )

    # Train/test split (temporal — no future leakage)
    (X_train, yl_train, yi_train), (X_test, yl_test, yi_test) = \
        train_test_split_temporal(X, y_labels, y_infiltration, test_ratio=0.2)

    print(f"  Train sequences: {X_train.shape[0]}")
    print(f"  Test sequences:  {X_test.shape[0]}")

    # Label distribution verification (P0-B)
    unique_classes, counts = np.unique(yl_train, return_counts=True)
    class_dist = {
        ATTACK_STAGE_NAMES.get(int(c), str(c)): int(cnt)
        for c, cnt in zip(unique_classes, counts)
    }
    benign_pct = float(np.sum(yl_train == 0) / yl_train.size) * 100.0
    print(f"  Label distribution in training: {class_dist}")
    print(f"  Benign ratio: {benign_pct:.1f}%")
    if benign_pct > 95.0:
        print("  [!] Warning: Extreme class imbalance detected (>95% BENIGN).")

    # Normalize features (per-feature z-score on training set with outlier clipping)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=0.0)

    train_mean = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
    train_std = X_train.reshape(-1, NUM_FEATURES).std(axis=0)
    train_std[train_std < 1e-8] = 1.0  # Avoid division by zero

    X_train_norm = np.clip((X_train - train_mean) / train_std, -10.0, 10.0)
    X_test_norm = np.clip((X_test - train_mean) / train_std, -10.0, 10.0)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.LongTensor(yl_train),
        torch.FloatTensor(yi_train),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.LongTensor(yl_test),
        torch.FloatTensor(yi_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = WorldModelTransformer(n_features=NUM_FEATURES).to(dev)
    criterion = WorldModelLoss(alpha=0.25, beta=0.30, gamma=0.25, delta=0.20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Model parameters: {model.get_num_params():,}")

    # Training loop history
    history = {
        "train_loss": [], "val_loss": [],
        "state_loss": [], "stage_loss": [], "infiltration_loss": [], "horizon_loss": [],
        "stage_accuracy": [], "infiltration_auc": [],
        "grad_norms": {"state": [], "stage": [], "infiltration": []},
    }
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_losses = []
        epoch_state_losses = []
        epoch_stage_losses = []
        epoch_inf_losses = []
        epoch_horizon_losses = []

        for X_batch, yl_batch, yi_batch in train_loader:
            X_batch = X_batch.to(dev)
            yl_batch = yl_batch.to(dev)
            yi_batch = yi_batch.to(dev)

            optimizer.zero_grad()
            outputs = model(X_batch)
            losses = criterion(outputs, X_batch, yl_batch, yi_batch)
            losses["total"].backward()

            # Record gradient norms for each head to verify active learning (P0-B)
            gn_state = _compute_grad_norm(model.state_head)
            gn_stage = _compute_grad_norm(model.stage_head)
            gn_inf = _compute_grad_norm(model.infiltration_head)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(losses["total"].item())
            epoch_state_losses.append(losses["state_loss"].item())
            epoch_stage_losses.append(losses["stage_loss"].item())
            epoch_inf_losses.append(losses["infiltration_loss"].item())
            epoch_horizon_losses.append(losses["horizon_loss"].item())

        train_loss = float(np.mean(epoch_losses))
        avg_state_loss = float(np.mean(epoch_state_losses))
        avg_stage_loss = float(np.mean(epoch_stage_losses))
        avg_inf_loss = float(np.mean(epoch_inf_losses))
        avg_horizon_loss = float(np.mean(epoch_horizon_losses))

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_losses = []
        all_stage_preds = []
        all_stage_true = []
        all_inf_preds = []
        all_inf_true = []

        with torch.no_grad():
            for X_batch, yl_batch, yi_batch in test_loader:
                X_batch = X_batch.to(dev)
                yl_batch = yl_batch.to(dev)
                yi_batch = yi_batch.to(dev)

                outputs = model(X_batch)
                losses = criterion(outputs, X_batch, yl_batch, yi_batch)
                val_losses.append(losses["total"].item())

                # Stage accuracy
                stage_preds = outputs["stage_logits"].argmax(dim=-1).cpu().numpy()
                all_stage_preds.append(stage_preds.flatten())
                all_stage_true.append(yl_batch.cpu().numpy().flatten())

                # Infiltration predictions
                inf_preds = torch.sigmoid(outputs["infiltration_logits"]).squeeze(-1).cpu().numpy()
                all_inf_preds.append(inf_preds.flatten())
                all_inf_true.append(yi_batch.cpu().numpy().flatten())

        val_loss = float(np.mean(val_losses))
        stage_acc = float(np.mean(
            np.concatenate(all_stage_preds) == np.concatenate(all_stage_true)
        ))

        # AUC approximation
        inf_preds_all = np.concatenate(all_inf_preds)
        inf_true_all = np.concatenate(all_inf_true)
        try:
            from sklearn.metrics import roc_auc_score
            inf_auc = float(roc_auc_score(inf_true_all, inf_preds_all))
        except (ValueError, ImportError):
            inf_auc = 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["state_loss"].append(avg_state_loss)
        history["stage_loss"].append(avg_stage_loss)
        history["infiltration_loss"].append(avg_inf_loss)
        history["horizon_loss"].append(avg_horizon_loss)
        history["stage_accuracy"].append(stage_acc)
        history["infiltration_auc"].append(inf_auc)
        history["grad_norms"]["state"].append(float(gn_state))
        history["grad_norms"]["stage"].append(float(gn_stage))
        history["grad_norms"]["infiltration"].append(float(gn_inf))

        if (epoch + 1) % 5 == 0 or epoch == 0 or validate_supervision:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"State: {avg_state_loss:.4f} | "
                f"Stage: {avg_stage_loss:.4f} (Acc: {stage_acc:.3f}) | "
                f"Inf: {avg_inf_loss:.4f} (AUC: {inf_auc:.3f}) | "
                f"Horizon: {avg_horizon_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            MODEL_DIR = WORLD_MODEL_PATH.parent
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_features": NUM_FEATURES,
                "train_mean": train_mean.tolist(),
                "train_std": train_std.tolist(),
                "feature_names": WORLD_MODEL_FEATURES,
            }, str(WORLD_MODEL_PATH))
        else:
            patience_counter += 1
            if patience_counter >= patience and not validate_supervision:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    train_time = time.time() - t0

    # Multi-task supervision validation checks (P0-B)
    supervision_passed = True
    validation_reasons = []
    if len(history["state_loss"]) >= 2:
        state_drop = history["state_loss"][0] - history["state_loss"][-1]
        stage_drop = history["stage_loss"][0] - history["stage_loss"][-1]
        inf_drop = history["infiltration_loss"][0] - history["infiltration_loss"][-1]
        all_grads_active = (
            history["grad_norms"]["state"][-1] > 0 and
            history["grad_norms"]["stage"][-1] > 0 and
            history["grad_norms"]["infiltration"][-1] > 0
        )

        if not all_grads_active:
            supervision_passed = False
            validation_reasons.append("Dead head detected (gradient norm = 0)")
        if state_drop < -0.05:
            supervision_passed = False
            validation_reasons.append("State head loss diverged")
        if stage_drop < -0.1:
            supervision_passed = False
            validation_reasons.append("Stage head loss diverged")

    if validate_supervision:
        if supervision_passed:
            print("\n  [✓] Multi-Task Supervision Validation PASSED:")
            print(f"      - All 3 prediction heads active with non-zero gradients")
            print(f"      - State head loss: {history['state_loss'][0]:.4f} → {history['state_loss'][-1]:.4f}")
            print(f"      - Stage head loss: {history['stage_loss'][0]:.4f} → {history['stage_loss'][-1]:.4f}")
            print(f"      - Infiltration loss: {history['infiltration_loss'][0]:.4f} → {history['infiltration_loss'][-1]:.4f}")
        else:
            print(f"\n  [✗] Multi-Task Supervision Validation WARNING: {validation_reasons}")

    # Final metrics
    metrics = {
        "data_source": data_source,
        "n_train_sequences": int(X_train.shape[0]),
        "n_test_sequences": int(X_test.shape[0]),
        "seq_len": seq_len,
        "n_features": NUM_FEATURES,
        "model_params": model.get_num_params(),
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": float(best_val_loss),
        "final_stage_accuracy": history["stage_accuracy"][-1],
        "final_infiltration_auc": history["infiltration_auc"][-1],
        "train_time_sec": round(train_time, 1),
        "device": device,
        "supervision_validated": supervision_passed,
        "validation_reasons": validation_reasons,
        "history": history,
    }

    # Save metrics
    WORLD_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORLD_MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Model saved to: {WORLD_MODEL_PATH}")
    print(f"  Metrics saved to: {WORLD_MODEL_METRICS_PATH}")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Stage accuracy: {metrics['final_stage_accuracy']:.3f}")
    print(f"  Infiltration AUC: {metrics['final_infiltration_auc']:.3f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the ARGUS World Model")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "cicids2018", "ctu13"])
    parser.add_argument("--path", default=None, help="Path to dataset directory/file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequences", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to load from real dataset")
    parser.add_argument("--validate-supervision", action="store_true", help="Run 5-epoch multi-task supervision sanity check")
    args = parser.parse_args()

    train_world_model(
        data_source=args.dataset,
        data_path=args.path,
        n_sequences=args.sequences,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        validate_supervision=args.validate_supervision,
        max_rows=args.max_rows,
    )
