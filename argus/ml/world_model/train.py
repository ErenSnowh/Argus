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

from ml.world_model.features import NUM_FEATURES, WORLD_MODEL_FEATURES
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
        Maximum training epochs
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

    Returns
    -------
    dict with training metrics and model info
    """
    _check_torch()

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
    )

    # Train/test split (temporal — no future leakage)
    (X_train, yl_train, yi_train), (X_test, yl_test, yi_test) = \
        train_test_split_temporal(X, y_labels, y_infiltration, test_ratio=0.2)

    print(f"  Train sequences: {X_train.shape[0]}")
    print(f"  Test sequences:  {X_test.shape[0]}")

    # Normalize features (per-feature z-score on training set)
    train_mean = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
    train_std = X_train.reshape(-1, NUM_FEATURES).std(axis=0)
    train_std[train_std < 1e-8] = 1.0  # Avoid division by zero

    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

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
    criterion = WorldModelLoss(alpha=0.3, beta=0.4, gamma=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Model parameters: {model.get_num_params():,}")

    # Training loop
    history = {
        "train_loss": [], "val_loss": [],
        "stage_accuracy": [], "infiltration_auc": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_losses = []
        for X_batch, yl_batch, yi_batch in train_loader:
            X_batch = X_batch.to(dev)
            yl_batch = yl_batch.to(dev)
            yi_batch = yi_batch.to(dev)

            optimizer.zero_grad()
            outputs = model(X_batch)
            losses = criterion(outputs, X_batch, yl_batch, yi_batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(losses["total"].item())

        train_loss = np.mean(epoch_losses)
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

        val_loss = np.mean(val_losses)
        stage_acc = np.mean(
            np.concatenate(all_stage_preds) == np.concatenate(all_stage_true)
        )

        # Simple AUC approximation
        inf_preds_all = np.concatenate(all_inf_preds)
        inf_true_all = np.concatenate(all_inf_true)
        try:
            from sklearn.metrics import roc_auc_score
            inf_auc = roc_auc_score(inf_true_all, inf_preds_all)
        except (ValueError, ImportError):
            inf_auc = 0.5

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["stage_accuracy"].append(float(stage_acc))
        history["infiltration_auc"].append(float(inf_auc))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Stage Acc: {stage_acc:.3f} | "
                f"Inf AUC: {inf_auc:.3f}"
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
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    train_time = time.time() - t0

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
    )
