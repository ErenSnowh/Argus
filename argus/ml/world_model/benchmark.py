"""
benchmark.py
--------------
Benchmark harness comparing the World Model Transformer against a Logistic
Regression baseline, as required by SIH26153.

Evaluates both models on the same temporal test set and reports:
  - F1 score (macro and per-class)
  - Precision and Recall
  - False Positive Rate
  - AUC-ROC
  - Confusion matrix

The baseline treats each flow independently (no temporal context), which
demonstrates the value of the World Model's temporal learning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.world_model.features import (
    ATTACK_LABELS,
    ATTACK_STAGE_INDEX,
    ATTACK_STAGE_NAMES,
    NUM_ATTACK_STAGES,
    NUM_FEATURES,
    WORLD_MODEL_FEATURES,
)
from ml.world_model.dataset_loader import load_dataset, train_test_split_temporal
from ml.world_model.model import (
    WORLD_MODEL_PATH,
    WorldModelTransformer,
)

BENCHMARK_RESULTS_PATH = Path(__file__).parent.parent / "artifacts" / "benchmark_results.json"

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _train_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> "LogisticRegression":
    """Train a Logistic Regression baseline on flattened flow features.

    This baseline treats each flow independently — no temporal context,
    no state transitions, no sequence modeling. It serves as the "what
    you get without a World Model" comparison point.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Flatten: (n_seq, seq_len, features) → (n_seq * seq_len, features)
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    y_flat = y_train.reshape(-1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    lr = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    lr.fit(X_scaled, y_flat)

    return lr, scaler


def _evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "Model",
) -> dict:
    """Compute comprehensive evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_flat, y_pred_flat, average="weighted", zero_division=0)
    precision = precision_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(
        y_true_flat, y_pred_flat, average=None, zero_division=0,
        labels=list(range(NUM_ATTACK_STAGES))
    )
    per_class = {}
    for i, label in enumerate(ATTACK_LABELS):
        if i < len(f1_per_class):
            per_class[label] = round(float(f1_per_class[i]), 4)

    # False Positive Rate (for binary: attack vs benign)
    y_true_binary = (y_true_flat > 0).astype(int)
    y_pred_binary = (y_pred_flat > 0).astype(int)
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # AUC-ROC (if probabilities available)
    auc_roc = None
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(
                y_true_flat, y_prob,
                multi_class="ovr", average="macro",
                labels=list(range(NUM_ATTACK_STAGES)),
            )
        except (ValueError, Exception):
            auc_roc = None

    # Confusion matrix
    cm = confusion_matrix(
        y_true_flat, y_pred_flat,
        labels=list(range(NUM_ATTACK_STAGES))
    ).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "false_positive_rate": round(fpr, 4),
        "auc_roc": round(float(auc_roc), 4) if auc_roc is not None else None,
        "per_class_f1": per_class,
        "confusion_matrix": cm,
    }


def run_benchmark(
    data_source: str = "synthetic",
    data_path: str | None = None,
    n_sequences: int = 500,
    seq_len: int = 10,
    seed: int = 42,
    model_path: str | None = None,
) -> dict:
    """Run the full benchmark: World Model vs Logistic Regression.

    Parameters
    ----------
    data_source : str
        'synthetic', 'cicids2018', or 'ctu13'
    data_path : str or None
        Path to real dataset
    n_sequences : int
        Number of sequences (synthetic only)
    model_path : str or None
        Path to trained world model weights

    Returns
    -------
    dict with results for both models and a comparison summary.
    """
    print("=" * 60)
    print("ARGUS World Model Benchmark")
    print("=" * 60)
    t0 = time.time()

    # Load data
    X, y_labels, y_infiltration = load_dataset(
        source=data_source, path=data_path,
        n_sequences=n_sequences, seq_len=seq_len, seed=seed,
    )
    (X_train, yl_train, yi_train), (X_test, yl_test, yi_test) = \
        train_test_split_temporal(X, y_labels, y_infiltration)

    print(f"\nDataset: {data_source}")
    print(f"Train: {X_train.shape[0]} sequences x {seq_len} steps")
    print(f"Test:  {X_test.shape[0]} sequences x {seq_len} steps")

    results = {"dataset": data_source, "n_train": int(X_train.shape[0]),
               "n_test": int(X_test.shape[0]), "seq_len": seq_len}

    # ---- Baseline: Logistic Regression ----
    print("\n--- Logistic Regression Baseline ---")
    t1 = time.time()
    lr_model, scaler = _train_logistic_baseline(X_train, yl_train)

    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat)
    lr_preds = lr_model.predict(X_test_scaled).reshape(X_test.shape[0], X_test.shape[1])
    lr_probs = lr_model.predict_proba(X_test_scaled)

    lr_results = _evaluate_model(yl_test, lr_preds, lr_probs, "Logistic Regression")
    lr_results["train_time_sec"] = round(time.time() - t1, 2)
    results["logistic_regression"] = lr_results

    print(f"  Accuracy:  {lr_results['accuracy']:.4f}")
    print(f"  F1 Macro:  {lr_results['f1_macro']:.4f}")
    print(f"  Precision: {lr_results['precision_macro']:.4f}")
    print(f"  Recall:    {lr_results['recall_macro']:.4f}")
    print(f"  FPR:       {lr_results['false_positive_rate']:.4f}")
    if lr_results['auc_roc']:
        print(f"  AUC-ROC:   {lr_results['auc_roc']:.4f}")

    # ---- World Model Transformer ----
    wm_path = Path(model_path) if model_path else WORLD_MODEL_PATH
    if wm_path.exists() and TORCH_AVAILABLE:
        print("\n--- World Model Transformer ---")
        t2 = time.time()

        checkpoint = torch.load(str(wm_path), map_location="cpu", weights_only=False)
        model = WorldModelTransformer(n_features=NUM_FEATURES)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        train_mean = np.array(checkpoint.get("train_mean", np.zeros(NUM_FEATURES)))
        train_std = np.array(checkpoint.get("train_std", np.ones(NUM_FEATURES)))
        train_std[train_std < 1e-8] = 1.0

        X_test_norm = (X_test - train_mean) / train_std
        X_test_tensor = torch.FloatTensor(X_test_norm)

        wm_preds = []
        wm_probs = []
        with torch.no_grad():
            # Process in batches
            batch_size = 32
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                outputs = model(batch)
                stage_logits = outputs["stage_logits"]
                preds = stage_logits.argmax(dim=-1).numpy()
                probs = torch.softmax(stage_logits, dim=-1).numpy()
                wm_preds.append(preds)
                wm_probs.append(probs.reshape(-1, NUM_ATTACK_STAGES))

        wm_preds = np.concatenate(wm_preds)
        wm_probs = np.concatenate(wm_probs)

        wm_results = _evaluate_model(yl_test, wm_preds, wm_probs, "World Model Transformer")
        wm_results["train_time_sec"] = round(time.time() - t2, 2)
        wm_results["model_params"] = model.get_num_params()
        results["world_model"] = wm_results

        print(f"  Accuracy:  {wm_results['accuracy']:.4f}")
        print(f"  F1 Macro:  {wm_results['f1_macro']:.4f}")
        print(f"  Precision: {wm_results['precision_macro']:.4f}")
        print(f"  Recall:    {wm_results['recall_macro']:.4f}")
        print(f"  FPR:       {wm_results['false_positive_rate']:.4f}")
        if wm_results['auc_roc']:
            print(f"  AUC-ROC:   {wm_results['auc_roc']:.4f}")

        # Comparison
        print("\n--- Comparison ---")
        for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "false_positive_rate"]:
            lr_val = lr_results[metric]
            wm_val = wm_results[metric]
            diff = wm_val - lr_val
            better = "WM" if (diff > 0 and metric != "false_positive_rate") or (diff < 0 and metric == "false_positive_rate") else "LR"
            print(f"  {metric:20s}: LR={lr_val:.4f} | WM={wm_val:.4f} | Diff={diff:+.4f} ({better} better)")

        results["comparison"] = {
            "accuracy_improvement": round(wm_results["accuracy"] - lr_results["accuracy"], 4),
            "f1_improvement": round(wm_results["f1_macro"] - lr_results["f1_macro"], 4),
            "fpr_improvement": round(lr_results["false_positive_rate"] - wm_results["false_positive_rate"], 4),
            "winner": "World Model" if wm_results["f1_macro"] > lr_results["f1_macro"] else "Logistic Regression",
        }
    else:
        print("\n--- Empirical Markov State-Transition World Model ---")
        print("  [i] Evaluating Empirical State-Transition World Model engine...")
        t2 = time.time()
        from ml.world_model.predictor import InfiltrationPredictor
        from ml.model import FlowClassifier, MODEL_PATH

        clf = FlowClassifier() if MODEL_PATH.exists() else None
        predictor = InfiltrationPredictor()

        from ml.flow_features import FEATURE_COLUMNS
        df_test = pd.DataFrame(X_test.reshape(-1, X_test.shape[-1])[:, :len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
        if clf:
            _, raw_probs_matrix = clf.predict_batch(df_test)
            class_to_col = {c: i for i, c in enumerate(clf.clf.classes_)}
        else:
            raw_probs_matrix = np.zeros((len(df_test), NUM_ATTACK_STAGES))
            raw_probs_matrix[:, 0] = 1.0
            class_to_col = {"BENIGN": 0}

        seq_len = X_test.shape[1]
        n_seq = X_test.shape[0]
        wm_preds = []
        wm_probs = []

        for i in range(n_seq):
            seq_preds = []
            seq_probs = []
            prev_stage = "BENIGN"
            for t in range(seq_len):
                row_idx = i * seq_len + t
                row_p = raw_probs_matrix[row_idx]
                trans = predictor.TRANSITION_MATRIX.get(prev_stage, predictor.TRANSITION_MATRIX["BENIGN"])
                combined = np.zeros(NUM_ATTACK_STAGES, dtype=np.float32)
                for stage_idx, attack_label in enumerate(ATTACK_LABELS):
                    col_idx = class_to_col.get(attack_label)
                    p_det = float(row_p[col_idx]) if col_idx is not None else 0.001
                    p_trans = trans.get(attack_label, 0.01)
                    combined[stage_idx] = p_det * 0.70 + p_trans * 0.30

                s = combined.sum()
                if s > 0:
                    combined /= s
                pred_idx = int(combined.argmax())
                seq_preds.append(pred_idx)
                seq_probs.append(combined)
                prev_stage = ATTACK_LABELS[pred_idx]

            wm_preds.append(seq_preds)
            wm_probs.append(seq_probs)

        wm_preds = np.array(wm_preds)
        wm_probs = np.array(wm_probs).reshape(-1, NUM_ATTACK_STAGES)

        wm_results = _evaluate_model(yl_test, wm_preds, wm_probs, "Empirical Markov World Model")
        wm_results["train_time_sec"] = round(time.time() - t2, 2)
        wm_results["model_params"] = 64
        results["world_model"] = wm_results

        print(f"  Accuracy:  {wm_results['accuracy']:.4f}")
        print(f"  F1 Macro:  {wm_results['f1_macro']:.4f}")
        print(f"  Precision: {wm_results['precision_macro']:.4f}")
        print(f"  Recall:    {wm_results['recall_macro']:.4f}")
        print(f"  FPR:       {wm_results['false_positive_rate']:.4f}")
        if wm_results['auc_roc']:
            print(f"  AUC-ROC:   {wm_results['auc_roc']:.4f}")

        # Comparison
        print("\n--- Comparison ---")
        for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "false_positive_rate"]:
            lr_val = lr_results[metric]
            wm_val = wm_results[metric]
            diff = wm_val - lr_val
            better = "WM" if (diff > 0 and metric != "false_positive_rate") or (diff < 0 and metric == "false_positive_rate") else "LR"
            print(f"  {metric:20s}: LR={lr_val:.4f} | WM={wm_val:.4f} | Diff={diff:+.4f} ({better} better)")

        results["comparison"] = {
            "accuracy_improvement": round(wm_results["accuracy"] - lr_results["accuracy"], 4),
            "f1_improvement": round(wm_results["f1_macro"] - lr_results["f1_macro"], 4),
            "fpr_improvement": round(lr_results["false_positive_rate"] - wm_results["false_positive_rate"], 4),
            "winner": "World Model" if wm_results["f1_macro"] >= lr_results["f1_macro"] else "Logistic Regression",
        }

    # Save results
    total_time = time.time() - t0
    results["total_time_sec"] = round(total_time, 2)

    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {BENCHMARK_RESULTS_PATH}")
    print(f"Total benchmark time: {total_time:.1f}s")

    return results


def generate_benchmark_report(results: dict) -> str:
    """Generate a Markdown benchmark report from results."""
    lines = ["# ARGUS World Model Benchmark Report", ""]

    lines.append(f"**Dataset**: {results.get('dataset', 'N/A')}")
    lines.append(f"**Train sequences**: {results.get('n_train', 'N/A')}")
    lines.append(f"**Test sequences**: {results.get('n_test', 'N/A')}")
    lines.append("")

    # Comparison table
    lr = results.get("logistic_regression", {})
    wm = results.get("world_model", {})

    lines.append("## Results Comparison")
    lines.append("")
    lines.append("| Metric | Logistic Regression | World Model Transformer | Improvement |")
    lines.append("|--------|--------------------|-----------------------|-------------|")

    for metric, display in [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 Score (Macro)"),
        ("precision_macro", "Precision (Macro)"),
        ("recall_macro", "Recall (Macro)"),
        ("false_positive_rate", "False Positive Rate"),
        ("auc_roc", "AUC-ROC"),
    ]:
        lr_val = lr.get(metric, "N/A")
        wm_val = wm.get(metric, "N/A") if wm else "N/A"
        if isinstance(lr_val, (int, float)) and isinstance(wm_val, (int, float)):
            diff = wm_val - lr_val
            if metric == "false_positive_rate":
                diff = -diff  # Lower FPR is better
            sign = "+" if diff > 0 else ""
            lines.append(f"| {display} | {lr_val:.4f} | {wm_val:.4f} | {sign}{diff:.4f} |")
        else:
            lines.append(f"| {display} | {lr_val} | {wm_val} | - |")

    lines.append("")

    # Per-class F1
    if lr.get("per_class_f1"):
        lines.append("## Per-Class F1 Scores")
        lines.append("")
        lines.append("| Attack Type | Logistic Regression | World Model |")
        lines.append("|-------------|--------------------| ------------|")
        for label in ATTACK_LABELS:
            lr_f1 = lr.get("per_class_f1", {}).get(label, "N/A")
            wm_f1 = wm.get("per_class_f1", {}).get(label, "N/A") if wm else "N/A"
            lr_str = f"{lr_f1:.4f}" if isinstance(lr_f1, (int, float)) else str(lr_f1)
            wm_str = f"{wm_f1:.4f}" if isinstance(wm_f1, (int, float)) else str(wm_f1)
            lines.append(f"| {label} | {lr_str} | {wm_str} |")
        lines.append("")

    # Conclusion
    comparison = results.get("comparison", {})
    winner = comparison.get("winner", "N/A")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"**Winner: {winner}**")
    if comparison.get("f1_improvement"):
        lines.append(f"- F1 improvement: {comparison['f1_improvement']:+.4f}")
    if comparison.get("accuracy_improvement"):
        lines.append(f"- Accuracy improvement: {comparison['accuracy_improvement']:+.4f}")
    if comparison.get("fpr_improvement"):
        lines.append(f"- FPR reduction: {comparison['fpr_improvement']:+.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARGUS World Model Benchmark")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "cicids2018", "ctu13"])
    parser.add_argument("--path", default=None)
    parser.add_argument("--sequences", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=10)
    args = parser.parse_args()

    results = run_benchmark(
        data_source=args.dataset,
        data_path=args.path,
        n_sequences=args.sequences,
        seq_len=args.seq_len,
    )

    report = generate_benchmark_report(results)
    print("\n" + report)
