"""
model.py
---------
Trains and serves the Random Forest flow classifier that anchors ARGUS's
detection layer -- the same methodology described in the author's standalone
IDS project (feature engineering -> RF classifier -> k-fold CV -> hyperparameter
tuning), reapplied here as the sensor that feeds the agent swarm.

Includes SHAP-based explainability so SOC analysts can understand *why*
the model flagged a flow — critical for trust in ML-driven detection
and for CERT-In incident reports that require detailed justification.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from ml.flow_features import FEATURE_COLUMNS, generate_dataset

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "rf_flow_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Typical "normal" ranges for human-readable SHAP explanations.
# These are approximate medians from BENIGN traffic in the training data.
_NORMAL_RANGES: dict[str, tuple[float, str]] = {
    "flow_duration_ms": (800, "ms"),
    "total_fwd_packets": (12, "pkts"),
    "total_bwd_packets": (11, "pkts"),
    "total_fwd_bytes": (1400, "B"),
    "total_bwd_bytes": (1300, "B"),
    "fwd_packet_len_mean": (450, "B"),
    "fwd_packet_len_std": (80, "B"),
    "bwd_packet_len_mean": (420, "B"),
    "bwd_packet_len_std": (75, "B"),
    "flow_bytes_per_sec": (3500, "B/s"),
    "flow_packets_per_sec": (15, "pkt/s"),
    "flow_iat_mean": (60, "ms"),
    "flow_iat_std": (20, "ms"),
    "fwd_iat_mean": (55, "ms"),
    "bwd_iat_mean": (58, "ms"),
    "syn_flag_count": (1, ""),
    "ack_flag_count": (10, ""),
    "rst_flag_count": (0, ""),
    "psh_flag_count": (3, ""),
    "fin_flag_count": (1, ""),
    "unique_dst_ports_per_src": (2, "ports"),
    "packets_per_flow": (23, "pkts"),
    "avg_packet_size": (430, "B"),
    "down_up_ratio": (0.95, ""),
}


def train(n_per_class: int = 1500, tune: bool = True, seed: int = 42) -> dict:
    df = generate_dataset(n_per_class=n_per_class, seed=seed)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 16],
            "min_samples_leaf": [1, 2],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = GridSearchCV(
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        clf = search.best_estimator_
        best_params = search.best_params_
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        best_params = {}

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_).tolist()

    importances = sorted(
        zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda t: -t[1]
    )

    metrics = {
        "best_params": best_params,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": {"labels": list(clf.classes_), "matrix": cm},
        "top_features": [{"feature": f, "importance": float(i)} for f, i in importances[:10]],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


class FlowClassifier:
    """Thin inference wrapper used by the MCP server's `classify_flow` tool."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python scripts/train_model.py` first."
            )
        self.clf: RandomForestClassifier = joblib.load(model_path)

    def predict(self, features: dict) -> dict:
        row = pd.DataFrame([{col: features.get(col, 0.0) for col in FEATURE_COLUMNS}])
        pred = self.clf.predict(row)[0]
        proba = self.clf.predict_proba(row)[0]
        confidence = float(np.max(proba))
        class_probs = {str(c): float(p) for c, p in zip(self.clf.classes_, proba)}
        return {
            "predicted_label": str(pred),
            "confidence": confidence,
            "class_probabilities": class_probs,
        }

    def predict_batch(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fast vectorized batch inference. Returns (predictions, probabilities)."""
        clean_df = df[FEATURE_COLUMNS].fillna(0)
        preds = self.clf.predict(clean_df)
        probs = self.clf.predict_proba(clean_df)
        return preds, probs

    def predict_with_explanation(self, features: dict, top_n: int = 5) -> dict:
        """Predict with SHAP-based feature attribution.

        Uses TreeSHAP (exact, fast SHAP for tree ensembles) to explain
        *why* the model predicted a particular label. Returns the standard
        prediction plus:
          - top_features: list of {feature, value, shap_value, direction, explanation}
          - explanation_text: human-readable natural-language explanation

        This is critical for SOC analyst trust — they need to understand
        the ML's reasoning before acting on an alert.
        """
        base = self.predict(features)
        row = pd.DataFrame([{col: features.get(col, 0.0) for col in FEATURE_COLUMNS}])

        # Use TreeSHAP for fast, exact explanations
        try:
            import shap
            explainer = shap.TreeExplainer(self.clf)
            shap_values = explainer.shap_values(row)
        except ImportError:
            # Graceful fallback: use feature importances if shap isn't installed
            return self._fallback_explanation(base, features, top_n)

        # shap_values shape: (n_classes, n_samples, n_features) for RF
        predicted_label = base["predicted_label"]
        class_idx = list(self.clf.classes_).index(predicted_label)

        # Get SHAP values for the predicted class
        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]  # (n_features,) for this sample
        else:
            sv = shap_values[0, :, class_idx] if shap_values.ndim == 3 else shap_values[0]

        # Pair features with their SHAP values and sort by absolute impact
        feature_impacts = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = float(row[col].iloc[0])
            shap_val = float(sv[i])
            direction = "↑ increases" if shap_val > 0 else "↓ decreases"
            normal, unit = _NORMAL_RANGES.get(col, (0, ""))

            # Build per-feature explanation
            if normal > 0 and val > 0:
                ratio = val / normal
                if ratio > 2.0:
                    explanation = f"{col}={val:.1f}{unit} is {ratio:.0f}x above normal ({normal}{unit})"
                elif ratio < 0.5:
                    explanation = f"{col}={val:.1f}{unit} is {1/ratio:.0f}x below normal ({normal}{unit})"
                else:
                    explanation = f"{col}={val:.1f}{unit} (near normal range)"
            else:
                explanation = f"{col}={val:.1f}"

            feature_impacts.append({
                "feature": col,
                "value": round(val, 2),
                "shap_value": round(shap_val, 4),
                "abs_shap": abs(shap_val),
                "direction": direction,
                "explanation": explanation,
            })

        # Sort by absolute SHAP value (most impactful first)
        feature_impacts.sort(key=lambda x: x["abs_shap"], reverse=True)
        top = feature_impacts[:top_n]

        # Build natural-language explanation
        explanation_text = self._build_explanation_text(
            predicted_label, base["confidence"], top
        )

        # Remove abs_shap from output (internal sorting key)
        for item in top:
            del item["abs_shap"]

        base["top_features"] = top
        base["explanation_text"] = explanation_text
        return base

    def _build_explanation_text(
        self, label: str, confidence: float, top_features: list[dict]
    ) -> str:
        """Generate a natural-language explanation for the detection."""
        if label == "BENIGN":
            return (
                f"Classified as BENIGN with {confidence:.0%} confidence. "
                "Flow characteristics are within normal parameters."
            )

        parts = [
            f"Flagged as **{label}** with {confidence:.0%} confidence. "
            "Key factors driving this detection:"
        ]
        for i, feat in enumerate(top_features[:3], 1):
            parts.append(f"  {i}. {feat['explanation']} ({feat['direction']} {label} likelihood)")

        return "\n".join(parts)

    def _fallback_explanation(self, base: dict, features: dict, top_n: int) -> dict:
        """Fallback when SHAP is not installed — use feature importances instead."""
        importances = sorted(
            zip(FEATURE_COLUMNS, self.clf.feature_importances_),
            key=lambda t: -t[1],
        )
        top = []
        for col, imp in importances[:top_n]:
            val = features.get(col, 0.0)
            normal, unit = _NORMAL_RANGES.get(col, (0, ""))
            if normal > 0 and val > 0:
                ratio = val / normal
                if ratio > 2.0:
                    explanation = f"{col}={val:.1f}{unit} is {ratio:.0f}x above normal"
                elif ratio < 0.5:
                    explanation = f"{col}={val:.1f}{unit} is {1/ratio:.0f}x below normal"
                else:
                    explanation = f"{col}={val:.1f}{unit} (near normal)"
            else:
                explanation = f"{col}={val:.1f}"
            top.append({
                "feature": col,
                "value": round(val, 2),
                "importance": round(float(imp), 4),
                "explanation": explanation,
            })

        explanation_text = self._build_explanation_text(
            base["predicted_label"], base["confidence"],
            [{"explanation": f["explanation"], "direction": "contributes to"} for f in top]
        )
        base["top_features"] = top
        base["explanation_text"] = explanation_text
        base["shap_available"] = False
        return base


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}, indent=2))
