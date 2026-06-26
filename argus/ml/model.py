"""
model.py
---------
Trains and serves the Random Forest flow classifier that anchors ARGUS's
detection layer -- the same methodology described in the author's standalone
IDS project (feature engineering -> RF classifier -> k-fold CV -> hyperparameter
tuning), reapplied here as the sensor that feeds the agent swarm.
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
        class_probs = {str(c): float(p) for c, p in zip(self.clf.classes_, proba)}
        confidence = float(max(proba))
        return {
            "predicted_label": str(pred),
            "confidence": confidence,
            "class_probabilities": class_probs,
        }


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}, indent=2))
