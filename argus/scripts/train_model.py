#!/usr/bin/env python3
"""Entry point: python scripts/train_model.py [--fast] [--n-per-class N]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.model import train  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Skip grid-search hyperparameter tuning")
    parser.add_argument("--n-per-class", type=int, default=1500)
    args = parser.parse_args()

    print("Training ARGUS Random Forest flow classifier on synthetic CICIDS2017-style data...")
    metrics = train(n_per_class=args.n_per_class, tune=not args.fast)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nTop features:")
    for f in metrics["top_features"][:8]:
        print(f"  {f['feature']:28s} {f['importance']:.4f}")
    print(f"\nFull metrics written to ml/artifacts/metrics.json")
    print(json.dumps(metrics["per_class"], indent=2))
