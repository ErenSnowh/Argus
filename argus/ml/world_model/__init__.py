"""
world_model — Temporal Transformer World Model for Network Attack Forecasting.

This subpackage implements the SIH26153 "World Model" architecture that learns
network state-transition dynamics P(S_t+1 | S_t) and performs K-step forward
simulation to forecast infiltration probability and MITRE ATT&CK stage
progression.

Modules:
    features        — Extended flow+packet feature schema & extraction
    dataset_loader  — CIC-IDS-2018/CTU-13 loaders + temporal sequence builder
    model           — Temporal Transformer world model (PyTorch)
    train           — Training script with early stopping
    predictor       — K-step forward simulation + attention-based explainability
    benchmark       — World Model vs Logistic Regression baseline comparison
"""
