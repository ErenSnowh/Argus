"""
model.py
---------
Temporal Transformer World Model for network attack forecasting.

Learns the state-transition dynamics P(S_t+1 | S_t) from network traffic
telemetry. Instead of classifying individual flows in isolation (like the
RF classifier), this model learns HOW network states evolve over time and
predicts FUTURE states via K-step forward simulation.

Architecture:
    Input(t) → [StateEncoder] → [PositionalEncoding] → [TransformerEncoder]
                                                            ├── state_head: predict next-state features
                                                            ├── stage_head: predict MITRE ATT&CK stage
                                                            └── infiltration_head: P(infiltration)

This is a PyTorch module — it's an optional dependency so the existing
RF classifier continues to work without PyTorch installed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.world_model.features import (
    ATTACK_STAGE_NAMES,
    NUM_ATTACK_STAGES,
    NUM_FEATURES,
)

MODEL_DIR = Path(__file__).parent.parent / "artifacts"
WORLD_MODEL_PATH = MODEL_DIR / "world_model.pt"
WORLD_MODEL_METRICS_PATH = MODEL_DIR / "world_model_metrics.json"

# Committed checkpoint directory. Unlike artifacts/ this is tracked by git,
# so a fresh clone and the evaluator can run the neural path without training.
PRETRAINED_MODEL_DIR = Path(__file__).parent.parent / "pretrained"
PRETRAINED_WORLD_MODEL_PATH = PRETRAINED_MODEL_DIR / WORLD_MODEL_PATH.name


def resolve_world_model_checkpoint() -> Path:
    """Return the checkpoint this deployment should load, or the default path.

    Lookup order:
      1. ml/artifacts/world_model.pt  - a local training run (gitignored)
      2. ml/pretrained/world_model.pt - the committed checkpoint

    Every consumer (the predictor loader, get_model_info and the dashboard
    health endpoint) calls this, so /api/health can never disagree with what
    /api/forecast actually runs. Callers that need to know whether a
    checkpoint exists must test ``.exists()`` on the result.
    """
    for candidate in (WORLD_MODEL_PATH, PRETRAINED_WORLD_MODEL_PATH):
        if candidate.exists():
            return candidate
    return WORLD_MODEL_PATH

# Try importing PyTorch — if unavailable, the module still loads
# but WorldModelTransformer.is_available() returns False.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InfiltrationForecast:
    """Output of the K-step forward simulation."""
    probability_timeline: list[float]       # Infiltration probability per step
    predicted_stages: list[str]             # Predicted MITRE ATT&CK stage per step
    predicted_stage_indices: list[int]      # Numeric stage index per step
    driving_features: list[list[dict]]      # Top features per step (from attention)
    risk_level: str                         # CRITICAL/HIGH/MEDIUM/LOW
    max_infiltration_prob: float            # Peak probability across timeline
    forecast_explanation: str               # Natural-language summary
    attention_weights: list[list[float]] | None = None  # Raw attention weights
    horizon_forecasts: dict[str, float] | None = None   # Multi-horizon anchor predictions (e.g. {'h30': 0.85, ...})


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the World Model. "
            "Install with: pip install torch>=2.0"
        )


# ---------------------------------------------------------------------------
# Model components (only defined if PyTorch is available)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class NetworkStateEncoder(nn.Module):
        """Encode raw feature vectors into a dense latent representation."""

        def __init__(self, n_features: int, d_model: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class PositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding for temporal ordering."""

        def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class WorldModelTransformer(nn.Module):
        """Temporal Transformer that learns network state-transition dynamics.

        Three prediction heads:
            1. state_head:        Predict next-state feature distribution (regression)
            2. stage_head:        Classify into MITRE ATT&CK stage (8-class)
            3. infiltration_head: Binary infiltration probability (sigmoid)
        """

        def __init__(
            self,
            n_features: int = NUM_FEATURES,
            n_stages: int = NUM_ATTACK_STAGES,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.n_stages = n_stages
            self.d_model = d_model

            # Input encoding
            self.state_encoder = NetworkStateEncoder(n_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

            # Prediction heads
            self.state_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_features),
            )

            self.stage_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_stages),
            )

            self.infiltration_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            # Multi-horizon anchor heads (P1-A): direct predictions for
            # specific future horizons to constrain autoregressive drift.
            # Each head outputs (n_stages + 1): stage logits + infiltration prob.
            self.horizon_heads = nn.ModuleDict({
                "h30":  nn.Linear(d_model, n_stages + 1),
                "h60":  nn.Linear(d_model, n_stages + 1),
                "h120": nn.Linear(d_model, n_stages + 1),
                "h300": nn.Linear(d_model, n_stages + 1),
            })
            self.horizon_names = ["h30", "h60", "h120", "h300"]

            # Attention hook storage
            self._attention_weights: list[torch.Tensor] = []
            self._register_attention_hooks()

        def _register_attention_hooks(self):
            """Register forward hooks to capture attention weights."""
            for layer in self.transformer_encoder.layers:
                layer.self_attn.register_forward_hook(self._attention_hook)

        def _attention_hook(self, module, input, output):
            # output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attention_weights.append(output[1].detach())

        def forward(
            self, x: torch.Tensor, return_attention: bool = False
        ) -> dict[str, torch.Tensor]:
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, n_features)

            Returns
            -------
            dict with keys:
                - predicted_state: (batch, seq_len, n_features)
                - stage_logits: (batch, seq_len, n_stages)
                - infiltration_logits: (batch, seq_len, 1)
                - attention_weights: list[Tensor] (optional)
            """
            self._attention_weights = []

            # Encode and add positional information
            encoded = self.state_encoder(x)  # (batch, seq_len, d_model)
            encoded = self.pos_encoder(encoded)

            # Create causal mask so each position can only attend to
            # earlier positions (prevents looking into the future)
            seq_len = x.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            if x.is_cuda:
                causal_mask = causal_mask.to(x.device)

            # Transformer encoding
            hidden = self.transformer_encoder(encoded, mask=causal_mask)

            # Prediction heads
            result = {
                "predicted_state": self.state_head(hidden),
                "stage_logits": self.stage_head(hidden),
                "infiltration_logits": self.infiltration_head(hidden),
            }

            # Multi-horizon anchor predictions
            horizon_preds = {}
            for name in self.horizon_names:
                horizon_preds[name] = self.horizon_heads[name](hidden)
            result["horizon_predictions"] = horizon_preds

            if return_attention:
                result["attention_weights"] = self._attention_weights

            return result

        def get_num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @staticmethod
        def is_available() -> bool:
            return TORCH_AVAILABLE

    # -----------------------------------------------------------------------
    # Loss function
    # -----------------------------------------------------------------------

    class WorldModelLoss(nn.Module):
        """Combined loss for all prediction heads.

        Loss = α·MSE(state) + β·CE(stage) + γ·BCE(infiltration) + δ·horizon_losses

        Primary heads (α+β+γ = 0.80):
            state_head:        Next-state feature regression
            stage_head:        MITRE ATT&CK stage classification
            infiltration_head: Binary infiltration probability

        Horizon anchors (δ = 0.20, split equally across 4 heads):
            h30/h60/h120/h300: Direct predictions at T+1/T+2/T+4/T+10 steps
        """

        # Step offsets for each horizon head (in sequence positions)
        HORIZON_OFFSETS = {"h30": 1, "h60": 2, "h120": 4, "h300": 10}

        def __init__(
            self,
            alpha: float = 0.25,
            beta: float = 0.30,
            gamma: float = 0.25,
            delta: float = 0.20,
        ):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
            self.mse = nn.MSELoss()
            self.ce = nn.CrossEntropyLoss()
            self.bce = nn.BCEWithLogitsLoss()

        def forward(
            self,
            outputs: dict[str, torch.Tensor],
            target_state: torch.Tensor,
            target_stage: torch.Tensor,
            target_infiltration: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            # State prediction loss (MSE) — shifted by 1
            pred_state = outputs["predicted_state"][:, :-1]
            true_state = target_state[:, 1:]
            loss_state = self.mse(pred_state, true_state)

            # Attack stage classification loss (Cross-Entropy)
            stage_logits = outputs["stage_logits"].reshape(-1, outputs["stage_logits"].size(-1))
            true_stages = target_stage.reshape(-1)
            loss_stage = self.ce(stage_logits, true_stages)

            # Infiltration probability loss (Binary Cross-Entropy)
            inf_logits = outputs["infiltration_logits"].squeeze(-1)
            loss_infiltration = self.bce(inf_logits, target_infiltration)

            # Multi-horizon anchor losses
            horizon_loss = torch.tensor(0.0, device=loss_state.device)
            n_horizon = 0
            horizon_preds = outputs.get("horizon_predictions", {})
            seq_len = target_stage.size(1)

            for h_name, offset in self.HORIZON_OFFSETS.items():
                if h_name not in horizon_preds or offset >= seq_len:
                    continue
                h_out = horizon_preds[h_name]  # (batch, seq_len, n_stages+1)
                n_stages = h_out.size(-1) - 1
                # Use target stage shifted forward by offset
                valid_len = seq_len - offset
                h_stage_logits = h_out[:, :valid_len, :n_stages]
                h_stage_targets = target_stage[:, offset:offset + valid_len]
                h_inf_logits = h_out[:, :valid_len, -1]
                h_inf_targets = target_infiltration[:, offset:offset + valid_len]

                h_stage_logits_flat = h_stage_logits.reshape(-1, n_stages)
                h_stage_targets_flat = h_stage_targets.reshape(-1)
                h_loss = self.ce(h_stage_logits_flat, h_stage_targets_flat) + \
                         self.bce(h_inf_logits, h_inf_targets)
                horizon_loss = horizon_loss + h_loss
                n_horizon += 1

            if n_horizon > 0:
                horizon_loss = horizon_loss / n_horizon

            total = (
                self.alpha * loss_state
                + self.beta * loss_stage
                + self.gamma * loss_infiltration
                + self.delta * horizon_loss
            )

            return {
                "total": total,
                "state_loss": loss_state,
                "stage_loss": loss_stage,
                "infiltration_loss": loss_infiltration,
                "horizon_loss": horizon_loss,
            }

else:
    # Fallback stubs when PyTorch is not available
    class WorldModelTransformer:  # type: ignore[no-redef]
        @staticmethod
        def is_available() -> bool:
            return False

    class WorldModelLoss:  # type: ignore[no-redef]
        pass


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Install with: pip install torch>=2.0")
    else:
        model = WorldModelTransformer()
        print(f"World Model parameters: {model.get_num_params():,}")
        print(f"Architecture:\n{model}")

        # Test forward pass
        batch = torch.randn(4, 10, NUM_FEATURES)
        output = model(batch, return_attention=True)
        print(f"\nInput shape:        {batch.shape}")
        print(f"Predicted state:    {output['predicted_state'].shape}")
        print(f"Stage logits:       {output['stage_logits'].shape}")
        print(f"Infiltration:       {output['infiltration_logits'].shape}")
        print(f"Attention layers:   {len(output.get('attention_weights', []))}")
