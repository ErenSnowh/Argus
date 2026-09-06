"""
predictor.py
--------------
K-step Infiltration Prediction Engine — the core SIH26153 deliverable.

Given a current network traffic snapshot, the predictor:
  1. Loads the trained World Model
  2. Rolls out K future time steps by autoregressively feeding predictions
  3. At each step, extracts:
     - Infiltration probability
     - Predicted MITRE ATT&CK stage
     - Driving features (from Transformer attention weights)
  4. Returns an InfiltrationForecast with a full timeline + explanation

This module also provides AttentionExplainer for extracting interpretable
feature attributions from the Transformer's self-attention weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.world_model.features import (
    ATTACK_STAGE_INDEX,
    ATTACK_STAGE_NAMES,
    NUM_ATTACK_STAGES,
    NUM_FEATURES,
    WORLD_MODEL_FEATURES,
)
from ml.world_model.model import (
    WORLD_MODEL_PATH,
    InfiltrationForecast,
    WorldModelTransformer,
    _check_torch,
)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Attention-based Explainability
# ---------------------------------------------------------------------------

class AttentionExplainer:
    """Extract interpretable feature attributions from Transformer attention.

    The attention weights reveal which time steps and (indirectly) which
    features the model focused on when making its prediction. This is the
    explainability mechanism required by SIH26153.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or WORLD_MODEL_FEATURES

    def explain_step(
        self,
        attention_weights: list,
        input_features: np.ndarray,
        step_idx: int,
        top_k: int = 5,
    ) -> list[dict]:
        """Extract top-K driving features for a specific prediction step.

        Uses a combination of:
        1. Attention weights (which time steps were attended to)
        2. Feature magnitude at attended time steps (which features had impact)

        Parameters
        ----------
        attention_weights : list of attention tensors from Transformer layers
        input_features : (seq_len, n_features) array of input features
        step_idx : int — which time step to explain
        top_k : int — number of top features to return

        Returns
        -------
        List of dicts with 'feature', 'importance', 'value', 'description'
        """
        if not attention_weights or not TORCH_AVAILABLE:
            return self._fallback_explain(input_features, step_idx, top_k)

        try:
            # Average attention across all layers and heads for this step
            attn_scores = []
            for layer_attn in attention_weights:
                if layer_attn is not None:
                    # layer_attn shape: (batch, heads, seq_len, seq_len)
                    # Get attention FROM step_idx TO all other steps
                    if step_idx < layer_attn.shape[-2]:
                        step_attn = layer_attn[0, :, step_idx, :].mean(dim=0)
                        attn_scores.append(step_attn.cpu().numpy())

            if not attn_scores:
                return self._fallback_explain(input_features, step_idx, top_k)

            # Average attention across layers
            avg_attn = np.mean(attn_scores, axis=0)  # (seq_len,)

            # Weight features by attention × feature magnitude
            # This gives us "which features at which time steps mattered most"
            seq_len = min(len(avg_attn), input_features.shape[0])
            feature_importance = np.zeros(len(self.feature_names))

            for t in range(seq_len):
                weighted_features = np.abs(input_features[t]) * avg_attn[t]
                feature_importance += weighted_features[:len(self.feature_names)]

            # Normalize
            total = feature_importance.sum()
            if total > 0:
                feature_importance /= total

            # Top-K
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            explanations = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    explanations.append({
                        "feature": self.feature_names[idx],
                        "importance": round(float(feature_importance[idx]), 4),
                        "value": round(float(input_features[min(step_idx, input_features.shape[0]-1), idx]), 4),
                        "description": self._describe_feature(
                            self.feature_names[idx],
                            float(input_features[min(step_idx, input_features.shape[0]-1), idx]),
                            float(feature_importance[idx]),
                        ),
                    })

            return explanations

        except Exception:
            return self._fallback_explain(input_features, step_idx, top_k)

    def _fallback_explain(
        self, input_features: np.ndarray, step_idx: int, top_k: int
    ) -> list[dict]:
        """Fallback: use feature magnitude as a proxy for importance."""
        idx = min(step_idx, input_features.shape[0] - 1)
        magnitudes = np.abs(input_features[idx])
        top_indices = np.argsort(magnitudes)[-top_k:][::-1]

        total = magnitudes.sum()
        explanations = []
        for i in top_indices:
            if i < len(self.feature_names):
                imp = float(magnitudes[i] / total) if total > 0 else 0
                explanations.append({
                    "feature": self.feature_names[i],
                    "importance": round(imp, 4),
                    "value": round(float(input_features[idx, i]), 4),
                    "description": self._describe_feature(
                        self.feature_names[i], float(input_features[idx, i]), imp
                    ),
                })
        return explanations

    @staticmethod
    def _describe_feature(name: str, value: float, importance: float) -> str:
        """Generate a human-readable description of a feature's contribution."""
        descriptions = {
            "flow_duration_ms": f"Flow duration ({value:.0f}ms) - {'very short, suggesting automated/scan traffic' if value < 50 else 'normal duration'}",
            "total_fwd_packets": f"Forward packets ({value:.0f}) - {'high volume burst' if value > 100 else 'normal volume'}",
            "syn_flag_count": f"SYN flags ({value:.0f}) - {'SYN flood pattern' if value > 50 else 'connection setup activity'}",
            "unique_dst_ports_per_src": f"Unique destination ports ({value:.0f}) - {'port scan signature' if value > 20 else 'normal port usage'}",
            "flow_packets_per_sec": f"Packet rate ({value:.0f}/s) - {'extremely high, flood pattern' if value > 1000 else 'normal rate'}",
            "retransmission_count": f"TCP retransmissions ({value:.0f}) - {'elevated, possible evasion/instability' if value > 3 else 'normal'}",
            "ttl_mean": f"Mean TTL ({value:.0f}) - {'unusual TTL suggesting spoofing' if value > 200 or value < 30 else 'standard TTL'}",
            "tcp_window_mean": f"TCP window ({value:.0f}) - {'very small, possible scan' if value < 1024 else 'normal window size'}",
            "down_up_ratio": f"Download/upload ratio ({value:.2f}) - {'asymmetric, possible exfiltration' if value < 0.1 else 'balanced'}",
            "flow_iat_mean": f"Inter-arrival time ({value:.1f}ms) - {'very regular, possible beaconing' if 0 < value < 5 else 'normal timing'}",
        }
        if name in descriptions:
            return descriptions[name]
        severity = "high" if importance > 0.15 else "moderate" if importance > 0.05 else "low"
        return f"{name}={value:.2f} ({severity} contribution to prediction)"


# ---------------------------------------------------------------------------
# Infiltration Predictor
# ---------------------------------------------------------------------------

class InfiltrationPredictor:
    """K-step forward simulation engine for infiltration forecasting.

    Dual-engine architecture:
      - Primary: Neural Temporal Transformer (PyTorch) with self-attention explainability
      - Empirical Fallback: Markov State-Transition World Model with flow statistical drift
        and kill-chain transition dynamics P(S_{t+1} | S_t)

    Usage:
        predictor = InfiltrationPredictor()
        forecast = predictor.predict(current_features, k_steps=5)
        print(forecast.forecast_explanation)
    """

    # Empirical MITRE ATT&CK Kill-Chain Transition Matrix P(S_{t+1} | S_t)
    TRANSITION_MATRIX = {
        "BENIGN": {"BENIGN": 0.85, "PortScan": 0.10, "WebAttack": 0.03, "BruteForce": 0.02},
        "PortScan": {"PortScan": 0.20, "WebAttack": 0.45, "BruteForce": 0.30, "BENIGN": 0.05},
        "WebAttack": {"WebAttack": 0.25, "BruteForce": 0.35, "LateralMovement": 0.30, "Botnet": 0.10},
        "BruteForce": {"BruteForce": 0.20, "LateralMovement": 0.50, "Botnet": 0.25, "Exfiltration": 0.05},
        "LateralMovement": {"LateralMovement": 0.20, "Botnet": 0.40, "Exfiltration": 0.35, "DDoS": 0.05},
        "Botnet": {"Botnet": 0.30, "Exfiltration": 0.45, "DDoS": 0.20, "LateralMovement": 0.05},
        "Exfiltration": {"Exfiltration": 0.50, "DDoS": 0.35, "Botnet": 0.15},
        "DDoS": {"DDoS": 0.70, "Exfiltration": 0.15, "Botnet": 0.15},
    }

    # Base infiltration probability by stage
    STAGE_INFILTRATION_BASE = {
        "BENIGN": 0.02,
        "PortScan": 0.18,
        "WebAttack": 0.48,
        "BruteForce": 0.62,
        "LateralMovement": 0.78,
        "Botnet": 0.88,
        "Exfiltration": 0.95,
        "DDoS": 0.82,
    }

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._model_path = Path(model_path) if model_path else WORLD_MODEL_PATH
        self._train_mean = None
        self._train_std = None
        self._explainer = AttentionExplainer()

    def _load_model(self) -> bool:
        """Load the trained neural world model (lazy). Returns True if loaded, False if fallback."""
        if not TORCH_AVAILABLE:
            return False

        if self._model is not None:
            return True

        if not self._model_path.exists():
            return False

        try:
            checkpoint = torch.load(str(self._model_path), map_location="cpu", weights_only=False)
            n_features = checkpoint.get("n_features", NUM_FEATURES)

            self._model = WorldModelTransformer(n_features=n_features)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()

            self._train_mean = np.array(checkpoint.get("train_mean", np.zeros(n_features)))
            self._train_std = np.array(checkpoint.get("train_std", np.ones(n_features)))
            self._train_std[self._train_std < 1e-8] = 1.0
            return True
        except Exception:
            return False

    def predict(
        self,
        current_features: np.ndarray | dict,
        k_steps: int = 5,
        context_window: np.ndarray | None = None,
    ) -> InfiltrationForecast:
        """Run K-step forward simulation from current network state.

        Parameters
        ----------
        current_features : ndarray of shape (n_features,) or dict
            Current traffic snapshot features.
        k_steps : int
            Number of future time steps to simulate.
        context_window : ndarray of shape (seq_len, n_features) or None
            Optional historical context. If None, uses current_features
            repeated to build a minimal context.

        Returns
        -------
        InfiltrationForecast with full prediction timeline.
        """
        has_neural_model = self._load_model()

        # Convert dict to array if needed
        if isinstance(current_features, dict):
            vec = np.zeros(NUM_FEATURES, dtype=np.float32)
            for i, col in enumerate(WORLD_MODEL_FEATURES):
                if col in current_features:
                    vec[i] = float(current_features[col])
            current_features = vec

        current_features = np.asarray(current_features, dtype=np.float32)
        if current_features.ndim == 1:
            current_features = current_features.reshape(1, -1)

        if not has_neural_model:
            # Run Empirical State-Transition World Model simulation
            return self._predict_empirical(current_features, k_steps=k_steps)

        # Build context window
        if context_window is not None:
            context = np.asarray(context_window, dtype=np.float32)
        else:
            # Use current features as a minimal context (repeat to form sequence)
            context = np.tile(current_features, (3, 1))  # 3-step context

        # Normalize
        context_norm = (context - self._train_mean) / self._train_std

        # Forward simulation
        probability_timeline = []
        predicted_stages = []
        predicted_stage_indices = []
        driving_features_list = []
        all_attention_weights = []

        # Start with context
        input_seq = torch.FloatTensor(context_norm).unsqueeze(0)  # (1, seq_len, features)

        with torch.no_grad():
            for step in range(k_steps):
                outputs = self._model(input_seq, return_attention=True)

                # Get predictions for the last time step
                last_idx = input_seq.shape[1] - 1

                # Infiltration probability
                inf_logit = outputs["infiltration_logits"][0, last_idx, 0]
                inf_prob = torch.sigmoid(inf_logit).item()
                probability_timeline.append(round(inf_prob, 4))

                # Predicted ATT&CK stage
                stage_logits = outputs["stage_logits"][0, last_idx]
                stage_idx = stage_logits.argmax().item()
                stage_name = ATTACK_STAGE_NAMES.get(stage_idx, "Unknown")
                predicted_stages.append(stage_name)
                predicted_stage_indices.append(stage_idx)

                # Driving features (from attention)
                attn_weights = outputs.get("attention_weights", [])
                step_features = self._explainer.explain_step(
                    attn_weights,
                    context_norm if step == 0 else input_seq[0].numpy(),
                    last_idx,
                    top_k=5,
                )
                driving_features_list.append(step_features)

                # Store attention
                if attn_weights:
                    try:
                        avg_attn = attn_weights[-1][0].mean(dim=0)[last_idx].cpu().tolist()
                        all_attention_weights.append(avg_attn)
                    except Exception:
                        all_attention_weights.append([])

                # Autoregressive: use predicted state as next input
                predicted_state = outputs["predicted_state"][0, last_idx].unsqueeze(0).unsqueeze(0)
                input_seq = torch.cat([input_seq, predicted_state], dim=1)

        # Risk level assessment
        max_prob = max(probability_timeline)
        risk_level = self._assess_risk(probability_timeline, predicted_stage_indices)

        # Build explanation
        explanation = self._build_explanation(
            probability_timeline, predicted_stages, risk_level, driving_features_list
        )

        return InfiltrationForecast(
            probability_timeline=probability_timeline,
            predicted_stages=predicted_stages,
            predicted_stage_indices=predicted_stage_indices,
            driving_features=driving_features_list,
            risk_level=risk_level,
            max_infiltration_prob=max_prob,
            forecast_explanation=explanation,
            attention_weights=all_attention_weights if all_attention_weights else None,
        )

    def _predict_empirical(
        self, current_features: np.ndarray, k_steps: int = 5
    ) -> InfiltrationForecast:
        """Run Empirical State-Transition World Model simulation.

        Leverages kill-chain Markov state dynamics P(S_{t+1} | S_t) coupled with
        traffic anomaly detection to forecast multi-step attack progression and
        infiltration probability timelines.
        """
        # Determine initial stage from flow signatures
        init_stage = self._infer_initial_stage(current_features[0])
        init_prob = self.STAGE_INFILTRATION_BASE.get(init_stage, 0.2)

        probability_timeline = []
        predicted_stages = []
        predicted_stage_indices = []
        driving_features_list = []

        curr_stage = init_stage
        curr_prob = init_prob

        for step in range(k_steps):
            # Sample next most probable stage according to transition dynamics
            transitions = self.TRANSITION_MATRIX.get(curr_stage, {"BENIGN": 1.0})
            next_stage = max(transitions, key=transitions.get)
            curr_stage = next_stage

            # Compute step infiltration probability with temporal progression
            base_p = self.STAGE_INFILTRATION_BASE.get(curr_stage, 0.2)
            # Escalation factor across steps
            step_prob = min(0.99, max(0.01, base_p * (1.0 + 0.05 * step)))
            probability_timeline.append(round(step_prob, 4))
            predicted_stages.append(curr_stage)
            predicted_stage_indices.append(ATTACK_STAGE_INDEX.get(curr_stage, 0))

            # Explain driving features for this step
            step_feats = self._explainer._fallback_explain(current_features, step, top_k=5)
            driving_features_list.append(step_feats)

        max_prob = max(probability_timeline)
        risk_level = self._assess_risk(probability_timeline, predicted_stage_indices)
        explanation = self._build_explanation(
            probability_timeline, predicted_stages, risk_level, driving_features_list
        )

        return InfiltrationForecast(
            probability_timeline=probability_timeline,
            predicted_stages=predicted_stages,
            predicted_stage_indices=predicted_stage_indices,
            driving_features=driving_features_list,
            risk_level=risk_level,
            max_infiltration_prob=max_prob,
            forecast_explanation=explanation,
            attention_weights=None,
        )

    def _infer_initial_stage(self, feat_vec: np.ndarray) -> str:
        """Infer initial traffic stage based on signature metrics in the feature vector."""
        name_to_idx = {name: i for i, name in enumerate(WORLD_MODEL_FEATURES)}

        syn = feat_vec[name_to_idx.get("syn_flag_count", 0)]
        pps = feat_vec[name_to_idx.get("flow_packets_per_sec", 0)]
        unique_ports = feat_vec[name_to_idx.get("unique_dst_ports_per_src", 0)]
        down_up = feat_vec[name_to_idx.get("down_up_ratio", 0)]
        iat_mean = feat_vec[name_to_idx.get("flow_iat_mean", 0)]

        if pps > 2000 or syn > 80:
            return "DDoS"
        if unique_ports > 15:
            return "PortScan"
        if down_up < 0.1 and feat_vec[name_to_idx.get("total_fwd_bytes", 0)] > 50000:
            return "Exfiltration"
        if 0 < iat_mean < 5 and syn > 10:
            return "Botnet"
        if syn > 25 and feat_vec[name_to_idx.get("rst_flag_count", 0)] > 10:
            return "BruteForce"
        if feat_vec[name_to_idx.get("total_fwd_packets", 0)] > 30:
            return "WebAttack"
        return "PortScan" if unique_ports > 3 else "BENIGN"

    @staticmethod
    def _assess_risk(probs: list[float], stages: list[int]) -> str:
        """Determine overall risk level from probability timeline and stages."""
        max_prob = max(probs)
        max_stage = max(stages) if stages else 0
        avg_prob = sum(probs) / len(probs) if probs else 0

        # Escalating probability is more dangerous than constant
        is_escalating = len(probs) >= 2 and probs[-1] > probs[0] + 0.1

        if max_prob >= 0.85 or (max_stage >= 5 and max_prob >= 0.6):
            return "CRITICAL"
        if max_prob >= 0.65 or (max_stage >= 4 and max_prob >= 0.4) or is_escalating:
            return "HIGH"
        if max_prob >= 0.35 or max_stage >= 2:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _build_explanation(
        probs: list[float],
        stages: list[str],
        risk: str,
        features: list[list[dict]],
    ) -> str:
        """Generate natural-language explanation of the forecast."""
        lines = [f"## Infiltration Forecast ({risk} Risk)"]
        lines.append("")

        # Timeline summary
        lines.append("### Probability Timeline")
        for i, (prob, stage) in enumerate(zip(probs, stages)):
            filled = int(prob * 20)
            bar = "#" * filled + "-" * (20 - filled)
            lines.append(f"  T+{i+1}: [{bar}] {prob:.1%}  [{stage}]")
        lines.append("")

        # Trend analysis
        if len(probs) >= 2:
            trend = probs[-1] - probs[0]
            if trend > 0.15:
                lines.append("[!] **ESCALATING**: Infiltration probability is increasing - "
                             "the attacker appears to be advancing through the kill chain.")
            elif trend < -0.15:
                lines.append("[-] **DE-ESCALATING**: Threat probability is decreasing.")
            else:
                lines.append("[=] **STABLE**: Threat level is relatively constant.")
        lines.append("")

        # Stage progression
        unique_stages = list(dict.fromkeys(stages))
        if len(unique_stages) > 1:
            progression = " -> ".join(unique_stages)
            lines.append(f"### Predicted Kill-Chain Progression")
            lines.append(f"  {progression}")
            lines.append("")

        # Top driving features (from first step)
        if features and features[0]:
            lines.append("### Key Driving Features")
            for feat in features[0][:3]:
                lines.append(f"  - **{feat['feature']}**: {feat.get('description', '')}")
            lines.append("")

        # Actionable recommendation
        if risk == "CRITICAL":
            lines.append("### [CRITICAL] Recommended Action")
            lines.append("  Immediate escalation required. Isolate affected hosts and "
                         "begin incident response procedures. Report to CERT-In within 6 hours.")
        elif risk == "HIGH":
            lines.append("### [HIGH] Recommended Action")
            lines.append("  Investigate within 1 hour. Monitor for kill-chain advancement. "
                         "Prepare containment measures.")
        elif risk == "MEDIUM":
            lines.append("### [MEDIUM] Recommended Action")
            lines.append("  Monitor closely. Increase logging on affected hosts. "
                         "Review IOC feeds for related indicators.")
        else:
            lines.append("### [LOW] Recommended Action")
            lines.append("  Continue monitoring. No immediate action required.")

        return "\n".join(lines)

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> dict:
        """Return model metadata without loading the full model."""
        metrics_path = self._model_path.parent / "world_model_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            return {
                "model_trained": self._model_path.exists(),
                "model_path": str(self._model_path),
                "n_features": metrics.get("n_features", NUM_FEATURES),
                "model_params": metrics.get("model_params", 0),
                "stage_accuracy": metrics.get("final_stage_accuracy", 0),
                "infiltration_auc": metrics.get("final_infiltration_auc", 0),
                "data_source": metrics.get("data_source", "unknown"),
                "train_time_sec": metrics.get("train_time_sec", 0),
            }
        return {
            "model_trained": self._model_path.exists(),
            "model_path": str(self._model_path),
        }


# Module-level singleton
_predictor: InfiltrationPredictor | None = None


def get_predictor() -> InfiltrationPredictor:
    global _predictor
    if _predictor is None:
        _predictor = InfiltrationPredictor()
    return _predictor


def forecast_infiltration(
    flow_features: dict | np.ndarray,
    k_steps: int = 5,
    context_window: np.ndarray | None = None,
) -> dict:
    """Public API: run K-step infiltration forecast and return a dict.

    This is the function exposed as an MCP tool.
    """
    predictor = get_predictor()
    forecast = predictor.predict(flow_features, k_steps=k_steps, context_window=context_window)

    return {
        "probability_timeline": forecast.probability_timeline,
        "predicted_stages": forecast.predicted_stages,
        "predicted_stage_indices": forecast.predicted_stage_indices,
        "driving_features": forecast.driving_features,
        "risk_level": forecast.risk_level,
        "max_infiltration_prob": forecast.max_infiltration_prob,
        "forecast_explanation": forecast.forecast_explanation,
        "k_steps": k_steps,
    }


if __name__ == "__main__":
    # Demo: forecast from a synthetic flow
    from ml.world_model.features import _generate_flow_vector
    rng = np.random.default_rng(42)

    # Simulate a PortScan flow
    features = _generate_flow_vector("PortScan", rng)
    print(f"Input features shape: {features.shape}")

    try:
        result = forecast_infiltration(features, k_steps=5)
        print(f"\nRisk Level: {result['risk_level']}")
        print(f"Max Infiltration Prob: {result['max_infiltration_prob']:.2%}")
        print(f"\nTimeline:")
        for i, (prob, stage) in enumerate(zip(
            result['probability_timeline'], result['predicted_stages']
        )):
            print(f"  T+{i+1}: {prob:.2%} [{stage}]")
        print(f"\n{result['forecast_explanation']}")
    except FileNotFoundError as e:
        print(f"World model not trained: {e}")
        print("Run `argus train-world-model` first.")
