"""
counterfactual.py
-------------------
Action-conditioned counterfactual forward simulation for SOC decision support.

Given a current network state, the simulator runs parallel rollouts:
  - Baseline: What happens if the SOC takes no action?
  - BLOCK_SRC: What if we block the source IP/subnet?
  - BLOCK_DST: What if we block the destination IP/port?
  - THROTTLE: What if we rate-limit the suspicious flow?

The output shows side-by-side risk trajectories with risk deltas, enabling
Tier-3 SOC commanders to evaluate proactive defensive interventions before
committing to them.

This is a deterministic feature-masking approach — no reinforcement learning
required. Each action is modeled as a transformation on the input feature
vector, and the existing World Model predicts the resulting trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ml.world_model.features import WORLD_MODEL_FEATURES, NUM_FEATURES


class MitigationAction(str, Enum):
    """Defensive actions available to the SOC."""
    NO_ACTION = "NO_ACTION"
    BLOCK_SRC = "BLOCK_SRC"       # Block source IP/subnet
    BLOCK_DST_PORT = "BLOCK_DST_PORT"  # Block destination IP/port
    THROTTLE = "THROTTLE"         # Rate-limit suspicious flow


# Feature groups affected by each action
_FEATURE_INDEX = {name: i for i, name in enumerate(WORLD_MODEL_FEATURES)}

_SRC_FEATURES = [
    "total_fwd_packets", "total_fwd_bytes", "fwd_packet_len_mean",
    "fwd_packet_len_std", "fwd_iat_mean", "fwd_iat_std", "fwd_header_len",
    "src_fanout", "unique_src_hosts",
]

_DST_FEATURES = [
    "total_bwd_packets", "total_bwd_bytes", "bwd_packet_len_mean",
    "bwd_packet_len_std", "bwd_iat_mean", "bwd_iat_std", "bwd_header_len",
    "unique_dst_ports_per_src", "dst_fanin", "unique_dst_hosts", "new_dst_ports",
]

_RATE_FEATURES = [
    "flow_packets_per_sec", "flow_bytes_per_sec", "packets_per_flow",
    "syn_flag_count", "ack_flag_count", "connection_repetition",
]


def _apply_action(
    features: np.ndarray, action: MitigationAction | str
) -> np.ndarray:
    """Apply a mitigation action by transforming the feature vector.

    Parameters
    ----------
    features : ndarray of shape (n_features,) or (seq_len, n_features)
        Current network state features.
    action : MitigationAction or str
        The defensive action to simulate.

    Returns
    -------
    Modified feature vector reflecting the action's effect.
    """
    modified = features.copy()
    act_str = action.value if hasattr(action, "value") else str(action)

    if act_str in (MitigationAction.NO_ACTION.value, "NO_ACTION"):
        return modified

    if act_str in (MitigationAction.BLOCK_SRC.value, "BLOCK_SRC"):
        # Zeroing source-related features simulates blocking the attacker
        for fname in _SRC_FEATURES:
            idx = _FEATURE_INDEX.get(fname)
            if idx is not None:
                if modified.ndim == 1:
                    modified[idx] = 0.0
                else:
                    modified[:, idx] = 0.0

    elif act_str in (MitigationAction.BLOCK_DST_PORT.value, "BLOCK_DST_PORT"):
        # Zeroing destination-related features simulates blocking the target
        for fname in _DST_FEATURES:
            idx = _FEATURE_INDEX.get(fname)
            if idx is not None:
                if modified.ndim == 1:
                    modified[idx] = 0.0
                else:
                    modified[:, idx] = 0.0

    elif act_str in (MitigationAction.THROTTLE.value, "THROTTLE"):
        # Reduce rate features by 90% to simulate bandwidth throttling
        for fname in _RATE_FEATURES:
            idx = _FEATURE_INDEX.get(fname)
            if idx is not None:
                if modified.ndim == 1:
                    modified[idx] *= 0.10
                else:
                    modified[:, idx] *= 0.10

    return modified


class CounterfactualResult(dict):
    """Result of a counterfactual simulation supporting both dict and attribute access."""

    def __init__(
        self,
        baseline: dict,
        baseline_risk: float,
        baseline_stages: list[str],
        baseline_timeline: list[float],
        mitigations: dict[str, dict],
        risk_deltas: dict[str, float],
        recommended_action: str,
        max_risk_reduction: float,
        k_steps: int = 5,
    ):
        super().__init__(
            baseline=baseline,
            baseline_risk=baseline_risk,
            baseline_stages=baseline_stages,
            baseline_timeline=baseline_timeline,
            mitigations=mitigations,
            risk_deltas=risk_deltas,
            recommended_action=recommended_action,
            max_risk_reduction=max_risk_reduction,
            k_steps=k_steps,
        )
        self.baseline = baseline
        self.baseline_risk = baseline_risk
        self.baseline_stages = baseline_stages
        self.baseline_timeline = baseline_timeline
        self.mitigations = mitigations
        self.risk_deltas = risk_deltas
        self.recommended_action = recommended_action
        self.max_risk_reduction = max_risk_reduction
        self.k_steps = k_steps

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'CounterfactualResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self[name] = value
        super().__setattr__(name, value)


class CounterfactualSimulator:
    """Action-conditioned forward simulation engine.

    Runs the existing InfiltrationPredictor with modified feature vectors
    to show defenders how each intervention would change the risk trajectory.

    Usage::

        sim = CounterfactualSimulator()
        result = sim.simulate(current_features, k_steps=5)
        print(f"Baseline risk: {result.baseline_risk:.0%}")
        print(f"Best action: {result.recommended_action}")
        print(f"Risk reduction: {result.max_risk_reduction:+.0%}")
    """

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from ml.world_model.predictor import InfiltrationPredictor
            self._predictor = InfiltrationPredictor(model_path=self._model_path)
        return self._predictor

    def simulate(
        self,
        current_features: np.ndarray | dict,
        k_steps: int = 5,
        actions: list[MitigationAction | str] | None = None,
        context_window: np.ndarray | None = None,
    ) -> CounterfactualResult:
        """Run parallel rollouts for baseline vs each mitigation action.

        Parameters
        ----------
        current_features : ndarray or dict
            Current network state features.
        k_steps : int
            Number of future time steps to simulate.
        actions : list of MitigationAction or str, optional
            Actions to evaluate. Defaults to all actions.
        context_window : ndarray, optional
            Optional historical context window.

        Returns
        -------
        CounterfactualResult with baseline and mitigated trajectories.
        """
        predictor = self._get_predictor()

        if actions is None:
            action_items = [
                MitigationAction.BLOCK_SRC,
                MitigationAction.BLOCK_DST_PORT,
                MitigationAction.THROTTLE,
            ]
        else:
            action_items = actions

        # Convert dict to array if needed
        if isinstance(current_features, dict):
            vec = np.zeros(NUM_FEATURES, dtype=np.float32)
            for i, col in enumerate(WORLD_MODEL_FEATURES):
                if col in current_features:
                    vec[i] = float(current_features[col])
            current_features = vec

        current_features = np.asarray(current_features, dtype=np.float32)

        # --- Baseline rollout ---
        baseline = predictor.predict(current_features, k_steps=k_steps, context_window=context_window)

        # --- Mitigated rollouts ---
        mitigations = {}
        risk_deltas = {}
        best_action = MitigationAction.NO_ACTION.value
        max_reduction = 0.0

        for act in action_items:
            action_str = act.value if hasattr(act, "value") else str(act)
            modified = _apply_action(current_features, act)
            mitigated = predictor.predict(modified, k_steps=k_steps, context_window=context_window)

            risk_delta = float(mitigated.max_infiltration_prob - baseline.max_infiltration_prob)

            mitigations[action_str] = {
                "risk": round(mitigated.max_infiltration_prob, 4),
                "stages": mitigated.predicted_stages,
                "timeline": mitigated.probability_timeline,
                "risk_delta": round(risk_delta, 4),
                "risk_level": mitigated.risk_level,
            }
            risk_deltas[action_str] = round(risk_delta, 4)

            if risk_delta < -max_reduction:
                max_reduction = -risk_delta
                best_action = action_str

        baseline_dict = {
            "max_risk": round(baseline.max_infiltration_prob, 4),
            "max_infiltration_prob": round(baseline.max_infiltration_prob, 4),
            "stages": baseline.predicted_stages,
            "predicted_stages": baseline.predicted_stages,
            "timeline": baseline.probability_timeline,
            "probability_timeline": baseline.probability_timeline,
            "risk_level": baseline.risk_level,
            "forecast_explanation": baseline.forecast_explanation,
            "horizon_forecasts": getattr(baseline, "horizon_forecasts", None),
        }

        return CounterfactualResult(
            baseline=baseline_dict,
            baseline_risk=round(baseline.max_infiltration_prob, 4),
            baseline_stages=baseline.predicted_stages,
            baseline_timeline=baseline.probability_timeline,
            mitigations=mitigations,
            risk_deltas=risk_deltas,
            recommended_action=best_action,
            max_risk_reduction=round(max_reduction, 4),
            k_steps=k_steps,
        )

    def simulate_to_dict(
        self,
        current_features: np.ndarray | dict,
        k_steps: int = 5,
        actions: list[MitigationAction | str] | None = None,
        context_window: np.ndarray | None = None,
    ) -> dict:
        """Run simulation and return a JSON-serializable dict.

        This is the function exposed via MCP and the dashboard API.
        """
        result = self.simulate(
            current_features,
            k_steps=k_steps,
            actions=actions,
            context_window=context_window,
        )
        return dict(result)


# Module-level singleton
_simulator: CounterfactualSimulator | None = None


def get_simulator() -> CounterfactualSimulator:
    global _simulator
    if _simulator is None:
        _simulator = CounterfactualSimulator()
    return _simulator


if __name__ == "__main__":
    from ml.world_model.features import _generate_flow_vector

    rng = np.random.default_rng(42)
    features = _generate_flow_vector("PortScan", rng)
    print(f"Input: PortScan flow vector ({features.shape})")

    sim = CounterfactualSimulator()
    result = sim.simulate(features, k_steps=5)

    print(f"\nBaseline risk: {result.baseline_risk:.2%}")
    print(f"Baseline stages: {result.baseline_stages}")

    print(f"\nMitigations:")
    for action, data in result.mitigations.items():
        print(f"  {action:20s}: risk={data['risk']:.2%}  delta={data['risk_delta']:+.2%}")

    print(f"\nRecommended action: {result.recommended_action}")
    print(f"Max risk reduction: {result.max_risk_reduction:.2%}")
"""

Description: Deterministic feature-masking counterfactual simulator. No RL, no
action embeddings — just transparent transformations on the feature vector that
let SOC analysts test "what if I block this IP?" scenarios.
"""
