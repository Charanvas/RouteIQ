from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


WEIGHT_PROFILES = {
    "cost": {"cost": 0.60, "time": 0.20, "risk": 0.20},
    "time": {"cost": 0.20, "time": 0.60, "risk": 0.20},
    "risk": {"cost": 0.20, "time": 0.20, "risk": 0.60},
    "balanced": {"cost": 0.40, "time": 0.30, "risk": 0.30},
}


CARGO_DEFAULTS = {
    "Electronics": "balanced",
    "Pharmaceuticals": "risk",
    "Textiles": "cost",
    "Machinery": "cost",
    "Perishable Goods": "time",
}


PRIORITY_DEFAULTS = {
    "Low": "cost",
    "Medium": "balanced",
    "High": "balanced",
    "Critical": "time",
}


@dataclass(frozen=True)
class EngineConfig:
    """Simulation assumptions; these are not calibrated logistics coefficients."""

    average_delay_days: float = 5.0
    congestion_weight: float = 0.35
    geopolitical_weight: float = 0.40
    weather_weight: float = 0.25

    def __post_init__(self):
        values = np.array(
            [
                self.average_delay_days,
                self.congestion_weight,
                self.geopolitical_weight,
                self.weather_weight,
            ],
            dtype=float,
        )

        if not np.isfinite(values).all():
            raise ValueError("Engine configuration must contain finite values.")

        if self.average_delay_days <= 0:
            raise ValueError("average_delay_days must be positive.")

        risk_weights = values[1:]

        if (risk_weights < 0).any():
            raise ValueError("Risk-component weights cannot be negative.")

        if not np.isclose(risk_weights.sum(), 1.0):
            raise ValueError("Risk-component weights must sum to one.")


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if set(weights) != {"cost", "time", "risk"}:
        raise ValueError("Weights must contain exactly: cost, time, risk.")

    try:
        values = np.array(
            [weights["cost"], weights["time"], weights["risk"]],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Weights must be numeric.") from exc

    if not np.isfinite(values).all():
        raise ValueError("Weights must be finite.")

    if (values < 0).any():
        raise ValueError("Weights cannot be negative.")

    if not np.isclose(values.sum(), 1.0):
        raise ValueError("Weights must sum to one.")

    return dict(zip(["cost", "time", "risk"], values.tolist()))


def resolve_profile(shipment, profile: str) -> str:
    if profile != "auto":
        if profile not in WEIGHT_PROFILES:
            raise ValueError(f"Unknown optimization profile: {profile}")
        return profile

    cargo = shipment.get("cargo_type", "")
    priority = shipment.get("shipment_priority", "Medium")

    # Explicit cargo rules take precedence over generic priority rules.
    return CARGO_DEFAULTS.get(
        cargo,
        PRIORITY_DEFAULTS.get(priority, "balanced"),
    )