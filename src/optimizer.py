from dataclasses import asdict
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import (
    EngineConfig,
    WEIGHT_PROFILES,
    resolve_profile,
    validate_weights,
)
from src.pareto import pareto_mask
from src.preprocessing import (
    calculate_features,
    normalize,
    on_time_probability,
    validate_routes,
    validate_shipments,
)


def optimize_routes(
    routes: pd.DataFrame,
    shipment: Mapping | pd.Series,
    profile: str = "auto",
    weights: Mapping[str, float] | None = None,
    config: EngineConfig | None = None,
) -> dict:
    """
    Select from the feasible Pareto frontier.

    If no feasible route exists:
      - recommended_route is None;
      - fallback_candidate is a diagnostic alternative requiring approval.

    Normalization is within this shipment's complete alternative set.
    """
    config = config or EngineConfig()

    clean_shipment = validate_shipments(
        pd.DataFrame([dict(shipment)])
    ).iloc[0].to_dict()

    clean_routes = validate_routes(routes, clean_shipment)

    effective_profile = resolve_profile(clean_shipment, profile)

    effective_weights = validate_weights(
        weights if weights is not None
        else WEIGHT_PROFILES[effective_profile]
    )

    df = calculate_features(clean_routes, config)

    budget = float(clean_shipment["max_budget"])
    deadline = float(clean_shipment["delivery_deadline"])
    reliability_requirement = float(clean_shipment["min_reliability"])

    df["budget_feasible"] = df["total_cost"] <= budget
    df["deadline_feasible"] = df["expected_time"] <= deadline
    df["reliability_feasible"] = (
        df["reliability"] >= reliability_requirement
    )

    df["feasible"] = (
        df["budget_feasible"]
        & df["deadline_feasible"]
        & df["reliability_feasible"]
    )

    df["budget_overrun_usd"] = np.maximum(
        df["total_cost"] - budget,
        0.0,
    )
    df["deadline_overrun_days"] = np.maximum(
        df["expected_time"] - deadline,
        0.0,
    )
    df["reliability_shortfall"] = np.maximum(
        reliability_requirement - df["reliability"],
        0.0,
    )

    # Dimensionless diagnostic fallback measure.
    # This is not an automatic relaxation of the hard constraints.
    df["violation_total"] = (
        df["budget_overrun_usd"] / budget
        + df["deadline_overrun_days"] / deadline
        + df["reliability_shortfall"]
        / max(reliability_requirement, 1e-12)
    )

    df["on_time_probability"] = on_time_probability(
        df,
        deadline=deadline,
        average_delay_days=config.average_delay_days,
    )

    criteria = {
        "cost": "total_cost",
        "time": "expected_time",
        "risk": "composite_risk",
    }

    for criterion, column in criteria.items():
        df[f"{criterion}_norm"] = normalize(df[column])
        df[f"{criterion}_contribution"] = (
            effective_weights[criterion] * df[f"{criterion}_norm"]
        )

    df["score"] = (
        df["cost_contribution"]
        + df["time_contribution"]
        + df["risk_contribution"]
    )

    has_feasible = bool(df["feasible"].any())

    # Pareto membership is scoped to feasible routes when any exist.
    # If none exists, show the unconstrained frontier for diagnosis only.
    scope = df["feasible"] if has_feasible else pd.Series(
        True,
        index=df.index,
    )

    df["pareto_optimal"] = False
    df.loc[scope, "pareto_optimal"] = pareto_mask(df.loc[scope])

    df["eligible"] = df["feasible"] & df["pareto_optimal"]

    if has_feasible:
        df["selection_tier"] = np.select(
            [df["eligible"], df["feasible"]],
            [0, 1],
            default=2,
        )
    else:
        df["selection_tier"] = 2

    # Stable, deterministic tie-breaking by route ID.
    df = df.sort_values(
        [
            "selection_tier",
            "violation_total",
            "score",
            "route_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    recommendation = None
    fallback = None

    if has_feasible:
        recommendation = df.loc[df["eligible"]].iloc[0].to_dict()
    else:
        fallback = df.iloc[0].to_dict()

    return {
        "shipment": clean_shipment,
        "profile": "custom" if weights is not None else effective_profile,
        "default_profile": effective_profile,
        "weights": effective_weights,
        "config": asdict(config),
        "status": "FEASIBLE" if has_feasible else "NO_FEASIBLE_ROUTE",
        "no_feasible_route": not has_feasible,
        "recommended_route": recommendation,
        "fallback_candidate": fallback,
        "pareto_scope": (
            "feasible alternatives"
            if has_feasible
            else "all alternatives; diagnostic only"
        ),
        "ranking": df,
    }