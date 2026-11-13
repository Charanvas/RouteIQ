import pandas as pd

from src.config import EngineConfig, WEIGHT_PROFILES
from src.optimizer import optimize_routes


def _selected(result: dict) -> dict:
    return (
        result["recommended_route"]
        or result["fallback_candidate"]
    )


def analyze_profiles(
    routes: pd.DataFrame,
    shipment,
    config: EngineConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    score_rows = []

    balanced = optimize_routes(
        routes,
        shipment,
        profile="balanced",
        config=config,
    )

    balanced_route_id = _selected(balanced)["route_id"]

    for profile in WEIGHT_PROFILES:
        result = optimize_routes(
            routes,
            shipment,
            profile=profile,
            config=config,
        )

        selected = _selected(result)

        summary_rows.append(
            {
                "profile": profile,
                "route_id": selected["route_id"],
                "status": result["status"],
                "changed_vs_balanced": (
                    selected["route_id"] != balanced_route_id
                ),
                "cost": selected["total_cost"],
                "expected_time": selected["expected_time"],
                "risk": selected["composite_risk"],
                "score": selected["score"],
            }
        )

        for _, route in result["ranking"].iterrows():
            score_rows.append(
                {
                    "profile": profile,
                    "route_id": route["route_id"],
                    "score": route["score"],
                    "feasible": bool(route["feasible"]),
                    "eligible": bool(route["eligible"]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(score_rows)


def explain_priority_change(
    previous: dict,
    current: dict,
) -> str:
    """
    Explain a weight-only change.

    This decomposition is valid because criteria, normalization,
    constraints, and alternative sets remain unchanged.
    """
    if previous["no_feasible_route"] or current["no_feasible_route"]:
        return (
            "No feasible recommendation exists in at least one scenario. "
            "Weight changes do not repair hard-constraint violations."
        )

    old_id = previous["recommended_route"]["route_id"]
    new_id = current["recommended_route"]["route_id"]

    if old_id == new_id:
        return (
            f"The recommendation remains {new_id}. "
            "It is stable under this change in business priorities."
        )

    table = current["ranking"].set_index("route_id")
    old_route = table.loc[old_id]
    new_route = table.loc[new_id]

    effects = {}

    for criterion in ["cost", "time", "risk"]:
        weight_change = (
            current["weights"][criterion]
            - previous["weights"][criterion]
        )

        normalized_difference = (
            new_route[f"{criterion}_norm"]
            - old_route[f"{criterion}_norm"]
        )

        effects[criterion] = weight_change * normalized_difference

    driver = min(effects, key=effects.get)

    return (
        f"The recommendation changes from {old_id} to {new_id}. "
        f"The largest favorable change in their relative weighted "
        f"score comes from {driver} "
        f"({effects[driver]:+.4f} score units). "
        "Negative values favor the newly selected route."
    )