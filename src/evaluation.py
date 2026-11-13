import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, REPORTS_DIR, EngineConfig
from src.data_generator import generate_dataset, save_dataset
from src.optimizer import optimize_routes
from src.preprocessing import validate_dataset
from src.sensitivity import analyze_profiles


def _percentage(numerator: int, denominator: int):
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 2)


def evaluate_dataset(
    shipments: pd.DataFrame,
    routes: pd.DataFrame,
    profile: str = "auto",
    config: EngineConfig | None = None,
) -> dict:
    shipments, routes = validate_dataset(shipments, routes)
    config = config or EngineConfig()

    decisions = []
    available_count = 0
    constraint_passes = 0
    pareto_passes = 0
    changed_count = 0

    for _, shipment in shipments.iterrows():
        alternatives = routes.loc[
            routes["shipment_id"] == shipment["shipment_id"]
        ]

        result = optimize_routes(
            alternatives,
            shipment,
            profile=profile,
            config=config,
        )

        table = result["ranking"]
        has_feasible = not result["no_feasible_route"]

        if has_feasible:
            available_count += 1

            selected = result["recommended_route"]

            constraint_passes += int(selected["feasible"])
            pareto_passes += int(selected["pareto_optimal"])

            pool = table.loc[table["feasible"]]

            choices = {
                "RouteIQ": selected,
                "Cheapest": pool.sort_values(
                    ["total_cost", "route_id"]
                ).iloc[0].to_dict(),
                "Fastest base time": pool.sort_values(
                    ["transit_time", "route_id"]
                ).iloc[0].to_dict(),
                "Lowest risk": pool.sort_values(
                    ["composite_risk", "route_id"]
                ).iloc[0].to_dict(),
            }

        else:
            # Excluded from feasible-route performance averages.
            choices = {
                name: None
                for name in [
                    "RouteIQ",
                    "Cheapest",
                    "Fastest base time",
                    "Lowest risk",
                ]
            }

        for method, selected in choices.items():
            decisions.append(
                {
                    "shipment_id": shipment["shipment_id"],
                    "method": method,
                    "status": result["status"],
                    "route_id": (
                        selected["route_id"] if selected else None
                    ),
                    "feasible": bool(selected is not None),
                    "cost": (
                        selected["total_cost"] if selected else None
                    ),
                    "expected_time": (
                        selected["expected_time"] if selected else None
                    ),
                    "risk": (
                        selected["composite_risk"] if selected else None
                    ),
                    "score": (
                        selected["score"] if selected else None
                    ),
                    "pareto_optimal": (
                        bool(selected["pareto_optimal"])
                        if selected else None
                    ),
                }
            )

        if has_feasible:
            profile_summary, _ = analyze_profiles(
                alternatives,
                shipment,
                config=config,
            )

            if profile_summary["route_id"].nunique() > 1:
                changed_count += 1

    decisions_df = pd.DataFrame(decisions)
    valid = decisions_df.loc[decisions_df["feasible"]]

    baseline_summary = (
        valid.groupby("method", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_cost=("cost", "mean"),
            mean_expected_time=("expected_time", "mean"),
            mean_risk=("risk", "mean"),
            mean_weighted_score=("score", "mean"),
        )
        .round(4)
    )

    routeiq = valid.loc[valid["method"] == "RouteIQ"]

    mean_score = (
        float(routeiq["score"].mean())
        if not routeiq.empty else None
    )

    summary = {
        "shipments_evaluated": len(shipments),
        "route_alternatives": len(routes),
        "shipments_with_feasible_routes": available_count,
        "no_feasible_shipments": len(shipments) - available_count,
        "feasible_recommendations": constraint_passes,
        "constraint_satisfaction_rate_pct": _percentage(
            constraint_passes,
            available_count,
        ),
        "feasible_recommendation_coverage_pct": _percentage(
            constraint_passes,
            len(shipments),
        ),
        "pareto_efficiency_rate_pct": _percentage(
            pareto_passes,
            available_count,
        ),
        "profile_sensitive_shipments": changed_count,
        "profile_change_rate_pct": _percentage(
            changed_count,
            available_count,
        ),
        "average_routeiq_score": mean_score,
        "profile": profile,
        "notes": [
            "Constraint satisfaction denominator includes only shipments "
            "with at least one feasible alternative.",
            "No-feasible shipments are reported separately and excluded "
            "from baseline averages.",
            "All baseline methods use the same feasible alternative pool.",
            "Fastest baseline minimizes base transit time.",
            "Scores use within-shipment normalization and are descriptive; "
            "they are not real-world utility or monetary savings.",
            "Synthetic evaluation demonstrates internal consistency, "
            "not deployment accuracy.",
        ],
    }

    return {
        "summary": summary,
        "decisions": decisions_df,
        "baseline_summary": baseline_summary,
    }


def save_evaluation(result: dict, directory: Path = REPORTS_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "evaluation_summary.json").write_text(
        json.dumps(result["summary"], indent=2, allow_nan=False),
        encoding="utf-8",
    )

    result["decisions"].to_csv(
        directory / "evaluation_decisions.csv",
        index=False,
    )

    result["baseline_summary"].to_csv(
        directory / "baseline_summary.csv",
        index=False,
    )


def load_or_generate_dataset():
    shipment_path = DATA_DIR / "shipments.csv"
    route_path = DATA_DIR / "routes.csv"

    if shipment_path.exists() != route_path.exists():
        raise ValueError(
            "Only one dataset file exists. Regenerate both CSV files."
        )

    if not shipment_path.exists():
        shipments, routes = generate_dataset()
        save_dataset(shipments, routes)
        return shipments, routes

    return validate_dataset(
        pd.read_csv(shipment_path),
        pd.read_csv(route_path),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RouteIQ.")
    parser.add_argument(
        "--profile",
        choices=["auto", "cost", "time", "risk", "balanced"],
        default="auto",
    )
    args = parser.parse_args()

    shipments, routes = load_or_generate_dataset()

    result = evaluate_dataset(
        shipments,
        routes,
        profile=args.profile,
    )

    save_evaluation(result)

    print(json.dumps(result["summary"], indent=2))
    print("\nBaseline comparison:")
    print(result["baseline_summary"].to_string(index=False))


if __name__ == "__main__":
    main()