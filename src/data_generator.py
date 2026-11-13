import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.preprocessing import validate_dataset


PORTS = [
    "Mumbai",
    "Singapore",
    "Dubai",
    "Colombo",
    "Rotterdam",
    "Hamburg",
    "Shanghai",
    "Hong Kong",
    "Durban",
    "New York",
    "Busan",
    "Salalah",
    "Port Said",
    "Antwerp",
]

CARGO_SETTINGS = {
    "Electronics": {
        "priority": "High",
        "min_reliability": 0.88,
        "value": 150_000,
    },
    "Pharmaceuticals": {
        "priority": "Critical",
        "min_reliability": 0.94,
        "value": 240_000,
    },
    "Textiles": {
        "priority": "Medium",
        "min_reliability": 0.0,
        "value": 55_000,
    },
    "Machinery": {
        "priority": "Low",
        "min_reliability": 0.0,
        "value": 180_000,
    },
    "Perishable Goods": {
        "priority": "Critical",
        "min_reliability": 0.88,
        "value": 80_000,
    },
}

# Deliberate trade-offs, rather than independent random columns.
# name, cost multiplier, time multiplier, risk level, intermediate stops
ROUTE_PROFILES = [
    ("Economy", 0.78, 1.25, 0.36, 3),
    ("Express", 1.25, 0.72, 0.18, 1),
    ("Balanced", 1.00, 1.00, 0.23, 2),
    ("High Risk", 0.68, 1.12, 0.70, 2),
    ("Reliable", 1.15, 0.90, 0.07, 2),
]


def generate_dataset(
    n_shipments: int = 50,
    routes_per_shipment: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(n_shipments, int) or n_shipments <= 0:
        raise ValueError("n_shipments must be a positive integer.")

    if routes_per_shipment != 5:
        raise ValueError(
            "This scenario generator defines exactly five route profiles."
        )

    rng = np.random.default_rng(seed)

    shipment_records = []
    route_records = []
    cargo_types = list(CARGO_SETTINGS)

    for index in range(n_shipments):
        shipment_id = f"SHP{index + 1:03d}"

        origin, destination = rng.choice(
            PORTS,
            size=2,
            replace=False,
        ).tolist()

        # Cycling ensures cargo diversity even in smaller datasets.
        cargo_type = cargo_types[index % len(cargo_types)]
        settings = CARGO_SETTINGS[cargo_type]

        base_cost = float(rng.uniform(6_000, 15_000))
        base_time = float(rng.uniform(14, 30))

        candidate_hubs = [
            port
            for port in PORTS
            if port not in {origin, destination}
        ]

        shipment_routes = []

        for route_index, profile in enumerate(ROUTE_PROFILES):
            name, cost_multiplier, time_multiplier, risk_level, stops = profile

            hubs = rng.choice(
                candidate_hubs,
                size=stops,
                replace=False,
            ).tolist()

            congestion = float(
                np.clip(risk_level + rng.normal(0, 0.035), 0.01, 0.98)
            )
            geopolitical = float(
                np.clip(risk_level * 1.05 + rng.normal(0, 0.04), 0.01, 0.98)
            )
            weather = float(
                np.clip(
                    risk_level * 0.80 + rng.normal(0, 0.035),
                    0.01,
                    0.98,
                )
            )

            delay_probability = float(
                np.clip(
                    0.025
                    + 0.36 * congestion
                    + 0.24 * geopolitical
                    + 0.18 * weather
                    + 0.012 * stops
                    + rng.normal(0, 0.008),
                    0.02,
                    0.90,
                )
            )

            reliability = float(
                np.clip(
                    0.995
                    - 0.25 * delay_probability
                    - 0.12 * geopolitical
                    - 0.005 * stops
                    + rng.normal(0, 0.004),
                    0.40,
                    0.995,
                )
            )

            total_cost = (
                base_cost * cost_multiplier * rng.uniform(0.97, 1.03)
                + 170 * stops
            )

            transit_time = (
                base_time * time_multiplier
                + 0.40 * stops
                + 1.50 * congestion
                + rng.uniform(-0.25, 0.25)
            )

            shipment_routes.append(
                {
                    "shipment_id": shipment_id,
                    "route_id": f"{shipment_id}-R{route_index + 1:02d}",
                    "route_profile": name,
                    "intermediate_hubs": "|".join(hubs),
                    "total_cost": round(total_cost, 2),
                    "transit_time": round(transit_time, 3),
                    "delay_probability": round(delay_probability, 4),
                    "congestion_risk": round(congestion, 4),
                    "geopolitical_risk": round(geopolitical, 4),
                    "weather_risk": round(weather, 4),
                    "reliability": round(reliability, 4),
                    "number_of_stops": stops,
                }
            )

        costs = np.array([r["total_cost"] for r in shipment_routes])
        expected_times = np.array(
            [
                r["transit_time"] + 5 * r["delay_probability"]
                for r in shipment_routes
            ]
        )

        max_budget = float(np.quantile(costs, 0.75) * 1.04)
        deadline = float(np.quantile(expected_times, 0.70) * 1.03)

        if cargo_type == "Pharmaceuticals":
            reliable = shipment_routes[4]
            max_budget = max(max_budget, reliable["total_cost"] * 1.05)
            deadline = (
                reliable["transit_time"]
                + 5 * reliable["delay_probability"]
            ) * 1.10

        elif cargo_type == "Perishable Goods":
            deadline = expected_times[1] * 1.08

        scenario = "Standard trade-off"

        # Deliberately construct impossible cases to test honest fallback behavior.
        if index % 11 == 0:
            max_budget = float(costs.min() * 0.90)
            scenario = "No feasible budget"

        elif index % 13 == 0:
            deadline = float(
                min(r["transit_time"] for r in shipment_routes) * 0.85
            )
            scenario = "No feasible deadline"

        shipment_records.append(
            {
                "shipment_id": shipment_id,
                "origin": origin,
                "destination": destination,
                "cargo_type": cargo_type,
                "cargo_value": round(
                    settings["value"] * rng.uniform(0.75, 1.30),
                    2,
                ),
                "shipment_priority": settings["priority"],
                "max_budget": round(max_budget, 2),
                "delivery_deadline": round(deadline, 3),
                "min_reliability": settings["min_reliability"],
                "scenario": scenario,
            }
        )

        route_records.extend(shipment_routes)

    shipments = pd.DataFrame(shipment_records)
    routes = pd.DataFrame(route_records)

    return validate_dataset(
        shipments,
        routes,
        expected_routes=routes_per_shipment,
    )


def save_dataset(
    shipments: pd.DataFrame,
    routes: pd.DataFrame,
    directory: Path = DATA_DIR,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    shipments.to_csv(directory / "shipments.csv", index=False)
    routes.to_csv(directory / "routes.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate RouteIQ synthetic data.")
    parser.add_argument("--shipments", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shipments, routes = generate_dataset(
        n_shipments=args.shipments,
        seed=args.seed,
    )

    save_dataset(shipments, routes)

    print(f"Saved {len(shipments)} shipments and {len(routes)} routes.")
    print(f"Directory: {DATA_DIR}")


if __name__ == "__main__":
    main()