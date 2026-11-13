"""Build synthetic alternatives for a user-defined shipment.

This module does not query shipping schedules or solve a real transport
network. It creates reproducible decision-model scenarios.
"""

import hashlib
import json
import math

import numpy as np
import pandas as pd

from src.data_generator import (
    CARGO_SETTINGS,
    PORTS,
    generate_dataset,
)
from src.preprocessing import validate_dataset


# Illustrative simulation assumptions, not real insurance rates.
CARGO_VALUE_SURCHARGE_RATES = {
    "Electronics": 0.0015,
    "Pharmaceuticals": 0.0020,
    "Textiles": 0.0008,
    "Machinery": 0.0010,
    "Perishable Goods": 0.0018,
}

# Illustrative handling allowances, not carrier quotations.
CARGO_HANDLING_FEES = {
    "Electronics": 150.0,
    "Pharmaceuticals": 650.0,
    "Textiles": 50.0,
    "Machinery": 350.0,
    "Perishable Goods": 450.0,
}


def _stable_digest(payload: dict) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _positive_number(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive number."
        ) from exc

    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )

    return number


def plan_shipment(
    origin: str,
    destination: str,
    cargo_type: str,
    cargo_value: float,
    max_budget: float | None = None,
    delivery_deadline: float | None = None,
    min_reliability: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Create a shipment and five synthetic route alternatives.

    The corridor controls the base simulation seed.

    Cargo value affects disclosed simulated surcharges, not the
    corridor's geography, delay estimates, or risk indicators.

    Budget/deadline overrides affect feasibility, not route generation.
    """
    if origin not in PORTS:
        raise ValueError("Unsupported origin.")

    if destination not in PORTS:
        raise ValueError("Unsupported destination.")

    if origin == destination:
        raise ValueError("Origin and destination must differ.")

    if cargo_type not in CARGO_SETTINGS:
        raise ValueError("Unsupported cargo type.")

    cargo_value = _positive_number(cargo_value, "cargo_value")

    corridor_digest = _stable_digest(
        {
            "origin": origin,
            "destination": destination,
            "scenario_version": 1,
        }
    )

    corridor_seed = int(corridor_digest[:8], 16)

    # Reuse the existing five-profile simulation.
    # Its generated shipment requirements are intentionally discarded.
    _, base_routes = generate_dataset(
        n_shipments=1,
        seed=corridor_seed,
    )

    routes = base_routes.copy()

    # Rebuild intermediate nodes for the user's endpoints.
    # These are synthetic node sequences, not verified maritime paths.
    available_hubs = [
        port
        for port in PORTS
        if port not in {origin, destination}
    ]

    hub_rng = np.random.default_rng(corridor_seed)

    routes["intermediate_hubs"] = [
        "|".join(
            hub_rng.choice(
                available_hubs,
                size=int(stops),
                replace=False,
            ).tolist()
        )
        for stops in routes["number_of_stops"]
    ]

    settings = CARGO_SETTINGS[cargo_type]

    routes["base_transport_cost"] = routes["total_cost"]

    handling_fee = CARGO_HANDLING_FEES[cargo_type]
    value_surcharge = round(
        cargo_value * CARGO_VALUE_SURCHARGE_RATES[cargo_type],
        2,
    )

    routes["cargo_handling_fee"] = handling_fee
    routes["cargo_value_surcharge"] = value_surcharge

    routes["total_cost"] = (
        routes["base_transport_cost"]
        + routes["cargo_handling_fee"]
        + routes["cargo_value_surcharge"]
    ).round(2)

    # Defaults use the project's baseline five-day mean delay.
    # They remain fixed if the operator later changes that assumption.
    default_expected_time = (
        routes["transit_time"]
        + 5.0 * routes["delay_probability"]
    )

    suggested_budget = round(
        float(routes["total_cost"].quantile(0.75)) * 1.05,
        2,
    )

    suggested_deadline = round(
        float(default_expected_time.quantile(0.70)) * 1.05,
        3,
    )

    if cargo_type == "Pharmaceuticals":
        reliable_index = routes.index[
            routes["route_profile"].eq("Reliable")
        ][0]

        suggested_budget = max(
            suggested_budget,
            round(
                float(
                    routes.loc[reliable_index, "total_cost"]
                ) * 1.05,
                2,
            ),
        )

        suggested_deadline = round(
            float(default_expected_time.loc[reliable_index]) * 1.10,
            3,
        )

    elif cargo_type == "Perishable Goods":
        express_index = routes.index[
            routes["route_profile"].eq("Express")
        ][0]

        suggested_deadline = round(
            float(default_expected_time.loc[express_index]) * 1.08,
            3,
        )

    budget = (
        suggested_budget
        if max_budget is None
        else _positive_number(max_budget, "max_budget")
    )

    deadline = (
        suggested_deadline
        if delivery_deadline is None
        else _positive_number(
            delivery_deadline,
            "delivery_deadline",
        )
    )

    if min_reliability is None:
        reliability_requirement = float(
            settings["min_reliability"]
        )
    else:
        if isinstance(min_reliability, bool):
            raise ValueError(
                "min_reliability must be between zero and one."
            )

        try:
            reliability_requirement = float(min_reliability)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "min_reliability must be numeric."
            ) from exc

        if (
            not math.isfinite(reliability_requirement)
            or not 0 <= reliability_requirement <= 1
        ):
            raise ValueError(
                "min_reliability must be between zero and one."
            )

    shipment_digest = _stable_digest(
        {
            "origin": origin,
            "destination": destination,
            "cargo_type": cargo_type,
            "cargo_value": cargo_value,
            "max_budget": budget,
            "delivery_deadline": deadline,
            "min_reliability": reliability_requirement,
            "scenario_version": 1,
        }
    )

    shipment_id = f"USR-{shipment_digest[:12].upper()}"

    shipment = {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "cargo_type": cargo_type,
        "cargo_value": cargo_value,
        "shipment_priority": settings["priority"],
        "max_budget": budget,
        "delivery_deadline": deadline,
        "min_reliability": reliability_requirement,
        "scenario": "User-defined synthetic corridor",
    }

    routes["shipment_id"] = shipment_id

    routes["route_id"] = [
        f"{shipment_id}-R{index + 1:02d}"
        for index in range(len(routes))
    ]

    clean_shipments, clean_routes = validate_dataset(
        pd.DataFrame([shipment]),
        routes,
        expected_routes=5,
    )

    return clean_shipments.iloc[0].to_dict(), clean_routes