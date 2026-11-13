from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.config import EngineConfig


SHIPMENT_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "cargo_type",
    "cargo_value",
    "shipment_priority",
    "max_budget",
    "delivery_deadline",
    "min_reliability",
]

ROUTE_COLUMNS = [
    "shipment_id",
    "route_id",
    "route_profile",
    "intermediate_hubs",
    "total_cost",
    "transit_time",
    "delay_probability",
    "congestion_risk",
    "geopolitical_risk",
    "weather_risk",
    "reliability",
    "number_of_stops",
]

PROBABILITY_COLUMNS = [
    "delay_probability",
    "congestion_risk",
    "geopolitical_risk",
    "weather_risk",
    "reliability",
]


def _require_columns(frame: pd.DataFrame, required: list[str], name: str):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _numeric(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{column} must be numeric.") from exc

        values = result[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} must contain finite values.")

    return result


def _validate_text(frame: pd.DataFrame, columns: list[str], name: str):
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{name}.{column} cannot contain nulls.")

        valid = frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )

        if not valid.all():
            raise ValueError(
                f"{name}.{column} must contain nonempty strings."
            )


def validate_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    if shipments.empty:
        raise ValueError("Shipment dataset is empty.")

    _require_columns(shipments, SHIPMENT_COLUMNS, "shipments")

    result = _numeric(
        shipments,
        [
            "cargo_value",
            "max_budget",
            "delivery_deadline",
            "min_reliability",
        ],
        "shipments",
    )

    _validate_text(
        result,
        [
            "shipment_id",
            "origin",
            "destination",
            "cargo_type",
            "shipment_priority",
        ],
        "shipments",
    )

    if result["shipment_id"].duplicated().any():
        raise ValueError("Shipment IDs must be unique.")

    if (result["origin"] == result["destination"]).any():
        raise ValueError("Shipment origin and destination must differ.")

    for column in ["max_budget", "delivery_deadline"]:
        if (result[column] <= 0).any():
            raise ValueError(f"{column} must be positive.")

    if (result["cargo_value"] < 0).any():
        raise ValueError("cargo_value cannot be negative.")

    if not result["min_reliability"].between(0, 1).all():
        raise ValueError("min_reliability must be between zero and one.")

    allowed_priorities = {"Low", "Medium", "High", "Critical"}

    if not result["shipment_priority"].isin(allowed_priorities).all():
        raise ValueError("Unsupported shipment priority.")

    return result.reset_index(drop=True)


def validate_routes(
    routes: pd.DataFrame,
    shipment: Mapping | pd.Series,
) -> pd.DataFrame:
    if routes.empty:
        raise ValueError("No route alternatives were supplied.")

    _require_columns(routes, ROUTE_COLUMNS, "routes")

    result = _numeric(
        routes,
        [
            "total_cost",
            "transit_time",
            "number_of_stops",
            *PROBABILITY_COLUMNS,
        ],
        "routes",
    )

    _validate_text(
        result,
        ["shipment_id", "route_id", "route_profile"],
        "routes",
    )

    if result["route_id"].duplicated().any():
        raise ValueError("Route IDs must be unique within a shipment.")

    if not result["shipment_id"].eq(shipment["shipment_id"]).all():
        raise ValueError(
            "All alternatives must belong to the supplied shipment."
        )

    for column in ["total_cost", "transit_time"]:
        if (result[column] <= 0).any():
            raise ValueError(f"{column} must be positive.")

    for column in PROBABILITY_COLUMNS:
        if not result[column].between(0, 1).all():
            raise ValueError(f"{column} must lie between zero and one.")

    stops = result["number_of_stops"]

    if (stops < 0).any() or not np.equal(stops, np.floor(stops)).all():
        raise ValueError("number_of_stops must be a nonnegative integer.")

    result["number_of_stops"] = stops.astype(int)

    # Empty strings are allowed for direct routes.
    result["intermediate_hubs"] = (
        result["intermediate_hubs"].fillna("").astype(str)
    )

    for _, row in result.iterrows():
        hubs = [
            hub.strip()
            for hub in row["intermediate_hubs"].split("|")
            if hub.strip()
        ]

        if len(hubs) != row["number_of_stops"]:
            raise ValueError("Hub count does not match number_of_stops.")

        if len(set(hubs)) != len(hubs):
            raise ValueError("A route cannot repeat an intermediate hub.")

        if shipment["origin"] in hubs or shipment["destination"] in hubs:
            raise ValueError(
                "Origin/destination cannot also be intermediate hubs."
            )

    return result.reset_index(drop=True)


def validate_dataset(
    shipments: pd.DataFrame,
    routes: pd.DataFrame,
    expected_routes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_shipments = validate_shipments(shipments)

    _require_columns(routes, ROUTE_COLUMNS, "routes")

    shipment_ids = set(clean_shipments["shipment_id"])
    route_shipment_ids = set(routes["shipment_id"])

    if shipment_ids != route_shipment_ids:
        raise ValueError(
            "Every shipment must have routes, and routes cannot reference "
            "unknown shipments."
        )

    clean_groups = []

    for _, shipment in clean_shipments.iterrows():
        group = routes.loc[
            routes["shipment_id"] == shipment["shipment_id"]
        ]

        if expected_routes is not None and len(group) != expected_routes:
            raise ValueError(
                f"{shipment['shipment_id']} must have "
                f"{expected_routes} routes."
            )

        clean_groups.append(validate_routes(group, shipment))

    return (
        clean_shipments,
        pd.concat(clean_groups, ignore_index=True),
    )


def normalize(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())

    if high == low:
        return pd.Series(0.0, index=series.index, dtype=float)

    return (series - low) / (high - low)


def calculate_features(
    routes: pd.DataFrame,
    config: EngineConfig,
) -> pd.DataFrame:
    result = routes.copy()

    result["expected_time"] = (
        result["transit_time"]
        + result["delay_probability"] * config.average_delay_days
    )

    result["composite_risk"] = (
        config.congestion_weight * result["congestion_risk"]
        + config.geopolitical_weight * result["geopolitical_risk"]
        + config.weather_weight * result["weather_risk"]
    )

    return result


def on_time_probability(
    routes: pd.DataFrame,
    deadline: float,
    average_delay_days: float,
) -> np.ndarray:
    """
    Model:
      delivery_time = base_time + Bernoulli(p_delay) * Exponential(mean_delay)

    This is a simulation-model estimate, not a calibrated forecast.
    """
    slack = deadline - routes["transit_time"].to_numpy(dtype=float)
    delay_probability = routes["delay_probability"].to_numpy(dtype=float)

    probability = 1.0 - delay_probability * np.exp(
        -np.maximum(slack, 0.0) / average_delay_days
    )

    return np.where(slack >= 0, probability, 0.0)