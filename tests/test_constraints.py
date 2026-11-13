import pandas as pd
import pytest

from src.config import EngineConfig
from src.data_generator import generate_dataset
from src.optimizer import optimize_routes


def test_budget_constraint_removes_expensive_options(
    routes,
    shipment,
):
    shipment["max_budget"] = 9_000.0

    result = optimize_routes(
        routes,
        shipment,
        profile="time",
    )

    assert result["recommended_route"]["route_id"] == "A"
    assert result["recommended_route"]["total_cost"] <= 9_000.0


def test_deadline_constraint_removes_slow_options(
    routes,
    shipment,
):
    shipment["delivery_deadline"] = 22.0

    result = optimize_routes(
        routes,
        shipment,
        profile="cost",
    )

    assert result["recommended_route"]["route_id"] == "C"
    assert result["recommended_route"]["expected_time"] <= 22.0


def test_reliability_is_a_hard_constraint(
    routes,
    shipment,
):
    shipment["min_reliability"] = 0.96

    result = optimize_routes(
        routes,
        shipment,
        profile="cost",
    )

    assert result["recommended_route"]["route_id"] == "C"


def test_no_feasible_route_returns_no_recommendation(
    routes,
    shipment,
):
    shipment["max_budget"] = 9_000.0
    shipment["delivery_deadline"] = 25.0

    result = optimize_routes(
        routes,
        shipment,
        profile="balanced",
    )

    assert result["no_feasible_route"] is True
    assert result["status"] == "NO_FEASIBLE_ROUTE"
    assert result["recommended_route"] is None
    assert result["fallback_candidate"] is not None

    assert not result["ranking"]["feasible"].any()


def test_fallback_minimizes_reported_violation(
    routes,
    shipment,
):
    shipment["max_budget"] = 1.0

    result = optimize_routes(
        routes,
        shipment,
        profile="risk",
    )

    candidate = result["fallback_candidate"]

    assert candidate["violation_total"] == pytest.approx(
        result["ranking"]["violation_total"].min()
    )


def test_infeasible_route_cannot_dominate_feasible_selection(
    routes,
    shipment,
):
    # A very cheap but insufficiently reliable route must not influence
    # feasible-frontier eligibility through dominance.
    invalid_route = routes.iloc[2].to_dict()

    invalid_route.update(
        {
            "route_id": "INVALID",
            "total_cost": 1_000.0,
            "transit_time": 10.0,
            "congestion_risk": 0.01,
            "geopolitical_risk": 0.01,
            "weather_risk": 0.01,
            "reliability": 0.10,
        }
    )

    shipment["min_reliability"] = 0.80

    alternatives = pd.concat(
        [routes, pd.DataFrame([invalid_route])],
        ignore_index=True,
    )

    result = optimize_routes(
        alternatives,
        shipment,
        profile="balanced",
    )

    assert result["recommended_route"]["route_id"] != "INVALID"
    assert result["recommended_route"]["feasible"]
    assert result["recommended_route"]["pareto_optimal"]


def test_generated_dataset_preserves_selection_invariants():
    shipments, routes = generate_dataset()

    no_feasible_count = 0

    for _, shipment in shipments.iterrows():
        alternatives = routes.loc[
            routes["shipment_id"] == shipment["shipment_id"]
        ]

        result = optimize_routes(
            alternatives,
            shipment,
        )

        if result["no_feasible_route"]:
            no_feasible_count += 1

            assert result["recommended_route"] is None
            assert not result["ranking"]["feasible"].any()

        else:
            selected = result["recommended_route"]

            assert selected["feasible"]
            assert selected["pareto_optimal"]
            assert selected["eligible"]

            assert selected["total_cost"] <= shipment["max_budget"]
            assert (
                selected["expected_time"]
                <= shipment["delivery_deadline"]
            )
            assert (
                selected["reliability"]
                >= shipment["min_reliability"]
            )

    # The generator deliberately creates impossible cases.
    assert no_feasible_count > 0


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("total_cost", -1.0),
        ("transit_time", 0.0),
        ("delay_probability", 1.1),
        ("congestion_risk", -0.1),
        ("reliability", float("nan")),
        ("weather_risk", float("inf")),
        ("number_of_stops", 1.5),
    ],
)
def test_invalid_route_values_are_rejected(
    routes,
    shipment,
    column,
    invalid_value,
):
    routes = routes.copy()
    routes.loc[0, column] = invalid_value

    with pytest.raises(ValueError):
        optimize_routes(routes, shipment)


def test_missing_route_column_is_rejected(routes, shipment):
    routes = routes.drop(columns=["total_cost"])

    with pytest.raises(ValueError):
        optimize_routes(routes, shipment)


def test_duplicate_route_ids_are_rejected(routes, shipment):
    routes = routes.copy()
    routes.loc[1, "route_id"] = routes.loc[0, "route_id"]

    with pytest.raises(ValueError):
        optimize_routes(routes, shipment)


def test_mixed_shipment_routes_are_rejected(routes, shipment):
    routes = routes.copy()
    routes.loc[0, "shipment_id"] = "OTHER_SHIPMENT"

    with pytest.raises(ValueError):
        optimize_routes(routes, shipment)


def test_origin_as_intermediate_hub_is_rejected(
    routes,
    shipment,
):
    routes = routes.copy()
    routes.loc[0, "intermediate_hubs"] = shipment["origin"]

    with pytest.raises(ValueError):
        optimize_routes(routes, shipment)


def test_empty_alternatives_are_rejected(routes, shipment):
    with pytest.raises(ValueError):
        optimize_routes(
            routes.iloc[:0],
            shipment,
        )


def test_invalid_engine_configuration_is_rejected():
    with pytest.raises(ValueError):
        EngineConfig(average_delay_days=0.0)

    with pytest.raises(ValueError):
        EngineConfig(
            congestion_weight=0.5,
            geopolitical_weight=0.5,
            weather_weight=0.5,
        )