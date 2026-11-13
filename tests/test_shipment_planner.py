import pandas as pd
import pytest

from src.optimizer import optimize_routes
from src.shipment_planner import plan_shipment


def test_user_inputs_are_preserved():
    shipment, routes = plan_shipment(
        origin="Mumbai",
        destination="Rotterdam",
        cargo_type="Electronics",
        cargo_value=200_000,
    )

    assert shipment["origin"] == "Mumbai"
    assert shipment["destination"] == "Rotterdam"
    assert shipment["cargo_type"] == "Electronics"
    assert shipment["cargo_value"] == 200_000

    assert len(routes) == 5
    assert routes["shipment_id"].eq(
        shipment["shipment_id"]
    ).all()


def test_identical_requests_are_reproducible():
    first_shipment, first_routes = plan_shipment(
        "Mumbai",
        "Rotterdam",
        "Electronics",
        150_000,
    )

    second_shipment, second_routes = plan_shipment(
        "Mumbai",
        "Rotterdam",
        "Electronics",
        150_000,
    )

    assert first_shipment == second_shipment

    pd.testing.assert_frame_equal(
        first_routes,
        second_routes,
    )


def test_user_endpoints_are_not_intermediate_hubs():
    shipment, routes = plan_shipment(
        "Singapore",
        "Hamburg",
        "Machinery",
        100_000,
    )

    for hubs in routes["intermediate_hubs"]:
        nodes = hubs.split("|")

        assert shipment["origin"] not in nodes
        assert shipment["destination"] not in nodes


def test_cargo_value_changes_cost_not_route_risk():
    _, low_value = plan_shipment(
        "Mumbai",
        "Rotterdam",
        "Electronics",
        100_000,
    )

    _, high_value = plan_shipment(
        "Mumbai",
        "Rotterdam",
        "Electronics",
        200_000,
    )

    assert (
        high_value["total_cost"] > low_value["total_cost"]
    ).all()

    unchanged_columns = [
        "intermediate_hubs",
        "transit_time",
        "delay_probability",
        "congestion_risk",
        "geopolitical_risk",
        "weather_risk",
        "reliability",
    ]

    pd.testing.assert_frame_equal(
        low_value[unchanged_columns],
        high_value[unchanged_columns],
    )


def test_total_cost_components_add_up():
    _, routes = plan_shipment(
        "Dubai",
        "New York",
        "Pharmaceuticals",
        250_000,
    )

    expected = (
        routes["base_transport_cost"]
        + routes["cargo_handling_fee"]
        + routes["cargo_value_surcharge"]
    ).round(2)

    pd.testing.assert_series_equal(
        routes["total_cost"],
        expected,
        check_names=False,
    )


def test_custom_constraints_are_preserved():
    shipment, _ = plan_shipment(
        "Mumbai",
        "Rotterdam",
        "Electronics",
        150_000,
        max_budget=9_000,
        delivery_deadline=18,
        min_reliability=0.95,
    )

    assert shipment["max_budget"] == 9_000
    assert shipment["delivery_deadline"] == 18
    assert shipment["min_reliability"] == 0.95


def test_impossible_budget_is_not_silently_relaxed():
    shipment, routes = plan_shipment(
        "Mumbai",
        "Rotterdam",
        "Electronics",
        150_000,
        max_budget=1,
    )

    result = optimize_routes(routes, shipment)

    assert result["no_feasible_route"]
    assert result["recommended_route"] is None
    assert result["fallback_candidate"] is not None


@pytest.mark.parametrize(
    ("origin", "destination", "cargo", "value"),
    [
        ("Mumbai", "Mumbai", "Electronics", 100_000),
        ("Unknown", "Rotterdam", "Electronics", 100_000),
        ("Mumbai", "Rotterdam", "Unknown Cargo", 100_000),
        ("Mumbai", "Rotterdam", "Electronics", -1),
        ("Mumbai", "Rotterdam", "Electronics", float("nan")),
    ],
)
def test_invalid_requests_are_rejected(
    origin,
    destination,
    cargo,
    value,
):
    with pytest.raises(ValueError):
        plan_shipment(
            origin,
            destination,
            cargo,
            value,
        )