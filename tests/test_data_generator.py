import pandas as pd
import pytest

from src.data_generator import generate_dataset, save_dataset
from src.preprocessing import validate_dataset


def test_default_dataset_dimensions():
    shipments, routes = generate_dataset()

    assert len(shipments) == 50
    assert len(routes) == 250

    route_counts = routes.groupby("shipment_id").size()

    assert route_counts.eq(5).all()


def test_seed_reproducibility():
    first_shipments, first_routes = generate_dataset(seed=123)
    second_shipments, second_routes = generate_dataset(seed=123)

    pd.testing.assert_frame_equal(
        first_shipments,
        second_shipments,
    )
    pd.testing.assert_frame_equal(
        first_routes,
        second_routes,
    )


def test_different_seeds_change_data():
    first_shipments, _ = generate_dataset(seed=123)
    second_shipments, _ = generate_dataset(seed=456)

    assert not first_shipments.equals(second_shipments)


def test_hubs_exclude_endpoints():
    shipments, routes = generate_dataset()

    shipment_lookup = shipments.set_index("shipment_id")

    for _, route in routes.iterrows():
        shipment = shipment_lookup.loc[route["shipment_id"]]

        hubs = [
            hub
            for hub in route["intermediate_hubs"].split("|")
            if hub
        ]

        assert shipment["origin"] not in hubs
        assert shipment["destination"] not in hubs
        assert len(hubs) == route["number_of_stops"]
        assert len(hubs) == len(set(hubs))


def test_generated_probability_ranges():
    _, routes = generate_dataset()

    for column in [
        "delay_probability",
        "congestion_risk",
        "geopolitical_risk",
        "weather_risk",
        "reliability",
    ]:
        assert routes[column].between(0, 1).all()


def test_risk_profiles_have_deliberate_relationships():
    _, routes = generate_dataset()

    grouped = routes.groupby("route_profile").agg(
        mean_cost=("total_cost", "mean"),
        mean_time=("transit_time", "mean"),
        mean_delay=("delay_probability", "mean"),
        mean_reliability=("reliability", "mean"),
    )

    assert (
        grouped.loc["Express", "mean_time"]
        < grouped.loc["Economy", "mean_time"]
    )

    assert (
        grouped.loc["Reliable", "mean_reliability"]
        > grouped.loc["High Risk", "mean_reliability"]
    )

    assert (
        grouped.loc["High Risk", "mean_delay"]
        > grouped.loc["Reliable", "mean_delay"]
    )

    assert (
        grouped.loc["Economy", "mean_cost"]
        < grouped.loc["Express", "mean_cost"]
    )


def test_cargo_specific_requirements():
    shipments, _ = generate_dataset()

    pharmaceutical = shipments.loc[
        shipments["cargo_type"] == "Pharmaceuticals"
    ]

    assert pharmaceutical["min_reliability"].eq(0.94).all()
    assert pharmaceutical["shipment_priority"].eq("Critical").all()


def test_save_dataset_creates_directory(tmp_path):
    shipments, routes = generate_dataset(n_shipments=5)

    directory = tmp_path / "nested" / "data"

    save_dataset(shipments, routes, directory)

    assert (directory / "shipments.csv").exists()
    assert (directory / "routes.csv").exists()

    loaded_shipments = pd.read_csv(directory / "shipments.csv")
    loaded_routes = pd.read_csv(directory / "routes.csv")

    validate_dataset(
        loaded_shipments,
        loaded_routes,
        expected_routes=5,
    )


def test_invalid_shipment_count():
    with pytest.raises(ValueError):
        generate_dataset(n_shipments=0)


def test_unsupported_route_count():
    with pytest.raises(ValueError):
        generate_dataset(routes_per_shipment=4)


def test_missing_route_group_is_rejected():
    shipments, routes = generate_dataset(n_shipments=5)

    routes = routes.loc[
        routes["shipment_id"] != shipments.iloc[0]["shipment_id"]
    ]

    with pytest.raises(ValueError):
        validate_dataset(shipments, routes)