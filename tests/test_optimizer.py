import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig
from src.evaluation import evaluate_dataset
from src.explainability import (
    alternative_comparisons,
    explain_recommendation,
)
from src.optimizer import optimize_routes
from src.preprocessing import normalize, on_time_probability
from src.sensitivity import (
    analyze_profiles,
    explain_priority_change,
)


def test_balanced_profile_selects_middle_tradeoff(routes, shipment):
    result = optimize_routes(
        routes,
        shipment,
        profile="balanced",
    )

    assert result["recommended_route"]["route_id"] == "B"

    expected_score = (
        0.40 * 0.50
        + 0.30 * 0.25
        + 0.30 * ((0.30 - 0.15) / (0.70 - 0.15))
    )

    assert result["recommended_route"]["score"] == pytest.approx(
        expected_score
    )


@pytest.mark.parametrize(
    ("profile", "expected_route"),
    [
        ("cost", "A"),
        ("time", "C"),
        ("risk", "C"),
        ("balanced", "B"),
    ],
)
def test_profiles_choose_expected_routes(
    routes,
    shipment,
    profile,
    expected_route,
):
    result = optimize_routes(
        routes,
        shipment,
        profile=profile,
    )

    assert result["recommended_route"]["route_id"] == expected_route


def test_auto_profile_uses_cargo_default(routes, shipment):
    shipment["cargo_type"] = "Pharmaceuticals"

    result = optimize_routes(
        routes,
        shipment,
        profile="auto",
    )

    assert result["profile"] == "risk"
    assert result["recommended_route"]["route_id"] == "C"


def test_constant_criterion_normalizes_to_zero():
    result = normalize(pd.Series([10.0, 10.0, 10.0]))

    assert result.eq(0.0).all()


def test_normalization_preserves_index():
    values = pd.Series(
        [10.0, 20.0, 30.0],
        index=[4, 8, 12],
    )

    result = normalize(values)

    assert result.index.tolist() == [4, 8, 12]
    assert result.tolist() == [0.0, 0.5, 1.0]


def test_expected_transit_time_uses_configured_delay(routes, shipment):
    routes = routes.copy()
    routes["delay_probability"] = 0.30

    result = optimize_routes(
        routes,
        shipment,
        config=EngineConfig(average_delay_days=5.0),
    )

    table = result["ranking"].set_index("route_id")

    assert table.loc["A", "expected_time"] == pytest.approx(30.5)
    assert table.loc["B", "expected_time"] == pytest.approx(24.5)


def test_dominated_route_is_not_selected(routes, shipment):
    dominated = routes.iloc[2].to_dict()

    dominated.update(
        {
            "route_id": "D",
            "total_cost": 13_000.0,
            "transit_time": 28.0,
            "congestion_risk": 0.50,
            "geopolitical_risk": 0.50,
            "weather_risk": 0.50,
        }
    )

    alternatives = pd.concat(
        [routes, pd.DataFrame([dominated])],
        ignore_index=True,
    )

    result = optimize_routes(
        alternatives,
        shipment,
        profile="balanced",
    )

    table = result["ranking"].set_index("route_id")

    assert not bool(table.loc["D", "pareto_optimal"])
    assert not bool(table.loc["D", "eligible"])
    assert result["recommended_route"]["route_id"] != "D"


def test_identical_routes_use_deterministic_id_tie_break(
    routes,
    shipment,
):
    first = routes.iloc[0].to_dict()
    second = dict(first)

    first["route_id"] = "Z"
    second["route_id"] = "A"

    alternatives = pd.DataFrame([first, second])

    result = optimize_routes(
        alternatives,
        shipment,
        profile="balanced",
    )

    assert result["recommended_route"]["route_id"] == "A"
    assert result["ranking"]["pareto_optimal"].all()


@pytest.mark.parametrize(
    "weights",
    [
        {"cost": 0.5, "time": 0.5},
        {"cost": 0.5, "time": 0.5, "risk": 0.5},
        {"cost": -0.1, "time": 0.5, "risk": 0.6},
        {"cost": float("nan"), "time": 0.5, "risk": 0.5},
    ],
)
def test_invalid_weights_are_rejected(routes, shipment, weights):
    with pytest.raises(ValueError):
        optimize_routes(
            routes,
            shipment,
            weights=weights,
        )


def test_zero_weight_criteria_are_allowed(routes, shipment):
    result = optimize_routes(
        routes,
        shipment,
        weights={
            "cost": 1.0,
            "time": 0.0,
            "risk": 0.0,
        },
    )

    assert result["recommended_route"]["route_id"] == "A"


def test_sensitivity_detects_route_changes(routes, shipment):
    summary, scores = analyze_profiles(routes, shipment)

    assert len(summary) == 4
    assert len(scores) == 4 * len(routes)
    assert summary["route_id"].nunique() > 1

    balanced = optimize_routes(
        routes,
        shipment,
        profile="balanced",
    )

    time_focused = optimize_routes(
        routes,
        shipment,
        profile="time",
    )

    explanation = explain_priority_change(
        balanced,
        time_focused,
    )

    assert "changes from B to C" in explanation


def test_explanation_uses_actual_comparisons(routes, shipment):
    result = optimize_routes(
        routes,
        shipment,
        profile="balanced",
    )

    explanation = explain_recommendation(result)

    assert explanation["headline"] == "Recommend B."
    assert "Mumbai" in explanation["path"]
    assert "Rotterdam" in explanation["path"]

    comparisons = alternative_comparisons(result).set_index(
        "alternative"
    )

    assert comparisons.loc[
        "A", "selected_cost_change_pct"
    ] == pytest.approx(25.0)

    assert comparisons.loc[
        "A", "selected_time_saved_days"
    ] == pytest.approx(6.0)


def test_on_time_probability_matches_delay_model():
    frame = pd.DataFrame(
        {
            "transit_time": [20.0, 20.0, 25.0],
            "delay_probability": [0.20, 0.20, 0.20],
        }
    )

    probability = on_time_probability(
        frame,
        deadline=20.0,
        average_delay_days=5.0,
    )

    assert probability[0] == pytest.approx(0.80)
    assert probability[1] == pytest.approx(0.80)
    assert probability[2] == 0.0

    later_deadline = on_time_probability(
        frame.iloc[:1],
        deadline=25.0,
        average_delay_days=5.0,
    )

    assert later_deadline[0] == pytest.approx(
        1.0 - 0.20 * np.exp(-1.0)
    )


def test_evaluation_accounting(routes, shipment):
    impossible_shipment = dict(shipment)

    impossible_shipment.update(
        {
            "shipment_id": "SHP_IMPOSSIBLE",
            "max_budget": 1.0,
        }
    )

    impossible_routes = routes.copy()
    impossible_routes["shipment_id"] = "SHP_IMPOSSIBLE"
    impossible_routes["route_id"] = (
        "IMP_" + impossible_routes["route_id"]
    )

    all_shipments = pd.DataFrame(
        [shipment, impossible_shipment]
    )

    all_routes = pd.concat(
        [routes, impossible_routes],
        ignore_index=True,
    )

    evaluation = evaluate_dataset(
        all_shipments,
        all_routes,
        profile="balanced",
    )

    summary = evaluation["summary"]

    assert summary["shipments_evaluated"] == 2
    assert summary["shipments_with_feasible_routes"] == 1
    assert summary["no_feasible_shipments"] == 1

    assert summary["constraint_satisfaction_rate_pct"] == 100.0
    assert summary["pareto_efficiency_rate_pct"] == 100.0
    assert summary["feasible_recommendation_coverage_pct"] == 50.0