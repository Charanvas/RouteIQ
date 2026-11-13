import pandas as pd
import pytest


@pytest.fixture
def shipment():
    return {
        "shipment_id": "SHP_TEST",
        "origin": "Mumbai",
        "destination": "Rotterdam",
        "cargo_type": "Electronics",
        "cargo_value": 150_000.0,
        "shipment_priority": "High",
        "max_budget": 20_000.0,
        "delivery_deadline": 40.0,
        "min_reliability": 0.0,
    }


@pytest.fixture
def routes():
    """
    Known trade-off example:

        A: cost 8000,  time 29, risk 0.70
        B: cost 10000, time 23, risk 0.30
        C: cost 12000, time 21, risk 0.15

    Delay probability is zero so expected time equals base time.

    Each route's three risk indicators are identical, making its
    composite risk equal to the specified risk under any valid
    risk-component weights.
    """
    records = [
        {
            "shipment_id": "SHP_TEST",
            "route_id": "A",
            "route_profile": "Economy",
            "intermediate_hubs": "Dubai",
            "total_cost": 8_000.0,
            "transit_time": 29.0,
            "delay_probability": 0.0,
            "congestion_risk": 0.70,
            "geopolitical_risk": 0.70,
            "weather_risk": 0.70,
            "reliability": 0.85,
            "number_of_stops": 1,
        },
        {
            "shipment_id": "SHP_TEST",
            "route_id": "B",
            "route_profile": "Balanced",
            "intermediate_hubs": "Colombo",
            "total_cost": 10_000.0,
            "transit_time": 23.0,
            "delay_probability": 0.0,
            "congestion_risk": 0.30,
            "geopolitical_risk": 0.30,
            "weather_risk": 0.30,
            "reliability": 0.94,
            "number_of_stops": 1,
        },
        {
            "shipment_id": "SHP_TEST",
            "route_id": "C",
            "route_profile": "Reliable",
            "intermediate_hubs": "Singapore",
            "total_cost": 12_000.0,
            "transit_time": 21.0,
            "delay_probability": 0.0,
            "congestion_risk": 0.15,
            "geopolitical_risk": 0.15,
            "weather_risk": 0.15,
            "reliability": 0.98,
            "number_of_stops": 1,
        },
    ]

    return pd.DataFrame(records)