# RouteIQ

Explainable multi-objective shipment-route selection under cost, time,
and risk constraints.

RouteIQ is a synthetic decision-support prototype for the Aiqon
AI Internship take-home assignment.

It uses a mathematical decision engine, not an LLM, to select routes.

## Features

- Reproducible synthetic dataset generation.
- 50 shipments and five alternatives per shipment by default.
- 14 named logistics nodes.
- Deliberate Economy, Express, Balanced, High Risk, and Reliable profiles.
- Cargo-aware default decision profiles.
- Budget, expected-time deadline, and minimum-reliability constraints.
- Risk-adjusted transit times.
- Configurable risk-component weights.
- Within-shipment min-max normalization.
- Feasible Pareto-frontier selection.
- Weighted multi-objective ranking.
- Deterministic tie-breaking.
- Explicit no-feasible-route handling.
- Data-grounded explanations and alternative comparisons.
- Interactive sensitivity analysis.
- Plotly visualizations.
- Baseline benchmarking.
- CSV and JSON exports.
- Automated PDF technical report generation.
- Pytest test suite.

## Important limitation

This is not a real maritime routing system.

Named locations and intermediate hubs are synthetic route nodes.
Paths are not checked against geography, shipping lanes, vessel
schedules, customs requirements, capacity, or live prices.

All risks, costs, and reliability values are simulated.

The application demonstrates a decision framework, not validated
operational forecasting accuracy.

## Requirements

Use Python 3.11 or newer.

The project has no API-key requirement.

## Installation

Create and activate a virtual environment.

Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

macOS/Linux:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Optional: capture exact installed versions:

    pip freeze > requirements-lock.txt

## Quick start

Run these commands from the repository root:

    python -m src.data_generator --shipments 50 --seed 42
    pytest
    python -m src.evaluation
    python -m src.reporting
    streamlit run app.py

Generated files:

    data/shipments.csv
    data/routes.csv

    reports/evaluation_summary.json
    reports/evaluation_decisions.csv
    reports/baseline_summary.csv
    reports/technical_report.pdf

The dashboard can generate an in-memory default dataset if neither
dataset CSV exists.

If only one CSV exists, regenerate the dataset so the files remain
consistent.

## Repository structure

    RouteIQ/
    ├── app.py
    ├── requirements.txt
    ├── pytest.ini
    ├── .gitignore
    ├── README.md
    ├── data/
    ├── reports/
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_generator.py
    │   ├── preprocessing.py
    │   ├── pareto.py
    │   ├── optimizer.py
    │   ├── explainability.py
    │   ├── sensitivity.py
    │   ├── evaluation.py
    │   └── reporting.py
    └── tests/
        ├── conftest.py
        ├── test_data_generator.py
        ├── test_optimizer.py
        └── test_constraints.py

## Decision framework

### 1. Validate inputs

Validation checks:

- Required columns.
- Unique shipment IDs.
- Unique route IDs within a shipment.
- Valid shipment references.
- Finite numeric values.
- Positive costs and transit times.
- Risk and probability values between zero and one.
- Consistent intermediate-hub counts.
- No repeated intermediate hubs.
- No origin or destination repeated as an intermediate hub.

### 2. Calculate expected transit time

    expected_time = transit_time + delay_probability * average_delay_days

The default average delay duration is five days.

It is a configurable simulation assumption.

### 3. Calculate composite risk

Default coefficients:

    composite_risk =
        0.35 * congestion_risk
        + 0.40 * geopolitical_risk
        + 0.25 * weather_risk

These are configurable design assumptions, not universal logistics
coefficients.

### 4. Apply hard constraints

A route is feasible only if:

    total_cost <= max_budget

    expected_time <= delivery_deadline

    reliability >= min_reliability

The deadline constraint applies to expected time.
It is not a guarantee of on-time delivery.

### 5. Identify the feasible Pareto frontier

All three objectives are minimized:

- Total cost.
- Expected transit time.
- Composite risk.

Route A dominates Route B if A is no worse on every objective and is
strictly better on at least one.

When feasible alternatives exist, Pareto membership is calculated only
within the feasible alternative pool.

### 6. Normalize objectives

Each criterion is min-max normalized within the shipment's complete
alternative set:

    normalized = (value - minimum) / (maximum - minimum)

If all values are identical, the normalized criterion is zero.

The normalization reference set is fixed for a given shipment during
weight-only sensitivity analysis.

Adding or removing alternatives can change normalized scores.

### 7. Calculate the weighted score

    score =
        cost_weight * normalized_cost
        + time_weight * normalized_expected_time
        + risk_weight * normalized_risk

Lower scores are better.

The final recommendation is selected from the feasible Pareto frontier.

Exact score ties are resolved by route ID.

## Standard business profiles

| Profile | Cost | Time | Risk |
|---|---:|---:|---:|
| Cost | 0.60 | 0.20 | 0.20 |
| Time | 0.20 | 0.60 | 0.20 |
| Risk | 0.20 | 0.20 | 0.60 |
| Balanced | 0.40 | 0.30 | 0.30 |

Automatic cargo defaults:

| Cargo | Default profile |
|---|---|
| Electronics | Balanced |
| Pharmaceuticals | Risk |
| Textiles | Cost |
| Machinery | Cost |
| Perishable Goods | Time |

Operators can override these defaults.

## No-feasible-route behavior

If no alternative satisfies all hard constraints:

    result["recommended_route"] is None

A separate diagnostic candidate is returned:

    result["fallback_candidate"]

Its ranking measure is:

    relative budget overrun
    + relative expected-time overrun
    + relative reliability shortfall

Weighted score and route ID break ties.

This diagnostic measure gives each relative violation equal weight.
It is an explicit heuristic, not a business-approved constraint-relaxation
policy.

No constraint is relaxed automatically.

## Uncertainty model

A supplementary model-based probability of meeting the deadline uses:

    delivery_time =
        base_time
        + Bernoulli(delay_probability)
          * Exponential(mean_delay_days)

For nonnegative deadline slack:

    P(on time) =
        1 - delay_probability * exp(-slack / mean_delay_days)

If the deadline is earlier than base transit time, the probability is zero.

This estimate is:

- A simulation-model output.
- Not a calibrated forecast.
- Not an extra weighted optimization criterion.
- Not the same quantity as the historical reliability indicator.

## Reliability and correlated criteria

Reliability is used as:

- A hard minimum requirement.
- An explanatory indicator.

It is not added as a fourth weighted criterion.

Composite risk and expected transit time are intentionally correlated:
risk influences simulated delay probability, while composite risk also
represents exposure beyond delay.

This is a modeling choice and should be reviewed before operational use.

## Dashboard

### Recommendation

- Shipment context.
- Selected route or explicit infeasibility warning.
- Cost, expected time, delay probability, and risk.
- Constraint satisfaction.
- Actual alternative comparisons.
- Rejection reasons.
- Full route ranking.

### Visual analytics

- Cost versus expected time.
- Pareto membership.
- Risk indicators.
- Weighted-score contributions.
- Model-based on-time probability.

### What-if analysis

- Standard profile comparison.
- Route-score heatmap.
- Custom business-priority sliders.
- Explanation of recommendation changes.
- Sidebar overrides for constraints and model assumptions.

### Evaluation

- Constraint satisfaction.
- Feasible recommendation coverage.
- Pareto efficiency.
- Profile sensitivity.
- Baseline comparisons.
- Per-shipment benchmark decisions.

### Data and exports

- Shipment and route inputs.
- Route ranking CSV.
- Decision JSON.
- Complete dataset CSVs.

## Evaluation methodology

### Constraint satisfaction

Denominator:

    shipments with at least one feasible alternative

This checks that the optimizer never recommends an infeasible route when
a feasible route exists.

### Recommendation coverage

Denominator:

    all shipments

This reports how often a valid recommendation can actually be made.

### Pareto efficiency

Denominator:

    shipments with at least one feasible alternative

This checks whether recommended routes belong to the feasible Pareto set.

### Baselines

- Cheapest feasible route.
- Fastest feasible route by base transit time.
- Lowest-risk feasible route.
- RouteIQ.

All methods use the same feasible pool.

No-feasible shipments are reported separately and excluded from average
cost, time, risk, and score comparisons.

### Sensitivity

The evaluation counts feasible shipments whose selected route changes
across the four standard business profiles.

A change is not automatically a failure: it may be the correct response
to different priorities.

### Interpreting scores

Scores are normalized within each shipment and depend on the selected
business profile.

Average scores are descriptive internal summaries, not monetary savings,
real-world utility, or evidence of deployment accuracy.

## Generate evaluation artifacts

Default automatic cargo profiles:

    python -m src.evaluation

Specific profile:

    python -m src.evaluation --profile balanced

## Generate the technical report

    python -m src.reporting

This regenerates evaluation results from the current dataset using
automatic cargo profiles and the default engine configuration, then
writes the PDF.

Dashboard overrides are not written into the generated report.

## Tests

Run:

    pytest

The test suite covers:

- Generator dimensions and reproducibility.
- Hub and probability validation.
- Deliberate route-profile relationships.
- Known weighted-score outcomes.
- Automatic cargo profiles.
- Normalization edge cases.
- Expected-time calculations.
- Pareto dominance.
- Deterministic ties.
- Invalid weights and route values.
- Budget, deadline, and reliability constraints.
- No-feasible-route behavior.
- Sensitivity analysis.
- Explanation comparisons.
- Evaluation denominator accounting.


## Limitations

- Synthetic data, not operational observations.
- No geographical route-feasibility checks.
- No vessel schedules or actual leg-level network optimization.
- No capacity, customs, or transshipment compatibility constraints.
- No network-wide allocation across multiple shipments.
- Simulated risk and reliability.
- Assumed delay distribution.
- Expected-time constraints are not chance constraints.
- Correlated criteria.
- Alternative-set sensitivity under min-max normalization.
- Weighted sums do not necessarily recover every Pareto-optimal point.
- No claim of real-world predictive accuracy.

## Future improvements

- Actual route legs and network connectivity.
- Calibrated cost, delay, and risk models.
- Chance constraints for service-level requirements.
- Robust optimization under parameter uncertainty.
- Capacity-aware allocation across shipments.
- Operational-data validation.
- User authentication and decision audit persistence.
- A separately deployed API if required.