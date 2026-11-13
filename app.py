"""RouteIQ — dark enterprise dashboard.

Required existing modules:
- src.config
- src.data_generator
- src.preprocessing
- src.shipment_planner
- src.optimizer
- src.explainability (latest version with route_assessments)
- src.sensitivity
- src.evaluation

Run:
    streamlit run app.py
"""

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_DIR, EngineConfig, WEIGHT_PROFILES
from src.data_generator import (
    CARGO_SETTINGS,
    PORTS,
    generate_dataset,
)
from src.evaluation import evaluate_dataset
from src.explainability import (
    alternative_comparisons,
    explain_recommendation,
)
from src.optimizer import optimize_routes
from src.preprocessing import validate_dataset
from src.shipment_planner import plan_shipment
from src.sensitivity import (
    analyze_profiles,
    explain_priority_change,
)


# ------------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="RouteIQ | Decision Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

NAVIGATION = [
    "Overview",
    "Route intelligence",
    "Decision audit",
    "Scenario lab",
    "Performance",
    "Export center",
]

PAGE_COPY = {
    "Overview": (
        "Shipment command center",
        "A clear view of your shipment, recommended route, "
        "and the trade-offs behind the decision.",
    ),
    "Route intelligence": (
        "Explore the trade-offs",
        "Compare cost, expected delivery, operational exposure, "
        "and the feasible Pareto frontier.",
    ),
    "Decision audit": (
        "Every decision, explained",
        "Inspect requirements, rejected alternatives, and the exact "
        "criteria behind the recommendation.",
    ),
    "Scenario lab": (
        "Test your business priorities",
        "Explore how changing cost, time, and risk preferences "
        "changes the selected route.",
    ),
    "Performance": (
        "Validate the framework",
        "Benchmark decision consistency against simpler methods "
        "on the synthetic test dataset.",
    ),
    "Export center": (
        "Take the analysis with you",
        "Download shipment inputs, route rankings, decision evidence, "
        "and benchmark datasets.",
    ),
}

COLORS = [
    "#2dd4bf",
    "#60a5fa",
    "#a78bfa",
    "#fbbf24",
    "#fb7185",
]


# ------------------------------------------------------------------
# Shared UI utilities
# ------------------------------------------------------------------

def html(content: str):
    st.markdown(content, unsafe_allow_html=True)


def safe(value) -> str:
    return escape(str(value), quote=True)


def inject_styles():
    path = ROOT / "assets" / "styles.css"

    if path.exists():
        html(f"<style>{path.read_text(encoding='utf-8')}</style>")
    else:
        st.warning(
            "assets/styles.css was not found. "
            "Add the stylesheet to enable the full visual design."
        )


def section(title: str, subtitle: str = ""):
    html(
        '<div class="rq-section">'
        f'<div class="rq-section-title">{safe(title)}</div>'
        f'<div class="rq-section-subtitle">{safe(subtitle)}</div>'
        "</div>"
    )


def pill(text: str, tone: str = "") -> str:
    return (
        f'<span class="rq-pill {safe(tone)}">'
        f"{safe(text)}</span>"
    )


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(
        frame.to_json(
            orient="records",
            double_precision=12,
        )
    )


def selected_route(result: dict) -> dict:
    return (
        result["recommended_route"]
        or result["fallback_candidate"]
    )


def percent(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.1f}%"


def draw_chart(figure, height=390):
    figure.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#101a2b",
        plot_bgcolor="#101a2b",
        font={
            "family": "Arial, sans-serif",
            "color": "#b9c8df",
            "size": 12,
        },
        margin={
            "l": 18,
            "r": 18,
            "t": 35,
            "b": 25,
        },
        legend={
            "title_text": "",
            "orientation": "h",
            "y": -0.22,
            "x": 0,
        },
        hoverlabel={
            "bgcolor": "#18273d",
            "font_color": "#eef4ff",
        },
        colorway=COLORS,
    )

    figure.update_xaxes(
        gridcolor="#203047",
        zerolinecolor="#203047",
    )

    figure.update_yaxes(
        gridcolor="#203047",
        zerolinecolor="#203047",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        theme=None,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


def metric_row(route: dict):
    columns = st.columns(5)

    values = [
        ("Shipping cost", f"${route['total_cost']:,.0f}"),
        ("Expected delivery", f"{route['expected_time']:.2f} days"),
        ("Composite risk", f"{route['composite_risk']:.3f}"),
        ("Reliability", f"{route['reliability']:.1%}"),
        ("Delay probability", f"{route['delay_probability']:.1%}"),
    ]

    for column, (label, value) in zip(columns, values):
        column.metric(label, value)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_test_data():
    shipment_path = DATA_DIR / "shipments.csv"
    route_path = DATA_DIR / "routes.csv"

    if shipment_path.exists() != route_path.exists():
        raise ValueError(
            "Both shipments.csv and routes.csv must exist together. "
            "Regenerate the dataset before continuing."
        )

    if shipment_path.exists():
        return validate_dataset(
            pd.read_csv(shipment_path),
            pd.read_csv(route_path),
        )

    return generate_dataset(
        n_shipments=50,
        seed=42,
    )


@st.cache_data(show_spinner=False)
def benchmark(shipments, routes, profile, config):
    return evaluate_dataset(
        shipments,
        routes,
        profile=profile,
        config=config,
    )


# ------------------------------------------------------------------
# Sidebar and input workflow
# ------------------------------------------------------------------

def sidebar(shipments, test_routes):
    with st.sidebar:
        html(
            '<div class="rq-brand">'
            '<div class="rq-logo">RQ</div>'
            "<div>"
            '<div class="rq-brand-name">RouteIQ</div>'
            '<div class="rq-brand-subtitle">Decision workspace</div>'
            "</div>"
            "</div>"
        )

        page = st.radio(
            "Workspace",
            NAVIGATION,
            label_visibility="collapsed",
            key="workspace_navigation",
        )

        st.divider()

        mode = st.selectbox(
            "Shipment source",
            [
                "New shipment",
                "Test dataset",
            ],
            key="shipment_source",
        )

        original = None
        active_routes = None

        if mode == "New shipment":
            with st.form("new_shipment_form"):
                st.markdown("**Plan a shipment**")

                origin = st.selectbox(
                    "Origin",
                    PORTS,
                    index=PORTS.index("Mumbai"),
                )

                destination = st.selectbox(
                    "Destination",
                    PORTS,
                    index=PORTS.index("Rotterdam"),
                )

                cargo_type = st.selectbox(
                    "Cargo type",
                    list(CARGO_SETTINGS),
                )

                cargo_value = st.number_input(
                    "Cargo value · USD",
                    min_value=1.0,
                    value=150_000.0,
                    step=1_000.0,
                )

                submitted = st.form_submit_button(
                    "Generate route options",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                try:
                    planned_shipment, planned_routes = plan_shipment(
                        origin=origin,
                        destination=destination,
                        cargo_type=cargo_type,
                        cargo_value=cargo_value,
                    )

                    st.session_state["active_user_plan"] = {
                        "shipment": planned_shipment,
                        "routes": planned_routes,
                    }

                    st.session_state["plan_revision"] = (
                        st.session_state.get("plan_revision", 0) + 1
                    )

                    st.success("Shipment plan updated.")

                except ValueError as exc:
                    st.error(str(exc))

            active = st.session_state.get("active_user_plan")

            if active is not None:
                original = dict(active["shipment"])
                active_routes = active["routes"].copy()

                st.caption(
                    "The workspace displays your last generated plan. "
                    "Submit the form to apply changed shipment details."
                )

        else:
            shipment_id = st.selectbox(
                "Test shipment",
                shipments["shipment_id"].tolist(),
                key="selected_test_shipment",
            )

            original = shipments.loc[
                shipments["shipment_id"] == shipment_id
            ].iloc[0].to_dict()

            active_routes = test_routes.loc[
                test_routes["shipment_id"] == shipment_id
            ].copy()

        profile = "auto"
        config = EngineConfig()
        shipment = None

        if original is not None:
            st.divider()

            profile = st.selectbox(
                "Decision profile",
                ["auto", "balanced", "cost", "time", "risk"],
                format_func=lambda value: {
                    "auto": "Auto · cargo-aware",
                    "balanced": "Balanced",
                    "cost": "Cost optimized",
                    "time": "Time optimized",
                    "risk": "Risk optimized",
                }[value],
                key="decision_profile",
            )

            revision = (
                st.session_state.get("plan_revision", 0)
                if mode == "New shipment"
                else 0
            )

            scope = (
                f"{mode}_{original['shipment_id']}_{revision}"
            )

            with st.expander(
                "Business requirements",
                expanded=False,
            ):
                budget = st.number_input(
                    "Maximum budget · USD",
                    min_value=1.0,
                    value=float(original["max_budget"]),
                    step=100.0,
                    key=f"budget_{scope}",
                )

                deadline = st.number_input(
                    "Expected-time deadline · days",
                    min_value=0.1,
                    value=float(original["delivery_deadline"]),
                    step=0.5,
                    key=f"deadline_{scope}",
                )

                reliability = st.slider(
                    "Minimum reliability",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(original["min_reliability"]),
                    step=0.01,
                    key=f"reliability_{scope}",
                )

                st.caption(
                    "New shipment defaults are suggested simulation "
                    "requirements—not actual customer commitments."
                )

            with st.expander("Model assumptions"):
                delay = st.number_input(
                    "Mean delay duration · days",
                    min_value=0.1,
                    value=5.0,
                    step=0.5,
                    key="model_delay",
                )

                congestion = st.slider(
                    "Congestion importance",
                    0,
                    100,
                    35,
                    key="model_congestion",
                )

                geopolitical = st.slider(
                    "Geopolitical importance",
                    0,
                    100,
                    40,
                    key="model_geopolitical",
                )

                weather = st.slider(
                    "Weather importance",
                    0,
                    100,
                    25,
                    key="model_weather",
                )

                total = congestion + geopolitical + weather

                if total == 0:
                    st.error(
                        "At least one risk importance must be positive."
                    )
                    st.stop()

                config = EngineConfig(
                    average_delay_days=delay,
                    congestion_weight=congestion / total,
                    geopolitical_weight=geopolitical / total,
                    weather_weight=weather / total,
                )

            shipment = dict(original)

            shipment.update(
                {
                    "max_budget": budget,
                    "delivery_deadline": deadline,
                    "min_reliability": reliability,
                }
            )

        st.divider()

        html(
            '<div class="rq-small">'
            "SYNTHETIC ANALYSIS<br>"
            "No live rates · No verified shipping schedules<br>"
            "Mathematical selection · No LLM routing"
            "</div>"
        )

        with st.expander("Data controls"):
            if st.button(
                "Reload benchmark data",
                use_container_width=True,
            ):
                load_test_data.clear()
                benchmark.clear()
                st.rerun()

    return (
        page,
        mode,
        shipment,
        active_routes,
        profile,
        config,
    )


# ------------------------------------------------------------------
# Header and empty state
# ------------------------------------------------------------------

def page_header(page, shipment=None):
    title, subtitle = PAGE_COPY[page]

    context = (
        f"{shipment['origin']} → {shipment['destination']}"
        if shipment is not None
        else "New shipment workspace"
    )

    html(
        '<div class="rq-topline">'
        '<div class="rq-eyebrow">'
        f"ROUTEIQ / {safe(page)}"
        "</div>"
        '<div class="rq-pills">'
        f"{pill(context)}"
        f"{pill('Synthetic model', 'amber')}"
        "</div>"
        "</div>"
        f'<div class="rq-page-title">{safe(title)}</div>'
        f'<div class="rq-page-subtitle">{safe(subtitle)}</div>'
    )


def empty_state():
    html(
        '<div class="rq-empty">'
        f"{pill('READY TO PLAN', 'teal')}"
        '<div class="rq-empty-title">'
        "Better decisions start with the right context."
        "</div>"
        '<div class="rq-empty-copy">'
        "Choose your origin, destination, cargo type, and cargo value "
        "in the sidebar. Generate a plan to compare five synthetic "
        "routing alternatives, inspect constraints, and understand "
        "the trade-offs behind the recommendation."
        "</div>"
        "</div>"
    )

    section(
        "One workflow. Three decision layers.",
        "A recommendation should be acceptable, competitive, "
        "and explainable.",
    )

    columns = st.columns(3)

    content = [
        (
            "01 · Requirements",
            "Budget, expected delivery deadline, and minimum "
            "reliability define which routes are acceptable.",
        ),
        (
            "02 · Trade-offs",
            "Pareto analysis and configurable weights compare "
            "cost, expected time, and operational exposure.",
        ),
        (
            "03 · Evidence",
            "Detailed explanations show why each option was "
            "selected, rejected, or less preferred.",
        ),
    ]

    for column, (title, body) in zip(columns, content):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(body)


# ------------------------------------------------------------------
# Route presentation
# ------------------------------------------------------------------

def route_status(route, result):
    chosen = selected_route(result)

    if route["route_id"] == chosen["route_id"]:
        if result["no_feasible_route"]:
            return "Diagnostic candidate", "amber"

        return "Recommended", "teal"

    if not bool(route["feasible"]):
        return "Constraint violation", "red"

    if not bool(route["pareto_optimal"]):
        return "Pareto dominated", ""

    return "Feasible alternative", ""


def itinerary(route, shipment):
    hubs = [
        node.strip()
        for node in str(route["intermediate_hubs"]).split("|")
        if node.strip()
    ]

    nodes = [
        shipment["origin"],
        *hubs,
        shipment["destination"],
    ]

    parts = []

    for index, node in enumerate(nodes):
        if index:
            parts.append('<span class="rq-arrow">→</span>')

        endpoint = (
            " endpoint"
            if index in (0, len(nodes) - 1)
            else ""
        )

        parts.append(
            f'<span class="rq-node{endpoint}">'
            f"{safe(node)}</span>"
        )

    return (
        '<div class="rq-itinerary">'
        + "".join(parts)
        + "</div>"
    )


def route_card(route, result):
    status, tone = route_status(route, result)
    chosen = selected_route(result)

    selected_class = (
        " selected"
        if route["route_id"] == chosen["route_id"]
        else ""
    )

    html(
        f'<div class="rq-route-card{selected_class}">'
        f"{pill(status, tone)}"
        f'<div class="rq-route-title">'
        f"{safe(route['route_profile'])}</div>"
        f'<div class="rq-route-id">'
        f"{safe(route['route_id'])}</div>"
        '<div class="rq-card-grid">'
        "<div>"
        '<div class="rq-card-label">Shipping cost</div>'
        f'<div class="rq-card-value">'
        f"${route['total_cost']:,.0f}</div>"
        "</div>"
        "<div>"
        '<div class="rq-card-label">Expected time</div>'
        f'<div class="rq-card-value">'
        f"{route['expected_time']:.2f} d</div>"
        "</div>"
        "<div>"
        '<div class="rq-card-label">Composite risk</div>'
        f'<div class="rq-card-value">'
        f"{route['composite_risk']:.3f}</div>"
        "</div>"
        "<div>"
        '<div class="rq-card-label">Weighted score</div>'
        f'<div class="rq-card-value">'
        f"{route['score']:.4f}</div>"
        "</div>"
        "</div>"
        "</div>"
    )


# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------

def render_overview(shipment, routes, result, mode):
    selected = selected_route(result)
    explanation = explain_recommendation(result)

    status, tone = route_status(selected, result)

    panel_class = (
        "warning"
        if result["no_feasible_route"]
        else "recommended"
    )

    summary = (
        "No available route meets every requirement. "
        "The candidate below is diagnostic only; constraints "
        "have not been relaxed."
        if result["no_feasible_route"]
        else "Selected from the feasible Pareto frontier under "
        "your current business priorities."
    )

    html(
        f'<div class="rq-panel {panel_class}">'
        '<div class="rq-pills">'
        f"{pill(status, tone)}"
        f"{pill(result['profile'].title() + ' profile')}"
        f"{pill(shipment['cargo_type'])}"
        "</div>"
        f'<div class="rq-panel-title">'
        f"{safe(selected['route_profile'])} route</div>"
        f'<div class="rq-muted">{safe(summary)}</div>'
        f"{itinerary(selected, shipment)}"
        "</div>"
    )

    metric_row(selected)

    section(
        "Decision summary",
        "The recommendation reflects the configured requirements "
        "and priorities—not an absolute best route.",
    )

    left, right = st.columns([1.65, 1])

    with left:
        with st.container(border=True):
            st.markdown("**Why this route**")

            for reason in explanation["reasons"]:
                st.write(f"• {reason}")

            for comparison in explanation["comparisons"]:
                st.info(comparison)

    with right:
        with st.container(border=True):
            st.markdown("**Shipment brief**")

            st.write(f"**Cargo:** {shipment['cargo_type']}")
            st.write(
                f"**Declared value:** "
                f"${shipment['cargo_value']:,.0f}"
            )
            st.write(
                f"**Priority:** {shipment['shipment_priority']}"
            )
            st.write(
                f"**Budget:** ${shipment['max_budget']:,.0f}"
            )
            st.write(
                f"**Expected-time deadline:** "
                f"{shipment['delivery_deadline']:.2f} days"
            )
            st.write(
                f"**Minimum reliability:** "
                f"{shipment['min_reliability']:.1%}"
            )

            st.caption(
                "Expected-time feasibility is not a guarantee "
                "of arrival before the deadline."
            )

    section(
        "Your routing options",
        "Every alternative remains visible, including routes "
        "that fail the current requirements.",
    )

    ranked = result["ranking"].to_dict("records")

    for start in range(0, len(ranked), 3):
        columns = st.columns(3)

        for column, route in zip(
            columns,
            ranked[start:start + 3],
        ):
            with column:
                route_card(route, result)

    if mode == "New shipment":
        with st.expander("Simulated cost composition"):
            st.caption(
                "Handling and cargo-value additions are illustrative "
                "assumptions, not actual carrier or insurance prices."
            )

            st.dataframe(
                routes[
                    [
                        "route_profile",
                        "base_transport_cost",
                        "cargo_handling_fee",
                        "cargo_value_surcharge",
                        "total_cost",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )


# ------------------------------------------------------------------
# Route intelligence
# ------------------------------------------------------------------

def render_intelligence(result):
    table = result["ranking"].copy()

    table["Status"] = table["feasible"].map(
        {
            True: "Feasible",
            False: "Infeasible",
        }
    )

    table["Frontier"] = table["pareto_optimal"].map(
        {
            True: "Pareto",
            False: "Dominated / outside scope",
        }
    )

    table["marker_size"] = table["composite_risk"].clip(
        lower=0.02
    )

    section(
        "Cost versus expected delivery",
        f"Pareto membership is evaluated over: "
        f"{result['pareto_scope']}. Marker size represents "
        "composite risk.",
    )

    figure = px.scatter(
        table,
        x="expected_time",
        y="total_cost",
        color="Status",
        symbol="Frontier",
        size="marker_size",
        size_max=30,
        text="route_profile",
        hover_name="route_id",
        hover_data={
            "composite_risk": ":.3f",
            "score": ":.4f",
            "marker_size": False,
        },
        color_discrete_map={
            "Feasible": "#2dd4bf",
            "Infeasible": "#fb7185",
        },
        labels={
            "expected_time": "Expected delivery · days",
            "total_cost": "Shipping cost · USD",
        },
    )

    figure.update_traces(textposition="top center")
    draw_chart(figure, height=440)

    left, right = st.columns(2)

    with left:
        section(
            "Exposure profile",
            "The underlying indicators behind composite risk.",
        )

        risk = table.melt(
            id_vars=["route_profile"],
            value_vars=[
                "congestion_risk",
                "geopolitical_risk",
                "weather_risk",
            ],
            var_name="Indicator",
            value_name="Exposure",
        )

        figure = px.bar(
            risk,
            x="route_profile",
            y="Exposure",
            color="Indicator",
            barmode="group",
            color_discrete_sequence=COLORS,
            labels={"route_profile": "Route profile"},
        )

        figure.update_yaxes(range=[0, 1])
        draw_chart(figure)

    with right:
        section(
            "Score composition",
            "Lower contributions are better; hard constraints "
            "still take precedence.",
        )

        breakdown = table.melt(
            id_vars=["route_profile"],
            value_vars=[
                "cost_contribution",
                "time_contribution",
                "risk_contribution",
            ],
            var_name="Criterion",
            value_name="Contribution",
        )

        figure = px.bar(
            breakdown,
            x="route_profile",
            y="Contribution",
            color="Criterion",
            barmode="stack",
            color_discrete_sequence=COLORS,
            labels={"route_profile": "Route profile"},
        )

        draw_chart(figure)

    section(
        "Deadline uncertainty",
        "Model-based estimates under the stated delay assumption; "
        "not calibrated shipping forecasts.",
    )

    figure = px.bar(
        table,
        x="route_profile",
        y="on_time_probability",
        color="Status",
        color_discrete_map={
            "Feasible": "#2dd4bf",
            "Infeasible": "#fb7185",
        },
        labels={
            "route_profile": "Route profile",
            "on_time_probability": "Probability of meeting deadline",
        },
    )

    figure.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
    )

    draw_chart(figure, height=330)

    section(
        "Comparison ledger",
        "Eligible frontier routes rank first, followed by other "
        "feasible routes and then infeasible alternatives.",
    )

    st.dataframe(
        table[
            [
                "rank",
                "route_profile",
                "total_cost",
                "expected_time",
                "composite_risk",
                "reliability",
                "score",
                "feasible",
                "pareto_optimal",
                "budget_overrun_usd",
                "deadline_overrun_days",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------------
# Decision audit
# ------------------------------------------------------------------

def render_assessments(result):
    explanation = explain_recommendation(result)

    assessments = explanation.get("route_assessments")

    if assessments is None:
        st.error(
            "Update src/explainability.py to the detailed version "
            "that returns route_assessments."
        )
        return

    weights = result["weights"]

    html(
        '<div class="rq-topline">'
        '<div class="rq-pills">'
        f"{pill('Cost ' + format(weights['cost'], '.0%'))}"
        f"{pill('Time ' + format(weights['time'], '.0%'))}"
        f"{pill('Risk ' + format(weights['risk'], '.0%'))}"
        "</div>"
        f"{pill('Deterministic decision model', 'teal')}"
        "</div>"
    )

    for assessment in assessments:
        title = (
            f"{assessment['route_profile']}  ·  "
            f"{assessment['status']}"
        )

        with st.expander(
            title,
            expanded=assessment["is_selected"],
        ):
            st.caption(assessment["route_id"])
            st.write(assessment["path"])

            metrics = assessment["metrics"]
            columns = st.columns(4)

            columns[0].metric(
                "Cost",
                f"${metrics['cost']:,.0f}",
            )
            columns[1].metric(
                "Expected time",
                f"{metrics['expected_time']:.2f} d",
            )
            columns[2].metric(
                "Risk",
                f"{metrics['composite_risk']:.3f}",
            )
            columns[3].metric(
                "Score",
                f"{metrics['score']:.4f}",
            )

            st.markdown("#### Decision reasoning")

            for reason in assessment["decision_reasons"]:
                st.write(f"• {reason}")

            left, right = st.columns(2)

            with left:
                st.markdown("**Requirements satisfied**")

                for item in assessment["passed_constraints"]:
                    st.write(f"✓ {item}")

                if not assessment["passed_constraints"]:
                    st.caption("No hard requirements satisfied.")

            with right:
                st.markdown("**Requirements not satisfied**")

                for item in assessment["failed_constraints"]:
                    st.error(item)

                if not assessment["failed_constraints"]:
                    st.success("All hard requirements satisfied.")

            if assessment["comparison_with_selected"]:
                st.markdown("**Relative to the selected route**")

                for item in assessment[
                    "comparison_with_selected"
                ]:
                    st.write(f"• {item}")

            st.markdown("**Weighted-score calculation**")

            for item in assessment["score_breakdown"]:
                st.write(f"• {item}")

            if assessment["failed_constraints"]:
                st.caption(
                    "This score is diagnostic only. An attractive "
                    "score does not override a hard requirement."
                )

    section(
        "Numerical alternative comparisons",
        "Positive time savings and risk reductions favor the "
        "selected route. Positive cost change means a premium.",
    )

    st.dataframe(
        alternative_comparisons(result),
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------------
# Scenario lab
# ------------------------------------------------------------------

def render_scenarios(shipment, routes, profile, config, baseline):
    section(
        "Priority controls",
        "Adjust relative importance. Effective weights are "
        "normalized automatically.",
    )

    with st.container(border=True):
        columns = st.columns(3)

        with columns[0]:
            cost = st.slider(
                "Cost importance",
                0,
                100,
                40,
                key="scenario_cost",
            )

        with columns[1]:
            time = st.slider(
                "Time importance",
                0,
                100,
                30,
                key="scenario_time",
            )

        with columns[2]:
            risk = st.slider(
                "Risk importance",
                0,
                100,
                30,
                key="scenario_risk",
            )

        total = cost + time + risk

        if total == 0:
            st.error("At least one importance must be positive.")
            return

        weights = {
            "cost": cost / total,
            "time": time / total,
            "risk": risk / total,
        }

        st.caption(
            f"Effective weights — Cost {weights['cost']:.1%} · "
            f"Time {weights['time']:.1%} · "
            f"Risk {weights['risk']:.1%}"
        )

    custom = optimize_routes(
        routes,
        shipment,
        profile=profile,
        weights=weights,
        config=config,
    )

    chosen = selected_route(custom)

    st.info(
        explain_priority_change(
            previous=baseline,
            current=custom,
        )
    )

    if custom["no_feasible_route"]:
        st.warning(
            "No feasible route exists. Changing preferences cannot "
            "repair hard-constraint violations."
        )
    else:
        st.success(
            f"Scenario recommendation: {chosen['route_profile']} "
            f"· {chosen['route_id']}"
        )

    metric_row(chosen)

    with st.expander("Inspect this scenario's full reasoning"):
        render_assessments(custom)

    st.download_button(
        "Download scenario ranking",
        data=custom["ranking"].to_csv(index=False),
        file_name=f"{shipment['shipment_id']}_scenario_ranking.csv",
        mime="text/csv",
        key="download_scenario",
    )

    section(
        "Standard profile comparison",
        "A changed recommendation may be the correct response "
        "to changed business objectives.",
    )

    summary, scores = analyze_profiles(
        routes,
        shipment,
        config=config,
    )

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
    )

    matrix = scores.pivot(
        index="route_id",
        columns="profile",
        values="score",
    )

    figure = px.imshow(
        matrix,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale=[
            "#0d292b",
            "#0d9488",
            "#99f6e4",
        ],
        labels={
            "x": "Decision profile",
            "y": "Route",
            "color": "Score",
        },
    )

    draw_chart(figure)

    st.caption(
        "Scores are profile-specific. Compare routes within a "
        "profile rather than treating scores across profiles "
        "as absolute business utility."
    )


# ------------------------------------------------------------------
# Performance
# ------------------------------------------------------------------

def render_performance(shipments, routes, profile, config):
    st.info(
        "This workspace evaluates the synthetic test dataset. "
        "User-created shipments and temporary sidebar constraint "
        "overrides are not added to the benchmark."
    )

    with st.container(border=True):
        st.markdown("**Benchmark scope**")

        st.write(
            f"{len(shipments)} shipments · {len(routes)} alternatives "
            f"· {profile.title()} profile"
        )

        st.caption(
            "Uses the original dataset requirements and current "
            "model assumptions."
        )

        enabled = st.checkbox(
            "Calculate dataset benchmarks",
            key="enable_benchmarks",
        )

    if not enabled:
        section(
            "Ready when you are",
            "Enable the benchmark to calculate actual results. "
            "No evaluation numbers are shown before calculation.",
        )
        return

    with st.spinner("Evaluating decision consistency..."):
        evaluation = benchmark(
            shipments,
            routes,
            profile,
            config,
        )

    summary = evaluation["summary"]

    columns = st.columns(4)

    values = [
        (
            "Constraint satisfaction",
            percent(summary["constraint_satisfaction_rate_pct"]),
        ),
        (
            "Pareto efficiency",
            percent(summary["pareto_efficiency_rate_pct"]),
        ),
        (
            "Recommendation coverage",
            percent(
                summary["feasible_recommendation_coverage_pct"]
            ),
        ),
        (
            "No-feasible shipments",
            summary["no_feasible_shipments"],
        ),
    ]

    for column, (label, value) in zip(columns, values):
        column.metric(label, value)

    section(
        "Baseline comparison",
        "Every method uses the same feasible route pool. "
        "The fastest baseline minimizes base transit time.",
    )

    st.dataframe(
        evaluation["baseline_summary"],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Per-shipment decisions"):
        st.dataframe(
            evaluation["decisions"],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Methodology and interpretation"):
        for note in summary["notes"]:
            st.write(f"• {note}")

        st.write(
            f"Profile-sensitive shipments: "
            f"{summary['profile_sensitive_shipments']}"
        )

        st.write(
            f"Profile-change rate: "
            f"{percent(summary['profile_change_rate_pct'])}"
        )

    columns = st.columns(3)

    with columns[0]:
        st.download_button(
            "Summary JSON",
            json.dumps(
                summary,
                indent=2,
                allow_nan=False,
            ),
            file_name="evaluation_summary.json",
            mime="application/json",
        )

    with columns[1]:
        st.download_button(
            "Decision ledger CSV",
            evaluation["decisions"].to_csv(index=False),
            file_name="evaluation_decisions.csv",
            mime="text/csv",
        )

    with columns[2]:
        st.download_button(
            "Baseline summary CSV",
            evaluation["baseline_summary"].to_csv(index=False),
            file_name="baseline_summary.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------
# Export center
# ------------------------------------------------------------------

def render_exports(
    mode,
    shipment,
    routes,
    result,
    test_shipments,
    test_routes,
):
    chosen = selected_route(result)
    chosen_record = records(pd.DataFrame([chosen]))[0]

    payload = {
        "workflow": mode,
        "data_type": "synthetic",
        "shipment": records(pd.DataFrame([shipment]))[0],
        "status": result["status"],
        "profile": result["profile"],
        "weights": result["weights"],
        "config": result["config"],
        "recommendation": (
            chosen_record
            if not result["no_feasible_route"]
            else None
        ),
        "diagnostic_candidate": (
            chosen_record
            if result["no_feasible_route"]
            else None
        ),
        "pareto_scope": result["pareto_scope"],
        "explanation": explain_recommendation(result),
        "ranking": records(result["ranking"]),
        "limitations": [
            "Synthetic alternatives, not live shipping quotations.",
            "Paths are not geographically validated.",
            "Expected time does not guarantee on-time delivery.",
            "Risk indicators are not cargo-loss probabilities.",
        ],
    }

    shipment_id = shipment["shipment_id"]

    section(
        "Current shipment package",
        "Exports use the main sidebar profile. Custom scenario "
        "rankings are downloaded separately in Scenario lab.",
    )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### Decision evidence")
            st.write(
                "Recommendation, constraints, route assessments, "
                "model assumptions, and complete ranking."
            )

            st.download_button(
                "Download decision JSON",
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                file_name=f"{shipment_id}_decision.json",
                mime="application/json",
                use_container_width=True,
            )

            st.download_button(
                "Download ranking CSV",
                result["ranking"].to_csv(index=False),
                file_name=f"{shipment_id}_ranking.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with right:
        with st.container(border=True):
            st.markdown("### Planning inputs")
            st.write(
                "Shipment requirements and the route alternatives "
                "used in this analysis."
            )

            st.download_button(
                "Download shipment CSV",
                pd.DataFrame([shipment]).to_csv(index=False),
                file_name=f"{shipment_id}_shipment.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.download_button(
                "Download alternatives CSV",
                routes.to_csv(index=False),
                file_name=f"{shipment_id}_alternatives.csv",
                mime="text/csv",
                use_container_width=True,
            )

    section(
        "Inspect source inputs",
        "Inputs are available for review before export.",
    )

    with st.expander("Shipment requirements"):
        st.dataframe(
            pd.DataFrame([shipment]),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Route alternatives"):
        st.dataframe(
            routes,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Decision-model configuration"):
        st.json(
            {
                "active_profile": result["profile"],
                "active_weights": result["weights"],
                "engine": result["config"],
                "standard_profiles": WEIGHT_PROFILES,
                "normalization": (
                    "Min-max within the shipment's "
                    "complete alternative set"
                ),
            }
        )

    section(
        "Benchmark datasets",
        "These are separate from your user-created shipment.",
    )

    columns = st.columns(2)

    with columns[0]:
        st.download_button(
            "Test shipments CSV",
            test_shipments.to_csv(index=False),
            file_name="shipments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with columns[1]:
        st.download_button(
            "Test alternatives CSV",
            test_routes.to_csv(index=False),
            file_name="routes.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    inject_styles()

    try:
        test_shipments, test_routes = load_test_data()
    except (
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"Unable to load benchmark data: {exc}")

        st.code(
            "python -m src.data_generator --shipments 50 --seed 42",
            language="bash",
        )

        st.stop()

    (
        page,
        mode,
        shipment,
        routes,
        profile,
        config,
    ) = sidebar(test_shipments, test_routes)

    page_header(page, shipment)

    # The performance workspace can be used without creating a plan.
    if page == "Performance":
        render_performance(
            test_shipments,
            test_routes,
            profile,
            config,
        )

    elif shipment is None:
        empty_state()

    else:
        try:
            result = optimize_routes(
                routes,
                shipment,
                profile=profile,
                config=config,
            )
        except ValueError as exc:
            st.error(f"Unable to analyze the shipment: {exc}")
            st.stop()

        if page == "Overview":
            render_overview(
                shipment,
                routes,
                result,
                mode,
            )

        elif page == "Route intelligence":
            render_intelligence(result)

        elif page == "Decision audit":
            render_assessments(result)

        elif page == "Scenario lab":
            render_scenarios(
                shipment,
                routes,
                profile,
                config,
                result,
            )

        elif page == "Export center":
            render_exports(
                mode,
                shipment,
                routes,
                result,
                test_shipments,
                test_routes,
            )

    html(
        '<div class="rq-footer">'
        "RouteIQ · Explainable decision support<br>"
        "Synthetic scenarios only. No live carrier rates or verified "
        "maritime paths. User-created plans remain in the current "
        "session unless downloaded."
        "</div>"
    )


if __name__ == "__main__":
    main()