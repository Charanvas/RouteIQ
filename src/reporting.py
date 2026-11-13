from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import REPORTS_DIR
from src.evaluation import (
    evaluate_dataset,
    load_or_generate_dataset,
    save_evaluation,
)


def generate_report(evaluation: dict, output_path=None):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = output_path or REPORTS_DIR / "technical_report.pdf"

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13

    story = []

    def heading(text):
        story.append(Paragraph(escape(text), styles["Heading2"]))

    def paragraph(text):
        story.append(Paragraph(escape(text), styles["BodyText"]))
        story.append(Spacer(1, 8))

    def add_table(rows, widths=None):
        table = Table(
            rows,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f2f5f9")],
                    ),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 12))

    # Page 1
    story.append(
        Paragraph(
            "RouteIQ: Explainable Shipment Route Selection",
            styles["Title"],
        )
    )

    heading("1. Problem and definition of optimality")
    paragraph(
        "RouteIQ selects a shipment route by balancing transportation "
        "cost, expected delivery time, and operational exposure. Optimality "
        "is conditional on the available alternatives, business constraints, "
        "configured weights, and simulation assumptions. The system does "
        "not claim global optimality over a real maritime network."
    )

    heading("2. Synthetic dataset")
    paragraph(
        "The default generator creates 50 shipments with five alternatives "
        "each, using 14 named logistics nodes. Economy, Express, Balanced, "
        "High Risk, and Reliable profiles introduce deliberate cost-time-risk "
        "trade-offs. Stops add handling cost and time. Congestion, geopolitical "
        "exposure, and weather influence delay probability and reliability. "
        "Cargo rules tighten pharmaceutical reliability and perishable-goods "
        "deadlines. Some shipments are deliberately infeasible."
    )

    paragraph(
        "A local seeded NumPy generator provides repeatability within a "
        "fixed software environment. Routes exclude origin/destination from "
        "intermediate hubs. Input validation checks identifiers, numeric "
        "ranges, finite values, hub counts, and shipment references."
    )

    heading("3. Architecture")
    paragraph(
        "Data generation -> validation and feature engineering -> feasibility "
        "filter -> Pareto frontier -> weighted selection -> deterministic "
        "explanations -> Streamlit dashboard. The optimizer is independent "
        "of the interface and does not use an LLM."
    )

    story.append(PageBreak())

    # Page 2
    heading("4. Decision framework")
    paragraph(
        "Expected time = base transit time + delay probability x mean delay. "
        "The default mean delay is five days. Composite risk = "
        "0.35 x congestion + 0.40 x geopolitical exposure + 0.25 x weather. "
        "These coefficients are configurable assumptions, not calibrated "
        "industry measurements."
    )

    paragraph(
        "Hard constraints require cost <= budget, expected time <= deadline, "
        "and reliability >= the shipment requirement. Among feasible routes, "
        "a route is dominated if another is no worse on cost, expected time, "
        "and composite risk and strictly better on at least one."
    )

    paragraph(
        "Each criterion is min-max normalized within the shipment's complete "
        "alternative set. A constant criterion maps to zero. The final score "
        "is the weighted sum of normalized cost, expected time, and risk; "
        "lower is better. Selection is restricted to the feasible Pareto set. "
        "Route ID resolves exact score ties."
    )

    add_table(
        [
            ["Profile", "Cost", "Time", "Risk"],
            ["Cost", "0.60", "0.20", "0.20"],
            ["Time", "0.20", "0.60", "0.20"],
            ["Risk", "0.20", "0.20", "0.60"],
            ["Balanced", "0.40", "0.30", "0.30"],
        ],
        [200, 85, 85, 85],
    )

    heading("5. Infeasibility, uncertainty, and explanations")
    paragraph(
        "When no route is feasible, the engine returns no recommendation. "
        "A separately labeled diagnostic candidate minimizes summed relative "
        "budget, expected-time, and reliability shortfalls. Constraints are "
        "never relaxed automatically."
    )

    paragraph(
        "A supplementary on-time estimate assumes a Bernoulli delay event "
        "and exponentially distributed positive delay duration. This estimate "
        "is not an additional scoring criterion or a calibrated forecast. "
        "Meeting an expected-time deadline does not guarantee on-time arrival."
    )

    paragraph(
        "Explanations report actual constraints, price premiums, time "
        "differences, score contributions, and rejection reasons. Historical "
        "reliability is used as a constraint and explanatory indicator, not "
        "as another weighted objective."
    )

    story.append(PageBreak())

    # Page 3
    heading("6. Evaluation results")

    summary = evaluation["summary"]

    metric_labels = [
        ("Shipments evaluated", "shipments_evaluated"),
        ("Route alternatives", "route_alternatives"),
        ("Shipments with feasible routes", "shipments_with_feasible_routes"),
        ("No-feasible shipments", "no_feasible_shipments"),
        ("Constraint satisfaction (%)", "constraint_satisfaction_rate_pct"),
        ("Recommendation coverage (%)", "feasible_recommendation_coverage_pct"),
        ("Pareto efficiency (%)", "pareto_efficiency_rate_pct"),
        ("Profile-sensitive shipments", "profile_sensitive_shipments"),
        ("Profile-change rate (%)", "profile_change_rate_pct"),
    ]

    rows = [["Metric", "Observed value"]]

    for label, key in metric_labels:
        value = summary[key]
        rows.append([label, "N/A" if value is None else str(value)])

    add_table(rows, [330, 125])

    paragraph(
        "Constraint satisfaction and Pareto efficiency use only shipments "
        "with at least one feasible route. Recommendation coverage uses all "
        "shipments. Cheapest, fastest-base-time, and lowest-risk baselines "
        "use the same feasibility constraints. Detailed baseline aggregates "
        "and per-shipment decisions are exported separately as CSV files."
    )

    heading("7. Limitations and next steps")
    paragraph(
        "The named-node paths are synthetic and are not checked for "
        "geographic or operational feasibility. Risk and reliability are "
        "simulated, not measured. Risk affects both expected delay and an "
        "exposure objective intentionally, so the criteria are correlated. "
        "There are no vessel schedules, capacities, customs rules, live "
        "prices, or network-wide resource constraints."
    )

    paragraph(
        "Min-max scoring is sensitive to the available alternative set; "
        "adding an extreme route can change normalized trade-offs. Weighted "
        "sums may not select every nondominated point. Future work should "
        "calibrate delays and risks, validate against real outcomes, add "
        "chance constraints or robust optimization, and model actual legs "
        "and capacity interactions."
    )

    paragraph(
        "Reproduction: generate the dataset, run pytest, run "
        "python -m src.evaluation, then python -m src.reporting. "
        "The current report is generated from the current dataset using "
        "the default engine configuration and automatic cargo profiles."
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 22, "RouteIQ | Synthetic decision-support prototype")
        canvas.drawRightString(
            A4[0] - 40,
            22,
            f"Page {document.page}",
        )
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=38,
    )

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    return output_path


def main():
    shipments, routes = load_or_generate_dataset()

    evaluation = evaluate_dataset(shipments, routes)
    save_evaluation(evaluation)

    output_path = generate_report(evaluation)

    print(f"Generated report: {output_path}")


if __name__ == "__main__":
    main()