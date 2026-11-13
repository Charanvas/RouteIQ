"""Data-grounded explanations for RouteIQ decisions.

Explanations describe the existing optimizer's results.
They do not change route selection.
"""

import pandas as pd


CRITERIA = ("cost", "time", "risk")


def route_path(route: dict, shipment: dict) -> str:
    hubs = [
        hub.strip()
        for hub in str(route["intermediate_hubs"]).split("|")
        if hub.strip()
    ]

    return " → ".join(
        [
            shipment["origin"],
            *hubs,
            shipment["destination"],
        ]
    )


def _relative_change(
    selected: float,
    reference: float,
) -> float:
    if reference == 0:
        return 0.0

    return 100.0 * (selected - reference) / reference


def _lower_is_better_comparison(
    candidate_value: float,
    selected_value: float,
    label: str,
    unit: str,
    decimals: int = 2,
) -> str:
    """Describe a candidate relative to the selected route."""
    difference = float(candidate_value) - float(selected_value)

    if abs(difference) < 1e-10:
        return f"{label} is equal to the selected route."

    magnitude = f"{abs(difference):,.{decimals}f}"

    if difference > 0:
        direction = "higher"
    else:
        direction = "lower"

    percentage = ""

    if selected_value != 0:
        percentage = (
            f" ({100 * abs(difference) / abs(selected_value):.1f}%)"
        )

    return (
        f"{label} is {magnitude}{unit} {direction}"
        f"{percentage} than the selected route."
    )


def _constraint_assessment(
    route: dict,
    shipment: dict,
) -> dict:
    cost = float(route["total_cost"])
    budget = float(shipment["max_budget"])

    expected_time = float(route["expected_time"])
    deadline = float(shipment["delivery_deadline"])

    reliability = float(route["reliability"])
    minimum_reliability = float(shipment["min_reliability"])

    passed = []
    failed = []

    if bool(route["budget_feasible"]):
        passed.append(
            f"Budget satisfied: ${cost:,.2f} ≤ ${budget:,.2f}; "
            f"${max(budget - cost, 0):,.2f} remains."
        )
    else:
        failed.append(
            f"Budget violated: ${cost:,.2f} exceeds "
            f"${budget:,.2f} by ${cost - budget:,.2f}."
        )

    if bool(route["deadline_feasible"]):
        passed.append(
            f"Expected-time deadline satisfied: "
            f"{expected_time:.2f} ≤ {deadline:.2f} days; "
            f"{max(deadline - expected_time, 0):.2f} days of "
            "expected-time slack."
        )
    else:
        failed.append(
            f"Expected-time deadline violated: "
            f"{expected_time:.2f} days exceeds "
            f"{deadline:.2f} days by "
            f"{expected_time - deadline:.2f} days."
        )

    if bool(route["reliability_feasible"]):
        passed.append(
            f"Reliability requirement satisfied: "
            f"{reliability:.2%} ≥ {minimum_reliability:.2%}; "
            f"{100 * max(reliability - minimum_reliability, 0):.2f} "
            "percentage points above the minimum."
        )
    else:
        failed.append(
            f"Reliability requirement violated: "
            f"{reliability:.2%} is below "
            f"{minimum_reliability:.2%} by "
            f"{100 * (minimum_reliability - reliability):.2f} "
            "percentage points."
        )

    return {
        "passed": passed,
        "failed": failed,
    }


def _find_dominators(
    candidate: dict,
    ranking: pd.DataFrame,
) -> list[dict]:
    """Find feasible routes dominating this feasible candidate."""
    objectives = [
        "total_cost",
        "expected_time",
        "composite_risk",
    ]

    pool = ranking.loc[
        ranking["feasible"]
        & ranking["route_id"].ne(candidate["route_id"])
    ]

    dominators = []

    for _, other in pool.iterrows():
        no_worse = all(
            float(other[column]) <= float(candidate[column])
            for column in objectives
        )

        strictly_better = any(
            float(other[column]) < float(candidate[column])
            for column in objectives
        )

        if no_worse and strictly_better:
            dominators.append(other.to_dict())

    return dominators


def _score_breakdown(
    route: dict,
    weights: dict,
) -> list[str]:
    lines = []

    for criterion in CRITERIA:
        normalized = float(route[f"{criterion}_norm"])
        contribution = float(
            route[f"{criterion}_contribution"]
        )

        lines.append(
            f"{criterion.title()}: normalized value "
            f"{normalized:.4f} × weight "
            f"{weights[criterion]:.2%} = "
            f"{contribution:.4f} score units."
        )

    lines.append(
        f"Total weighted score: {route['score']:.4f}. "
        "Lower is better among eligible routes."
    )

    return lines


def _raw_comparisons(
    candidate: dict,
    selected: dict,
) -> list[str]:
    lines = [
        _lower_is_better_comparison(
            candidate["total_cost"],
            selected["total_cost"],
            label="Shipping cost",
            unit=" USD",
        ),
        _lower_is_better_comparison(
            candidate["expected_time"],
            selected["expected_time"],
            label="Expected transit time",
            unit=" days",
        ),
        _lower_is_better_comparison(
            candidate["composite_risk"],
            selected["composite_risk"],
            label="Composite risk",
            unit="",
            decimals=3,
        ),
    ]

    geopolitical_difference = (
        float(candidate["geopolitical_risk"])
        - float(selected["geopolitical_risk"])
    )

    if abs(geopolitical_difference) < 1e-10:
        lines.append(
            "Geopolitical exposure is equal to the selected route."
        )
    else:
        direction = (
            "higher"
            if geopolitical_difference > 0
            else "lower"
        )

        lines.append(
            f"Geopolitical exposure is "
            f"{abs(geopolitical_difference):.3f} {direction} "
            "than the selected route."
        )

    reliability_difference = 100 * (
        float(candidate["reliability"])
        - float(selected["reliability"])
    )

    if abs(reliability_difference) < 1e-10:
        lines.append(
            "Historical reliability is equal to the selected route."
        )
    else:
        direction = (
            "higher"
            if reliability_difference > 0
            else "lower"
        )

        lines.append(
            f"Historical reliability is "
            f"{abs(reliability_difference):.2f} percentage points "
            f"{direction} than the selected route."
        )

    lines.append(
        "Reliability is a hard constraint, not an additional "
        "weighted scoring criterion."
    )

    return lines


def _weighted_comparison(
    candidate: dict,
    selected: dict,
) -> list[str]:
    """Explain score differences for two eligible routes."""
    differences = {
        criterion: (
            float(candidate[f"{criterion}_contribution"])
            - float(selected[f"{criterion}_contribution"])
        )
        for criterion in CRITERIA
    }

    disadvantages = {
        criterion: difference
        for criterion, difference in differences.items()
        if difference > 1e-12
    }

    advantages = {
        criterion: -difference
        for criterion, difference in differences.items()
        if difference < -1e-12
    }

    lines = []

    if disadvantages:
        details = ", ".join(
            f"{criterion} (+{difference:.4f})"
            for criterion, difference in disadvantages.items()
        )

        lines.append(
            "This alternative receives larger score contributions "
            f"than the selected route from: {details}. "
            "Larger contributions are disadvantages."
        )

    if advantages:
        details = ", ".join(
            f"{criterion} (−{difference:.4f})"
            for criterion, difference in advantages.items()
        )

        lines.append(
            "This alternative receives smaller score contributions "
            f"than the selected route from: {details}. "
            "These are genuine advantages of the alternative."
        )

    margin = (
        float(candidate["score"])
        - float(selected["score"])
    )

    if float(candidate["score"]) == float(selected["score"]):
        lines.append(
            "The weighted scores are exactly equal. "
            "The selected route wins because its route ID sorts "
            "first under the deterministic tie-breaking rule."
        )
    elif margin > 0:
        lines.append(
            f"The alternative's disadvantages outweigh its advantages "
            f"by {margin:.4f} score units: "
            f"{candidate['score']:.4f} versus "
            f"{selected['score']:.4f}. "
            "That is why it loses under the current priorities."
        )
    else:
        # Defensive explanation if the supplied result is inconsistent.
        lines.append(
            "This alternative has a lower weighted score than the "
            "reported selected route. If both are eligible, this "
            "indicates an inconsistency that should be investigated."
        )

    return lines


def _route_assessment(
    route: dict,
    result: dict,
    selected: dict,
) -> dict:
    shipment = result["shipment"]
    ranking = result["ranking"]

    constraints = _constraint_assessment(route, shipment)

    is_selected = route["route_id"] == selected["route_id"]

    decision_reasons = []
    comparisons = []

    # ---------------------------------------------------------------
    # No feasible route exists
    # ---------------------------------------------------------------
    if result["no_feasible_route"]:
        status = (
            "Diagnostic candidate — not an approved recommendation"
            if is_selected
            else "Infeasible alternative"
        )

        if is_selected:
            decision_reasons.append(
                "No route satisfies all hard requirements. "
                "This route is shown only because it has the smallest "
                "summed relative constraint violation."
            )
        else:
            violation_difference = (
                float(route["violation_total"])
                - float(selected["violation_total"])
            )

            if violation_difference > 0:
                decision_reasons.append(
                    "Its summed relative constraint violation is "
                    f"{route['violation_total']:.4f}, compared with "
                    f"{selected['violation_total']:.4f} for the "
                    "diagnostic candidate."
                )
            elif (
                float(route["score"])
                > float(selected["score"])
            ):
                decision_reasons.append(
                    "It ties the diagnostic candidate on total "
                    "relative violation but has a higher weighted score."
                )
            else:
                decision_reasons.append(
                    "It ties on the preceding diagnostic ranking "
                    "criteria and loses route-ID tie-breaking."
                )

        decision_reasons.append(
            "No requirement is relaxed automatically. "
            "The failed constraints below require operator action."
        )

    # ---------------------------------------------------------------
    # Valid selected recommendation
    # ---------------------------------------------------------------
    elif is_selected:
        status = "Selected recommendation"

        eligible_count = int(ranking["eligible"].sum())

        decision_reasons.extend(
            [
                "The route satisfies every hard constraint.",
                "It is not dominated by another feasible route "
                "on cost, expected time, and composite risk.",
                f"It has the lowest weighted score among "
                f"{eligible_count} eligible feasible Pareto "
                f"alternative(s), under the "
                f"{result['profile']} profile.",
            ]
        )

        tied = ranking.loc[
            ranking["eligible"]
            & ranking["route_id"].ne(route["route_id"])
            & ranking["score"].eq(route["score"])
        ]

        if not tied.empty:
            decision_reasons.append(
                "Another eligible route has exactly the same score; "
                "route-ID ordering resolves the tie."
            )

        feasible = ranking.loc[ranking["feasible"]]

        labels = {
            "total_cost": "lowest cost",
            "expected_time": "shortest expected transit time",
            "composite_risk": "lowest composite risk",
        }

        strengths = []
        tradeoffs = []

        for column, label in labels.items():
            minimum = float(feasible[column].min())
            value = float(route[column])

            if value == minimum:
                strengths.append(label)
            elif column == "total_cost":
                tradeoffs.append(
                    f"It costs ${value - minimum:,.2f} more "
                    "than the cheapest feasible route."
                )
            elif column == "expected_time":
                tradeoffs.append(
                    f"It takes {value - minimum:.2f} expected days "
                    "more than the fastest feasible route."
                )
            else:
                tradeoffs.append(
                    f"Its composite risk is {value - minimum:.3f} "
                    "above the lowest-risk feasible route."
                )

        if strengths:
            decision_reasons.append(
                "Among feasible alternatives, it achieves: "
                + ", ".join(strengths)
                + "."
            )

        decision_reasons.extend(tradeoffs)

        if tradeoffs:
            decision_reasons.append(
                "These disadvantages are accepted because its combined "
                "weighted score is lowest under the configured "
                "preferences—not because it is best on every metric."
            )

    # ---------------------------------------------------------------
    # Infeasible alternative
    # ---------------------------------------------------------------
    elif not bool(route["feasible"]):
        status = "Rejected — violates hard constraints"

        decision_reasons.append(
            "This route is excluded before final selection because "
            "it fails one or more mandatory requirements."
        )

        decision_reasons.append(
            "A lower price or attractive weighted score cannot "
            "compensate for a hard-constraint violation."
        )

        comparisons = _raw_comparisons(route, selected)

    # ---------------------------------------------------------------
    # Feasible but dominated alternative
    # ---------------------------------------------------------------
    elif not bool(route["pareto_optimal"]):
        status = "Rejected — Pareto dominated"

        dominators = _find_dominators(route, ranking)

        for other in dominators:
            decision_reasons.append(
                f"{other['route_id']} dominates this route: "
                f"cost ${other['total_cost']:,.2f} versus "
                f"${route['total_cost']:,.2f}; "
                f"expected time {other['expected_time']:.2f} versus "
                f"{route['expected_time']:.2f} days; "
                f"composite risk {other['composite_risk']:.3f} versus "
                f"{route['composite_risk']:.3f}. "
                "It is no worse on all three objectives and "
                "strictly better on at least one."
            )

        decision_reasons.append(
            "This route passes the hard requirements, but it is "
            "excluded from final selection because another feasible "
            "option offers an objectively better combination of the "
            "three modeled objectives."
        )

        comparisons = _raw_comparisons(route, selected)

    # ---------------------------------------------------------------
    # Feasible Pareto alternative losing on preferences
    # ---------------------------------------------------------------
    else:
        status = "Not selected — less preferred trade-off"

        decision_reasons.append(
            "This is a valid, nondominated alternative. "
            "It is not inherently a bad route; it is less preferred "
            "under the current business weights."
        )

        decision_reasons.extend(
            _weighted_comparison(route, selected)
        )

        comparisons = _raw_comparisons(route, selected)

    return {
        "route_id": route["route_id"],
        "route_profile": route["route_profile"],
        "status": status,
        "is_selected": is_selected,
        "path": route_path(route, shipment),
        "decision_reasons": decision_reasons,
        "passed_constraints": constraints["passed"],
        "failed_constraints": constraints["failed"],
        "comparison_with_selected": comparisons,
        "score_breakdown": _score_breakdown(
            route,
            result["weights"],
        ),
        "metrics": {
            "cost": float(route["total_cost"]),
            "expected_time": float(route["expected_time"]),
            "composite_risk": float(route["composite_risk"]),
            "reliability": float(route["reliability"]),
            "score": float(route["score"]),
        },
    }


def alternative_comparisons(result: dict) -> pd.DataFrame:
    """Retain the existing comparison-table interface."""
    selected = (
        result["recommended_route"]
        or result["fallback_candidate"]
    )

    rows = []

    for _, alternative in result["ranking"].iterrows():
        if alternative["route_id"] == selected["route_id"]:
            continue

        rows.append(
            {
                "alternative": alternative["route_id"],
                "alternative_feasible": bool(
                    alternative["feasible"]
                ),
                "selected_cost_change_pct": _relative_change(
                    selected["total_cost"],
                    alternative["total_cost"],
                ),
                "selected_time_saved_days": (
                    alternative["expected_time"]
                    - selected["expected_time"]
                ),
                "selected_risk_reduction": (
                    alternative["composite_risk"]
                    - selected["composite_risk"]
                ),
                "selected_reliability_gain_pp": (
                    100
                    * (
                        selected["reliability"]
                        - alternative["reliability"]
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def explain_recommendation(result: dict) -> dict:
    selected = (
        result["recommended_route"]
        or result["fallback_candidate"]
    )

    assessments = [
        _route_assessment(
            route=row.to_dict(),
            result=result,
            selected=selected,
        )
        for _, row in result["ranking"].iterrows()
    ]

    selected_assessment = next(
        assessment
        for assessment in assessments
        if assessment["is_selected"]
    )

    if result["no_feasible_route"]:
        headline = (
            f"No feasible route exists for "
            f"{result['shipment']['shipment_id']}."
        )
    else:
        headline = f"Recommend {selected['route_id']}."

    reasons = list(selected_assessment["decision_reasons"])

    reasons.append(
        f"Expected delivery is {selected['expected_time']:.2f} days, "
        f"composite risk is {selected['composite_risk']:.3f}, "
        f"and the historical reliability indicator is "
        f"{selected['reliability']:.2%}."
    )

    reasons.append(
        f"Model-based probability of meeting the deadline is "
        f"{selected['on_time_probability']:.2%}. "
        "This uses the assumed delay distribution and is not "
        "a calibrated real-world forecast."
    )

    comparisons = []

    if not result["no_feasible_route"]:
        feasible = result["ranking"].loc[
            result["ranking"]["feasible"]
        ]

        cheapest = feasible.sort_values(
            ["total_cost", "route_id"]
        ).iloc[0]

        if cheapest["route_id"] != selected["route_id"]:
            cost_change = _relative_change(
                selected["total_cost"],
                cheapest["total_cost"],
            )

            time_change = (
                selected["expected_time"]
                - cheapest["expected_time"]
            )

            risk_change = (
                selected["composite_risk"]
                - cheapest["composite_risk"]
            )

            comparisons.append(
                f"Compared with the cheapest feasible option "
                f"{cheapest['route_id']}, the selected route costs "
                f"{cost_change:.1f}% more. "
                f"Its expected-time difference is "
                f"{time_change:+.2f} days and its composite-risk "
                f"difference is {risk_change:+.3f}. "
                "Negative differences indicate improvements."
            )
        else:
            comparisons.append(
                "The recommendation is also the cheapest feasible route."
            )

    # Preserve compatibility with older dashboard consumers while
    # providing structured assessments for the improved UI.
    alternatives = [
        (
            f"{assessment['route_id']}: "
            + " ".join(
                assessment["decision_reasons"]
                + assessment["failed_constraints"]
                + assessment["comparison_with_selected"]
            )
        )
        for assessment in assessments
        if not assessment["is_selected"]
    ]

    return {
        "headline": headline,
        "path": route_path(selected, result["shipment"]),
        "reasons": reasons,
        "comparisons": comparisons,
        "alternatives": alternatives,
        "selected_assessment": selected_assessment,
        "route_assessments": assessments,
    }