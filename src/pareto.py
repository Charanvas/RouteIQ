import numpy as np
import pandas as pd


OBJECTIVES = [
    "total_cost",
    "expected_time",
    "composite_risk",
]


def pareto_mask(
    frame: pd.DataFrame,
    objectives: list[str] | None = None,
) -> np.ndarray:
    """
    Return True for nondominated rows.

    All objectives are minimized.
    Identical alternatives do not dominate one another.
    Complexity: O(n^2 * objectives), suitable for small route sets.
    """
    columns = objectives or OBJECTIVES
    values = frame[columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be finite.")

    mask = np.ones(len(values), dtype=bool)

    for index, candidate in enumerate(values):
        dominated = (
            np.all(values <= candidate, axis=1)
            & np.any(values < candidate, axis=1)
        )

        if dominated.any():
            mask[index] = False

    return mask