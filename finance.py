"""Financial primitives used by the VAN project comparator.

All cash flows are assumed to occur at the end of each year. The initial
investment is made at t=0. Rates are expressed as decimals (0.10 = 10%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Project:
    """A simple investment project.

    Attributes:
        name: Display name of the project.
        initial_investment: Cash outlay at t=0 (a positive number).
        cash_flows: Net cash flows for years 1..n. They may be negative.
    """

    name: str
    initial_investment: float
    cash_flows: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("The project name cannot be empty.")
        if not np.isfinite(self.initial_investment) or self.initial_investment < 0:
            raise ValueError("The initial investment must be a finite, positive number.")
        if not self.cash_flows:
            raise ValueError("At least one year of cash flow is required.")
        if not all(np.isfinite(cash_flow) for cash_flow in self.cash_flows):
            raise ValueError("Cash flows must be finite numbers.")

    @property
    def duration(self) -> int:
        """Project duration in years."""
        return len(self.cash_flows)

    @property
    def total_net_cash_flow(self) -> float:
        """Undiscounted sum of all cash flows."""
        return float(sum(self.cash_flows))


@dataclass(frozen=True)
class ProjectAnalysis:
    """Result of analysing one project at a given discount rate."""

    project: Project
    discount_rate: float
    npv: float
    irr: float | None
    profitability_index: float | None
    discounted_payback: float | None
    pv_of_cash_flows: float


@dataclass(frozen=True)
class Comparison:
    """Analysis of two mutually exclusive projects."""

    project_a: ProjectAnalysis
    project_b: ProjectAnalysis
    npv_difference: float
    crossover_rate: float | None

    @property
    def higher_npv(self) -> ProjectAnalysis:
        """Project with the highest NPV, even when both are negative."""
        if self.project_a.npv >= self.project_b.npv:
            return self.project_a
        return self.project_b

    @property
    def lower_npv(self) -> ProjectAnalysis:
        """Project with the lowest NPV."""
        if self.project_a.npv >= self.project_b.npv:
            return self.project_b
        return self.project_a

    @property
    def recommended(self) -> ProjectAnalysis | None:
        """Best value-creating project, or None if neither passes the NPV test."""
        acceptable = [
            analysis
            for analysis in (self.project_a, self.project_b)
            if analysis.npv > 0
        ]
        if not acceptable:
            return None
        return max(acceptable, key=lambda analysis: analysis.npv)


def _validate_rate(rate: float) -> float:
    rate = float(rate)
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError("The discount rate must be finite and greater than -100%.")
    return rate


def present_value(cash_flow: float, rate: float, year: int) -> float:
    """Discount one cash flow received at the end of ``year``."""
    rate = _validate_rate(rate)
    if year < 0:
        raise ValueError("The year cannot be negative.")
    return float(cash_flow) / (1.0 + rate) ** year


def npv(project: Project, rate: float) -> float:
    """Net present value (VAN) of a project at ``rate``."""
    rate = _validate_rate(rate)
    discounted_flows = np.asarray(project.cash_flows, dtype=float) / (
        1.0 + rate
    ) ** np.arange(1, project.duration + 1)
    return float(-project.initial_investment + discounted_flows.sum())


def npv_from_flows(initial_investment: float, cash_flows: Sequence[float], rate: float) -> float:
    """NPV helper for arrays that are not wrapped in a :class:`Project`."""
    project = Project("Project", float(initial_investment), tuple(map(float, cash_flows)))
    return npv(project, rate)


def pv_of_cash_flows(project: Project, rate: float) -> float:
    """Present value of future cash flows, excluding the initial outlay."""
    return float(npv(project, rate) + project.initial_investment)


def profitability_index(project: Project, rate: float) -> float | None:
    """Present value of future cash flows per unit of initial investment."""
    if project.initial_investment <= 0:
        return None
    return pv_of_cash_flows(project, rate) / project.initial_investment


def discounted_payback(project: Project, rate: float) -> float | None:
    """Discounted payback period in years, or ``None`` if never recovered.

    A fractional year is interpolated within the year in which the cumulative
    discounted cash flow turns positive.
    """
    rate = _validate_rate(rate)
    if project.initial_investment <= 0:
        return 0.0

    cumulative = -project.initial_investment
    for year, cash_flow in enumerate(project.cash_flows, start=1):
        discounted_flow = present_value(cash_flow, rate, year)
        if cumulative + discounted_flow >= 0 and discounted_flow > 0:
            return float(year - 1 + (-cumulative / discounted_flow))
        cumulative += discounted_flow
    return None


def _solve_root(
    function,
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Bisection root solver with a known sign-changing bracket."""
    f_lower = function(lower)
    f_upper = function(upper)
    if abs(f_lower) < tolerance:
        return lower
    if abs(f_upper) < tolerance:
        return upper

    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        f_midpoint = function(midpoint)
        if abs(f_midpoint) < tolerance or (upper - lower) / 2.0 < tolerance:
            return float(midpoint)
        if f_lower * f_midpoint <= 0:
            upper, f_upper = midpoint, f_midpoint
        else:
            lower, f_lower = midpoint, f_midpoint
    return float((lower + upper) / 2.0)


def irr(project: Project, *, max_rate: float = 10.0) -> float | None:
    """Internal rate of return (TRI) for a project with a sign change.

    The rate grid is scanned before bisection so a root can be found even when
    the initial bracket idea ([-100%, 1000%]) is not enough. For conventional
    cash flows (one initial outflow followed by inflows), the IRR is unique.
    """
    if project.initial_investment <= 0:
        return None

    full_flows = np.asarray([-project.initial_investment, *project.cash_flows], dtype=float)
    if not (np.any(full_flows > 0) and np.any(full_flows < 0)):
        return None

    def function(rate: float) -> float:
        return float(np.sum(full_flows / (1.0 + rate) ** np.arange(full_flows.size)))

    rates = np.unique(
        np.concatenate(
            (
                np.linspace(-0.9999, -0.5, 400),
                np.linspace(-0.5, 1.0, 3000),
                np.linspace(1.0, max_rate, 1000),
            )
        )
    )
    values = np.array([function(rate) for rate in rates], dtype=float)
    scale = max(1.0, abs(project.initial_investment))
    tolerance = 1e-9 * scale

    roots: list[float] = []
    for index in range(len(rates) - 1):
        left_value, right_value = values[index], values[index + 1]
        if abs(left_value) <= tolerance:
            candidate = float(rates[index])
        elif left_value * right_value < 0:
            candidate = _solve_root(function, float(rates[index]), float(rates[index + 1]))
        else:
            continue
        if candidate > -1 + 1e-9 and not any(abs(candidate - root) < 1e-7 for root in roots):
            roots.append(candidate)

    if not roots:
        return None
    # Conventional projects have one IRR. If a non-conventional schedule has
    # several, show the smallest positive one for a more intuitive result.
    positive_roots = [root for root in roots if root >= 0]
    return float(min(positive_roots or roots))


def analyze(project: Project, rate: float) -> ProjectAnalysis:
    """Compute the dashboard metrics for one project."""
    rate = _validate_rate(rate)
    return ProjectAnalysis(
        project=project,
        discount_rate=rate,
        npv=npv(project, rate),
        irr=irr(project),
        profitability_index=profitability_index(project, rate),
        discounted_payback=discounted_payback(project, rate),
        pv_of_cash_flows=pv_of_cash_flows(project, rate),
    )


def _aligned_incremental_flows(project_a: Project, project_b: Project) -> np.ndarray:
    """Cash-flow differences at t=0, 1, ..., max duration."""
    horizon = max(project_a.duration, project_b.duration)
    flows_a = np.zeros(horizon + 1)
    flows_b = np.zeros(horizon + 1)
    flows_a[0] = -project_a.initial_investment
    flows_b[0] = -project_b.initial_investment
    flows_a[1 : project_a.duration + 1] = project_a.cash_flows
    flows_b[1 : project_b.duration + 1] = project_b.cash_flows
    return flows_a - flows_b


def crossover_rate(project_a: Project, project_b: Project, *, max_rate: float = 2.0) -> float | None:
    """Positive discount rate at which both projects have the same NPV.

    Returns ``None`` when one project dominates over the positive-rate range or
    when the projects have identical cash-flow profiles.
    """
    incremental = _aligned_incremental_flows(project_a, project_b)
    if np.allclose(incremental, 0.0):
        return None

    powers = np.arange(incremental.size)

    def difference(rate: float) -> float:
        return float(np.sum(incremental / (1.0 + rate) ** powers))

    rates = np.linspace(0.0, max_rate, 3000)
    values = np.array([difference(rate) for rate in rates])

    if abs(values[0]) <= 1e-9 * max(1.0, abs(incremental).sum()):
        return 0.0

    for index in range(len(rates) - 1):
        if values[index] * values[index + 1] < 0:
            root = _solve_root(difference, float(rates[index]), float(rates[index + 1]))
            if root > 1e-9:
                return root
    return None


def compare(project_a: Project, project_b: Project, rate: float) -> Comparison:
    """Analyse and compare two mutually exclusive projects."""
    analysis_a = analyze(project_a, rate)
    analysis_b = analyze(project_b, rate)
    return Comparison(
        project_a=analysis_a,
        project_b=analysis_b,
        npv_difference=analysis_a.npv - analysis_b.npv,
        crossover_rate=crossover_rate(project_a, project_b),
    )


def discounted_cumulative_flows(project: Project, rate: float) -> list[float]:
    """Cumulative NPV trajectory from t=0 through the project horizon."""
    rate = _validate_rate(rate)
    trajectory = [-project.initial_investment]
    cumulative = -project.initial_investment
    for year, cash_flow in enumerate(project.cash_flows, start=1):
        cumulative += present_value(cash_flow, rate, year)
        trajectory.append(cumulative)
    return trajectory
