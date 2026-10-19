from __future__ import annotations

import numpy as np
import pytest

from finance import (
    Project,
    analyze,
    compare,
    crossover_rate,
    discounted_cumulative_flows,
    discounted_payback,
    irr,
    npv,
    pv_of_cash_flows,
    profitability_index,
)


def test_npv_uses_end_of_year_discounting() -> None:
    project = Project("Test", 1_000.0, (500.0, 400.0, 300.0))
    expected = -1_000.0 + 500.0 / 1.1 + 400.0 / 1.1**2 + 300.0 / 1.1**3
    assert npv(project, 0.10) == pytest.approx(expected)


def test_npv_at_zero_rate_is_sum_of_net_cash_flows() -> None:
    project = Project("Test", 1_000.0, (500.0, 400.0, 300.0))
    assert npv(project, 0.0) == pytest.approx(200.0)


def test_npv_accepts_negative_future_cash_flows() -> None:
    project = Project("Test", 1_000.0, (600.0, -100.0, 600.0))
    expected = -1_000.0 + 600.0 / 1.1 - 100.0 / 1.1**2 + 600.0 / 1.1**3
    assert npv(project, 0.10) == pytest.approx(expected)


def test_irr_solves_the_npv_equation() -> None:
    project = Project("Test", 1_000.0, (500.0, 500.0, 500.0))
    rate = irr(project)
    assert rate is not None
    assert rate == pytest.approx(0.2337, abs=1e-4)
    assert npv(project, rate) == pytest.approx(0.0, abs=1e-5)


def test_irr_can_be_negative_when_cash_flows_underperform() -> None:
    project = Project("Weak", 1_000.0, (100.0, 100.0))
    rate = irr(project)
    assert rate is not None
    assert rate < 0
    assert npv(project, rate) == pytest.approx(0.0, abs=1e-5)


def test_irr_is_none_without_an_initial_outlay() -> None:
    assert irr(Project("No outlay", 0.0, (100.0, 100.0))) is None


def test_profitability_index() -> None:
    project = Project("Test", 1_000.0, (600.0, 600.0))
    expected = (600.0 / 1.1 + 600.0 / 1.1**2) / 1_000.0
    assert profitability_index(project, 0.10) == pytest.approx(expected)


def test_profitability_index_is_not_defined_without_initial_outlay() -> None:
    assert profitability_index(Project("Free", 0.0, (100.0,)), 0.10) is None


def test_discounted_payback_interpolates_within_the_recovery_year() -> None:
    project = Project("Test", 100.0, (60.0, 60.0))
    assert discounted_payback(project, 0.0) == pytest.approx(1 + 40.0 / 60.0)


def test_discounted_payback_returns_none_when_never_recovered() -> None:
    project = Project("Test", 1_000.0, (10.0, 10.0))
    assert discounted_payback(project, 0.10) is None


def test_discounted_cumulative_curve_ends_at_npv() -> None:
    project = Project("Test", 1_000.0, (500.0, 500.0, 500.0))
    trajectory = discounted_cumulative_flows(project, 0.10)
    assert trajectory[0] == -1_000.0
    assert trajectory[-1] == pytest.approx(npv(project, 0.10))


def test_crossover_rate_equalises_npvs() -> None:
    project_a = Project("A", 100.0, (50.0, 50.0))
    project_b = Project("B", 150.0, (80.0, 80.0))
    rate = crossover_rate(project_a, project_b)
    assert rate is not None
    assert 0 < rate < 2
    assert npv(project_a, rate) == pytest.approx(npv(project_b, rate), abs=1e-5)


def test_crossover_rate_is_none_when_one_project_dominates() -> None:
    project_a = Project("A", 100.0, (100.0, 100.0))
    project_b = Project("B", 100.0, (90.0, 90.0))
    assert crossover_rate(project_a, project_b) is None


def test_crossover_rate_is_none_for_identical_projects() -> None:
    project_a = Project("A", 100.0, (60.0, 60.0))
    project_b = Project("B", 100.0, (60.0, 60.0))
    assert crossover_rate(project_a, project_b) is None


def test_comparison_recommends_the_highest_positive_npv() -> None:
    project_a = Project("A", 100.0, (40.0, 40.0))
    project_b = Project("B", 100.0, (60.0, 60.0))
    comparison = compare(project_a, project_b, 0.10)
    assert comparison.recommended is not None
    assert comparison.recommended.project.name == "B"


def test_comparison_recommends_nothing_when_both_npvs_are_negative() -> None:
    project_a = Project("A", 1_000.0, (100.0,))
    project_b = Project("B", 1_000.0, (200.0,))
    comparison = compare(project_a, project_b, 0.10)
    assert comparison.project_a.npv < 0
    assert comparison.project_b.npv < 0
    assert comparison.recommended is None
    assert comparison.higher_npv.project.name == "B"


def test_comparison_supports_different_project_lives() -> None:
    project_a = Project("A", 100.0, (60.0,))
    project_b = Project("B", 100.0, (40.0, 40.0, 40.0))
    comparison = compare(project_a, project_b, 0.10)
    assert comparison.project_a.npv == pytest.approx(-100.0 + 60.0 / 1.1)
    assert comparison.project_b.npv == pytest.approx(
        -100.0 + 40.0 / 1.1 + 40.0 / 1.1**2 + 40.0 / 1.1**3
    )
    assert comparison.project_a.project.duration == 1
    assert comparison.project_b.project.duration == 3


def test_analysis_returns_the_expected_metrics() -> None:
    project = Project("Test", 1_000.0, (500.0, 500.0, 500.0))
    result = analyze(project, 0.10)
    assert result.npv == pytest.approx(npv(project, 0.10))
    assert result.pv_of_cash_flows == pytest.approx(pv_of_cash_flows(project, 0.10))
    assert result.irr is not None
    assert result.profitability_index is not None
    assert result.discounted_payback is not None


@pytest.mark.parametrize("rate", [-1.0, -2.0, np.nan, np.inf])
def test_invalid_discount_rates_are_rejected(rate: float) -> None:
    project = Project("Test", 100.0, (50.0,))
    with pytest.raises(ValueError):
        npv(project, rate)


def test_invalid_project_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        Project("", 100.0, (50.0,))
    with pytest.raises(ValueError):
        Project("Test", -100.0, (50.0,))
    with pytest.raises(ValueError):
        Project("Test", 100.0, ())
