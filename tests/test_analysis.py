"""Tests for the analysis helpers that do not touch the network."""

import math

import pytest

import analysis as an
import il_math as ilm

P0 = 2500.0
CAPITAL = 100_000.0


def test_bounds_are_symmetric_around_entry():
    Pa, Pb = an.bounds(P0, 0.20)
    assert (Pa, Pb) == pytest.approx((2000.0, 3000.0))


def test_none_width_means_full_range():
    assert an.bounds(P0, None) == (0.0, math.inf)


def test_labelled_ranges_covers_every_width_plus_benchmark():
    labels = [label for label, _, _ in an.labelled_ranges(P0)]
    assert labels == ["+/-5%", "+/-20%", "+/-50%", "full (v2)"]


@pytest.mark.parametrize("width", [0.05, 0.20, 0.50, None])
def test_deposit_uses_exactly_the_capital(width):
    """The (x0, y0) split must be worth `capital` at the entry price."""
    Pa, Pb = an.bounds(P0, width)
    x0, y0, L = an.deposit(P0, Pa, Pb, CAPITAL)
    assert ilm.position_value(L, P0, Pa, Pb) == pytest.approx(CAPITAL, rel=1e-10)
    assert ilm.hold_value(x0, y0, P0) == pytest.approx(CAPITAL, rel=1e-10)


def test_synthetic_prices_are_deterministic():
    first, second = an.synthetic_prices(), an.synthetic_prices()
    assert first.equals(second)
    assert len(first) > 300 and (first > 0).all()


def test_scenario_table_is_flat_at_entry():
    table = an.scenario_table(P0, 0.20, CAPITAL)
    at_entry = table.loc[table["scenario"] == "+0%"].iloc[0]
    assert at_entry["IL %"] == pytest.approx(0.0, abs=1e-9)
    assert at_entry["LP"] == at_entry["hold"] == pytest.approx(CAPITAL, rel=1e-6)
