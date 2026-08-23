"""Tests for the headline number: the breakeven corrected for time in range.

That figure is a product of two things il_math never sees - how many days the
price sat inside the band, and the division by that fraction - so it needs its
own coverage. Every series here has its in-range count fixed by construction,
so an off-by-one on a bound fails loudly instead of shifting the answer.
"""

import math

import pandas as pd
import pytest

import analysis as an
import il_math as ilm

P0 = 100.0
CAPITAL = 100_000.0


def series(values: list[float], year: int = 2020) -> pd.Series:
    """Daily series starting on 1 January, so P0 is values[0]."""
    return pd.Series(values, index=pd.date_range(f"{year}-01-01", periods=len(values), freq="D"))


def test_bounds_are_inclusive():
    """A price sitting exactly on Pa or Pb counts as in range, not out of it."""
    prices = series([P0, 80.0, 120.0, 79.9, 120.1])          # +/-20% band is [80, 120]
    row = an.year_table(prices).iloc[0]
    assert row["+/-20% in"] == pytest.approx(60.0)           # 100, 80 and 120 are inside


def test_in_range_share_matches_construction():
    prices = series([P0] * 30 + [P0 * 3] * 70)               # 30 inside every band, 70 outside
    row = an.year_table(prices).iloc[0]
    for label in ("+/-5%", "+/-20%", "+/-50%"):
        assert row[f"{label} in"] == pytest.approx(30.0)
    assert row["full (v2) in"] == pytest.approx(100.0)       # a full range is never out


def test_corrected_breakeven_is_the_nominal_divided_by_time_in_range():
    prices = series([P0] * 40 + [P0 * 3] * 60)
    table = an.year_table(prices)
    row = table.iloc[0]
    sigma = float((prices.apply(math.log).diff().std(ddof=1)) * math.sqrt(365))
    Pa, Pb = an.bounds(P0, 0.20)
    nominal = ilm.breakeven_fee_apr(P0, Pa, Pb, sigma, len(prices))
    # year_table rounds to one decimal, so compare within half a rounding unit.
    assert row["+/-20% corr"] == pytest.approx(100 * nominal / 0.40, abs=0.05)


def test_full_range_correction_is_a_no_op():
    """In range 100% of the time, so the corrected figure equals the nominal one."""
    prices = series([P0, 250.0, 40.0, 130.0] * 25)
    row = an.year_table(prices).iloc[0]
    sigma = float((prices.apply(math.log).diff().std(ddof=1)) * math.sqrt(365))
    nominal = ilm.breakeven_fee_apr(P0, 0.0, math.inf, sigma, len(prices))
    assert row["full (v2) corr"] == pytest.approx(100 * nominal, rel=1e-6)


def test_years_are_split_and_each_reprices_from_its_own_open():
    """Every year opens a fresh position: the second year must not inherit the first P0."""
    first = series([P0] * 10, year=2020)
    second = series([P0 * 4] * 10, year=2021)
    table = an.year_table(pd.concat([first, second]))
    assert list(table["year"]) == [2020, 2021]
    assert list(table["days"]) == [10, 10]
    # Both years are flat at their own opening price, so both sit fully in range.
    assert list(table["+/-5% in"]) == [100.0, 100.0]


def test_width_and_year_tables_agree_on_time_in_range():
    """The two tables compute the same quantity by different expressions.

    width_table counts days outside as (P < Pa) | (P > Pb); year_table counts days
    inside as (P >= Pa) & (P <= Pb). They must remain exact complements.
    """
    prices = series([P0] * 25 + [P0 * 0.5] * 35 + [P0 * 1.9] * 40)
    sigma = float((prices.apply(math.log).diff().std(ddof=1)) * math.sqrt(365))
    widths = an.width_table(P0, prices, sigma, CAPITAL).set_index("range")
    years = an.year_table(prices).iloc[0]
    for label in ("+/-5%", "+/-20%", "+/-50%", "full (v2)"):
        out = widths.loc[label, "out of range %"]
        assert out + years[f"{label} in"] == pytest.approx(100.0, abs=0.05)


def test_rank_correlation_is_perfect_on_a_monotone_column():
    label = f"+/-{100 * an.CONFIG['reference_width']:.0f}%"
    table = pd.DataFrame({"driver": [1.0, 2.0, 3.0, 4.0],
                          "mirror": [4.0, 3.0, 2.0, 1.0],
                          f"{label} corr": [10.0, 20.0, 30.0, 40.0]})
    assert an.rank_correlation(table, "driver") == pytest.approx(1.0)
    assert an.rank_correlation(table, "mirror") == pytest.approx(-1.0)


def test_a_flat_year_needs_no_fees():
    """No movement, no loss: the corrected breakeven collapses to zero."""
    prices = series([P0] * 50)
    row = an.year_table(prices).iloc[0]
    assert row["vol %"] == pytest.approx(0.0)
    assert math.isnan(row["|move|/vol"])          # undefined, not zero
    for label in ("+/-5%", "+/-20%", "+/-50%", "full (v2)"):
        assert row[f"{label} corr"] == pytest.approx(0.0, abs=1e-9)
