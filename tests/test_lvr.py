"""Loss-versus-rebalancing, checked against the results it must reproduce.

The point of these tests is that LVR is not a new model bolted on: it is the
same curvature the greeks already measure, so it has to agree with the two
quantities the repository had before it - the sigma**2/8 constant-product rate
and the small-sigma limit of the breakeven quadrature.
"""

import math

import pytest

import il_math as ilm

L, PA, PB = 1000.0, 2000.0, 4000.0
P0 = 3000.0
SIGMA = 0.6438


@pytest.mark.parametrize("P", [500.0, 3000.0, 9000.0])
def test_full_range_yield_is_sigma_squared_over_eight(P):
    """The constant-product result, exact at every price and every volatility."""
    assert ilm.lvr_yield(L, P, 0.0, math.inf, SIGMA) == pytest.approx(SIGMA ** 2 / 8, rel=1e-12)


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.6, 1.2])
def test_full_range_yield_is_sigma_squared_over_eight_at_any_vol(sigma):
    assert ilm.lvr_yield(L, P0, 0.0, math.inf, sigma) == pytest.approx(sigma ** 2 / 8, rel=1e-12)


def test_lvr_yield_is_the_small_sigma_limit_of_the_breakeven():
    """Two independent routes to the same number, converging as sigma shrinks.

    breakeven_fee_apr integrates the terminal impermanent loss by quadrature;
    lvr_yield is a closed form in the curvature. They must meet in the limit.
    """
    errors = []
    for sigma in (0.4, 0.2, 0.1, 0.05):
        quadrature = ilm.breakeven_fee_apr(P0, 0.0, math.inf, sigma, 365)
        closed_form = ilm.lvr_yield(L, P0, 0.0, math.inf, sigma)
        errors.append(abs(quadrature - closed_form) / closed_form)
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 1e-3


@pytest.mark.parametrize("P", [1000.0, 1999.0, 4001.0, 8000.0])
def test_lvr_vanishes_out_of_range(P):
    """No curvature, no leak: out of range the position stops losing to arbitrage."""
    assert ilm.lvr_rate(L, P, PA, PB, SIGMA) == 0.0


@pytest.mark.parametrize("P", [2100.0, 3000.0, 3900.0])
def test_lvr_is_strictly_positive_in_range(P):
    assert ilm.lvr_rate(L, P, PA, PB, SIGMA) > 0.0


@pytest.mark.parametrize("P", [2100.0, 3000.0, 3900.0])
def test_lvr_is_the_gamma_identity(P):
    expected = 0.5 * SIGMA ** 2 * P ** 2 * -ilm.position_gamma(L, P, PA, PB)
    assert ilm.lvr_rate(L, P, PA, PB, SIGMA) == pytest.approx(expected, rel=1e-12)


def test_absolute_lvr_does_not_depend_on_the_bounds():
    """Same liquidity, same gamma: narrowing the range does not change the leak.

    What changes is the capital it is charged against, which is exactly why the
    yield rises while the rate stays put.
    """
    wide = ilm.lvr_rate(L, P0, P0 * 0.5, P0 * 1.5, SIGMA)
    narrow = ilm.lvr_rate(L, P0, P0 * 0.95, P0 * 1.05, SIGMA)
    assert wide == pytest.approx(narrow, rel=1e-12)
    assert ilm.lvr_yield(L, P0, P0 * 0.95, P0 * 1.05, SIGMA) > ilm.lvr_yield(
        L, P0, P0 * 0.5, P0 * 1.5, SIGMA)


def test_concentration_multiplier_grows_as_the_range_tightens():
    base = SIGMA ** 2 / 8
    multipliers = [ilm.lvr_yield(L, P0, P0 * (1 - w), P0 * (1 + w), SIGMA) / base
                   for w in (0.50, 0.20, 0.05)]
    assert multipliers == sorted(multipliers)
    assert multipliers[0] > 1.0


def test_lvr_scales_linearly_with_liquidity_and_quadratically_with_vol():
    assert ilm.lvr_rate(2 * L, P0, PA, PB, SIGMA) == pytest.approx(
        2 * ilm.lvr_rate(L, P0, PA, PB, SIGMA))
    assert ilm.lvr_rate(L, P0, PA, PB, 2 * SIGMA) == pytest.approx(
        4 * ilm.lvr_rate(L, P0, PA, PB, SIGMA))


def test_zero_volatility_means_no_leak():
    assert ilm.lvr_rate(L, P0, PA, PB, 0.0) == 0.0


def test_negative_volatility_raises():
    with pytest.raises(ValueError):
        ilm.lvr_rate(L, P0, PA, PB, -0.1)


def test_realised_lvr_is_zero_when_the_price_never_enters_the_band():
    """Out of range there is no curvature to arbitrage against, so nothing leaks."""
    import pandas as pd
    import analysis as an
    P0 = 100.0
    prices = pd.Series([P0] + [P0 * 5] * 99,
                       index=pd.date_range("2020-01-01", periods=100, freq="D"))
    table = an.lvr_table(P0, 0.6, prices.iloc[1:], 100_000.0).set_index("range")
    assert table.loc["+/-5%", "realised on the path %"] == pytest.approx(0.0, abs=1e-9)
    assert table.loc["full (v2)", "realised on the path %"] > 0.0


def test_realised_lvr_matches_the_in_range_rate_on_a_flat_path():
    """A price pinned at entry is in range every day, so both columns agree."""
    import pandas as pd
    import analysis as an
    P0 = 100.0
    prices = pd.Series([P0] * 50, index=pd.date_range("2020-01-01", periods=50, freq="D"))
    table = an.lvr_table(P0, 0.6, prices, 100_000.0).set_index("range")
    for label in ("+/-5%", "+/-20%", "+/-50%", "full (v2)"):
        assert table.loc[label, "realised on the path %"] == pytest.approx(
            table.loc[label, "LVR yield % in range"], rel=1e-6)
