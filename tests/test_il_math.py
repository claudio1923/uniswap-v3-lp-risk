"""Self-consistency tests for the concentrated-liquidity math.

Network-free by construction: il_math is pure and the analysis helpers tested
here are the ones that do not touch yfinance.
"""

import math

import pytest

import il_math as ilm

P0 = 3000.0
WIDTHS = (0.05, 0.20, 0.50)


def unit_position(Pa: float, Pb: float) -> tuple[float, float]:
    """Deposit implied by one unit of liquidity at P0."""
    return ilm.position_amounts(1.0, P0, Pa, Pb)


# --- The identity that anchors everything: Pa -> 0, Pb -> inf is Uniswap v2 ---

@pytest.mark.parametrize("k", [0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 4.0])
def test_full_range_matches_v2_formula(k):
    x0, y0 = unit_position(0.0, math.inf)
    assert ilm.impermanent_loss(k * P0, P0, 0.0, math.inf, x0, y0) == pytest.approx(
        ilm.il_full_range(k), abs=1e-12)


@pytest.mark.parametrize("k", [0.5, 2.0, 3.0])
def test_v2_formula_equals_sech_identity(k):
    """IL(k) = sech(X/2) - 1 with X = ln k; the form the -X^2/8 expansion comes from."""
    X = math.log(k)
    assert ilm.il_full_range(k) == pytest.approx(1.0 / math.cosh(X / 2) - 1.0, abs=1e-14)


def test_wide_finite_range_approaches_full_range():
    lo, hi = P0 * 1e-9, P0 * 1e9
    x0, y0 = unit_position(lo, hi)
    assert ilm.impermanent_loss(2 * P0, P0, lo, hi, x0, y0) == pytest.approx(
        ilm.il_full_range(2.0), abs=1e-4)


# --- Shape of the loss ---

def test_no_price_move_means_no_loss():
    x0, y0 = unit_position(0.0, math.inf)
    assert ilm.impermanent_loss(P0, P0, 0.0, math.inf, x0, y0) == pytest.approx(0.0, abs=1e-15)


def test_tighter_ranges_lose_more():
    losses = []
    for width in WIDTHS:
        lo, hi = P0 * (1 - width), P0 * (1 + width)
        x0, y0 = unit_position(lo, hi)
        losses.append(ilm.impermanent_loss(1.2 * P0, P0, lo, hi, x0, y0))
    assert losses[0] < losses[1] < losses[2] < ilm.il_full_range(1.2)


def test_impermanent_loss_is_scale_invariant():
    lo, hi = P0 * 0.8, P0 * 1.2
    x0, y0 = unit_position(lo, hi)
    single = ilm.impermanent_loss(1.1 * P0, P0, lo, hi, x0, y0)
    doubled = ilm.impermanent_loss(1.1 * P0, P0, lo, hi, 2 * x0, 2 * y0)
    assert single == pytest.approx(doubled, rel=1e-12)


# --- The three regimes ---

def test_below_range_is_all_eth():
    x, y = ilm.position_amounts(1.0, 1000.0, 2000.0, 4000.0)
    assert y == 0.0 and x > 0.0


def test_above_range_is_all_usdc():
    x, y = ilm.position_amounts(1.0, 5000.0, 2000.0, 4000.0)
    assert x == 0.0 and y > 0.0


def test_liquidity_round_trip():
    """Amounts -> L -> amounts must return the original deposit."""
    lo, hi = 2000.0, 4000.0
    x0, y0 = ilm.position_amounts(1234.0, P0, lo, hi)
    L = ilm.liquidity_from_amounts(x0, y0, P0, lo, hi)
    assert L == pytest.approx(1234.0, rel=1e-12)
    assert ilm.position_amounts(L, P0, lo, hi) == pytest.approx((x0, y0), rel=1e-12)


# --- Breakeven fee APR ---

def test_breakeven_rises_as_range_tightens():
    aprs = [ilm.breakeven_fee_apr(P0, P0 * (1 - w), P0 * (1 + w), 0.60, 365) for w in WIDTHS]
    assert aprs[0] > aprs[1] > aprs[2] > 0.0


@pytest.mark.parametrize("sigma,tol", [(0.10, 0.01), (0.20, 0.01), (0.40, 0.02)])
def test_full_range_breakeven_matches_small_sigma_expansion(sigma, tol):
    """E[IL] ~ -sigma^2*T/8, so APR* ~ sigma^2/8 as sigma -> 0.

    Independent check on the quadrature: it must reproduce the analytic
    asymptotics in the regime where the expansion holds, and the relative error
    must shrink with sigma.
    """
    numeric = ilm.breakeven_fee_apr(P0, 0.0, math.inf, sigma, 365)
    assert numeric == pytest.approx(sigma ** 2 / 8.0, rel=tol)


def test_expansion_error_shrinks_with_sigma():
    errors = [abs(ilm.breakeven_fee_apr(P0, 0.0, math.inf, s, 365) - s ** 2 / 8) / (s ** 2 / 8)
              for s in (0.10, 0.20, 0.40, 0.80)]
    assert errors == sorted(errors), errors


def test_zero_volatility_needs_no_fees():
    assert ilm.breakeven_fee_apr(P0, P0 * 0.9, P0 * 1.1, 0.0, 365) == pytest.approx(0.0, abs=1e-12)


# --- Input validation ---

@pytest.mark.parametrize("Pa,Pb", [(-1.0, 100.0), (100.0, 100.0), (200.0, 100.0)])
def test_invalid_ranges_raise(Pa, Pb):
    with pytest.raises(ValueError):
        ilm.position_amounts(1.0, 150.0, Pa, Pb)


@pytest.mark.parametrize("P", [0.0, -100.0])
def test_non_positive_price_raises(P):
    with pytest.raises(ValueError):
        ilm.position_amounts(1.0, P, 100.0, 200.0)


def test_negative_sigma_raises():
    with pytest.raises(ValueError):
        ilm.breakeven_fee_apr(P0, 0.0, math.inf, -0.1, 365)


def test_non_positive_horizon_raises():
    with pytest.raises(ValueError):
        ilm.breakeven_fee_apr(P0, 0.0, math.inf, 0.6, 0.0)


# --- A deposit whose legs disagree cannot be placed in full ---

@pytest.mark.parametrize("scale_x,scale_y", [(1.0, 2.0), (2.0, 1.0), (1.0, 1.001)])
def test_unbalanced_deposit_raises(scale_x, scale_y):
    """The surplus would otherwise be counted as loss instead of returned."""
    lo, hi = P0 * 0.8, P0 * 1.2
    x0, y0 = unit_position(lo, hi)
    with pytest.raises(ValueError, match="unbalanced deposit"):
        ilm.impermanent_loss(P0, P0, lo, hi, scale_x * x0, scale_y * y0)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 1000.0])
def test_a_balanced_deposit_of_any_size_is_accepted(scale):
    lo, hi = P0 * 0.8, P0 * 1.2
    x0, y0 = unit_position(lo, hi)
    assert ilm.impermanent_loss(P0, P0, lo, hi, scale * x0, scale * y0) == pytest.approx(0.0, abs=1e-12)
