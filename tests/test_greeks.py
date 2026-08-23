"""Delta and gamma checked against centred finite differences.

Step sizes: the two derivatives want opposite things and get their own step.
The first difference is limited by truncation, O(h**2) * V''', so it wants h
small: h = P * 1e-5 lands the relative error near 1e-10. The second difference
cannot afford that - (V(P+h) - 2V(P) + V(P-h)) collapses into rounding noise
against values of order 1e5 - so it takes h = P * 1e-3 and is checked on a
relative tolerance, its residual error being ~3e-7 of gamma rather than absolute.
Both steps are scaled to the price, never fixed in absolute terms.

Points closer than h to Pa or Pb are excluded: gamma is discontinuous there, so a
centred difference straddling a bound is meaningless by construction, not by
numerical error.
"""

import math

import pytest

import il_math as ilm

L, PA, PB = 1000.0, 2000.0, 4000.0
DELTA_STEP = 1e-5    # truncation-limited: smaller is better
GAMMA_STEP = 1e-3    # cancellation-limited: smaller is catastrophic
IN_RANGE = [2100.0, 2500.0, 3000.0, 3500.0, 3900.0]
OUT_OF_RANGE = [1000.0, 1500.0, 4500.0, 6000.0]


def value(P: float) -> float:
    return ilm.position_value(L, P, PA, PB)


def far_from_bounds(P: float, h: float) -> bool:
    return abs(P - PA) > h and abs(P - PB) > h


@pytest.mark.parametrize("P", IN_RANGE + OUT_OF_RANGE)
def test_delta_matches_finite_difference(P):
    h = P * DELTA_STEP
    assert far_from_bounds(P, h)
    numeric = (value(P + h) - value(P - h)) / (2 * h)
    assert ilm.position_delta(L, P, PA, PB) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("P", IN_RANGE + OUT_OF_RANGE)
def test_gamma_matches_finite_difference(P):
    h = P * GAMMA_STEP
    assert far_from_bounds(P, h)
    numeric = (value(P + h) - 2 * value(P) + value(P - h)) / h ** 2
    # Relative tolerance: the residual is ~3e-7 of gamma, not an absolute quantity.
    assert ilm.position_gamma(L, P, PA, PB) == pytest.approx(numeric, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("P", IN_RANGE + OUT_OF_RANGE)
def test_delta_equals_the_eth_balance(P):
    """The identity worth stating: dV/dP is exactly the ETH held, in every regime."""
    x, _ = ilm.position_amounts(L, P, PA, PB)
    assert ilm.position_delta(L, P, PA, PB) == x


@pytest.mark.parametrize("P", IN_RANGE)
def test_gamma_is_strictly_negative_in_range(P):
    assert ilm.position_gamma(L, P, PA, PB) == pytest.approx(-L / (2 * P ** 1.5), rel=1e-12)
    assert ilm.position_gamma(L, P, PA, PB) < 0.0


@pytest.mark.parametrize("P", OUT_OF_RANGE)
def test_gamma_vanishes_out_of_range(P):
    assert ilm.position_gamma(L, P, PA, PB) == 0.0


@pytest.mark.parametrize("bound", [PA, PB])
def test_delta_is_continuous_at_the_bounds(bound):
    """C1: the one-sided deltas do not merely agree, their gap vanishes linearly.

    A jump would leave the gap constant as eps shrinks; here it falls by exactly
    one decade per decade of eps, which is what continuity looks like numerically.
    """
    gaps = [abs(ilm.position_delta(L, bound * (1 + eps), PA, PB)
                - ilm.position_delta(L, bound * (1 - eps), PA, PB))
            for eps in (1e-3, 1e-4, 1e-5)]
    assert gaps[1] == pytest.approx(gaps[0] / 10, rel=0.01)
    assert gaps[2] == pytest.approx(gaps[1] / 10, rel=0.01)


def test_gamma_jumps_at_the_bounds():
    """Not C2: gamma is bounded away from zero just inside and exactly zero just outside."""
    eps = 1e-9
    assert ilm.position_gamma(L, PA * (1 + eps), PA, PB) < -1e-4
    assert ilm.position_gamma(L, PA * (1 - eps), PA, PB) == 0.0
    assert ilm.position_gamma(L, PB * (1 - eps), PA, PB) < -1e-4
    assert ilm.position_gamma(L, PB * (1 + eps), PA, PB) == 0.0


def test_delta_at_the_upper_bound_is_zero():
    assert ilm.position_delta(L, PB, PA, PB) == pytest.approx(0.0, abs=1e-12)


def test_greeks_scale_linearly_with_liquidity():
    P = 3000.0
    assert ilm.position_delta(2 * L, P, PA, PB) == pytest.approx(2 * ilm.position_delta(L, P, PA, PB))
    assert ilm.position_gamma(2 * L, P, PA, PB) == pytest.approx(2 * ilm.position_gamma(L, P, PA, PB))


def test_full_range_gamma_is_the_v2_curvature():
    """With Pa = 0 and Pb = inf the position is a v2 LP: gamma = -L/(2*P**1.5) everywhere."""
    for P in (500.0, 3000.0, 9000.0):
        assert ilm.position_gamma(L, P, 0.0, math.inf) == pytest.approx(-L / (2 * P ** 1.5))


@pytest.mark.parametrize("P", [0.0, -1.0])
def test_gamma_rejects_non_positive_price(P):
    with pytest.raises(ValueError):
        ilm.position_gamma(L, P, PA, PB)
