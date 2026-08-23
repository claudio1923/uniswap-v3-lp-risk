"""Closed-form math for Uniswap v3 concentrated-liquidity positions.

Pure functions only: no I/O, no plotting, no module-level state.
Prices P, Pa, Pb are in USDC per ETH; x is ETH (token0), y is USDC (token1);
L is liquidity in sqrt(USDC*ETH); every value is denominated in USDC.
Pa = 0 with Pb = math.inf degenerates the position into a full-range Uniswap v2
LP, the benchmark implemented by il_full_range.
"""

import math

import numpy as np


def _validate_range(Pa: float, Pb: float) -> None:
    """Raise ValueError unless the bounds form a valid 0 <= Pa < Pb interval."""
    if Pa < 0.0 or math.isnan(Pa):
        raise ValueError(f"Pa must be non-negative, got {Pa}")
    if not Pb > Pa:
        raise ValueError(f"Pb must exceed Pa, got Pa={Pa}, Pb={Pb}")


def liquidity_from_amounts(x0: float, y0: float, P0: float, Pa: float, Pb: float) -> float:
    """Liquidity L in sqrt(USDC*ETH) from depositing x0 ETH and y0 USDC at spot P0.

    In range each leg implies its own liquidity and the binding one is the smaller:
    Lx = x0 / (1/sqrt(P0) - 1/sqrt(Pb)),  Ly = y0 / (sqrt(P0) - sqrt(Pa)).
    Out of range the position is single-sided, so one leg alone constrains L.
    """
    _validate_range(Pa, Pb)
    if P0 <= 0.0:
        raise ValueError(f"P0 must be strictly positive, got {P0}")
    if P0 <= Pa:                                    # entirely in ETH
        return x0 / (1.0 / math.sqrt(Pa) - 1.0 / math.sqrt(Pb))
    if P0 >= Pb:                                    # entirely in USDC
        return y0 / (math.sqrt(Pb) - math.sqrt(Pa))
    lx = x0 / (1.0 / math.sqrt(P0) - 1.0 / math.sqrt(Pb))
    ly = y0 / (math.sqrt(P0) - math.sqrt(Pa))
    return min(lx, ly)


def position_amounts(L: float, P: float, Pa: float, Pb: float) -> tuple[float, float]:
    """Token balances (x ETH, y USDC) held by liquidity L at spot P.

    Pa <= P <= Pb : x = L*(1/sqrt(P) - 1/sqrt(Pb)),  y = L*(sqrt(P) - sqrt(Pa))
    P < Pa        : x = L*(1/sqrt(Pa) - 1/sqrt(Pb)), y = 0   (fully in ETH)
    P > Pb        : x = 0,                           y = L*(sqrt(Pb) - sqrt(Pa))
    Out of range the LP has been bought out of one asset: the source of the loss.
    """
    _validate_range(Pa, Pb)
    if P <= 0.0:
        raise ValueError(f"P must be strictly positive, got {P}")
    if P < Pa:
        return L * (1.0 / math.sqrt(Pa) - 1.0 / math.sqrt(Pb)), 0.0
    if P > Pb:
        return 0.0, L * (math.sqrt(Pb) - math.sqrt(Pa))
    return L * (1.0 / math.sqrt(P) - 1.0 / math.sqrt(Pb)), L * (math.sqrt(P) - math.sqrt(Pa))


def position_value(L: float, P: float, Pa: float, Pb: float) -> float:
    """LP value in USDC: V = x*P + y, with (x, y) from position_amounts. No fees."""
    x, y = position_amounts(L, P, Pa, Pb)
    return x * P + y


def hold_value(x0: float, y0: float, P: float) -> float:
    """Buy-and-hold value in USDC of the initial basket: V = x0*P + y0."""
    return x0 * P + y0


def impermanent_loss(P: float, P0: float, Pa: float, Pb: float,
                     x0: float, y0: float) -> float:
    """Impermanent loss versus buy and hold at spot P: V_lp / V_hold - 1 (decimal).

    Negative means the LP lags the basket. The deposit (x0, y0) made at P0 fixes L
    and is itself the benchmark. Scale-invariant in (x0, y0): doubling both leaves
    the result unchanged. Excludes trading fees, gas and any rebalancing.
    """
    L = liquidity_from_amounts(x0, y0, P0, Pa, Pb)
    v_hold = hold_value(x0, y0, P)
    if v_hold <= 0.0:
        raise ValueError("buy and hold value must be strictly positive")
    return position_value(L, P, Pa, Pb) / v_hold - 1.0


def il_full_range(price_ratio: float) -> float:
    """Uniswap v2 impermanent loss: 2*sqrt(k)/(1+k) - 1, k = P/P0 (dimensionless).
    Limit of impermanent_loss as Pa -> 0 and Pb -> inf. Always <= 0, zero at k = 1.
    """
    if price_ratio <= 0.0:
        raise ValueError(f"price_ratio must be strictly positive, got {price_ratio}")
    return 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0


def breakeven_fee_apr(P0: float, Pa: float, Pb: float, sigma: float,
                      horizon_days: float, nodes: int = 128) -> float:
    """Minimum annualised fee yield (decimal, 0.25 = 25%/yr) offsetting expected IL.

    Assumes a driftless geometric Brownian motion for the price:
        P_T = P0 * exp(-0.5*sigma^2*T + sigma*sqrt(T)*Z),   Z ~ N(0, 1)
    so E[P_T] = P0: no directional view, only volatility matters. sigma is the
    annualised volatility of log returns and T = horizon_days / 365. E[IL] is
    evaluated by Gauss-Hermite quadrature (deterministic, no sampling) because the
    concentrated payoff has no tractable closed form; the result is -E[IL] / T.

    Simplification: fees are assumed to accrue on the whole position value for the
    whole horizon, while a real position earns nothing while out of range. The
    figure is therefore a LOWER bound on what the position must actually earn.
    """
    _validate_range(Pa, Pb)
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if horizon_days <= 0.0:
        raise ValueError(f"horizon_days must be strictly positive, got {horizon_days}")
    T = horizon_days / 365.0
    x0, y0 = position_amounts(1.0, P0, Pa, Pb)            # unit-liquidity reference
    t, w = np.polynomial.hermite.hermgauss(nodes)
    prices = P0 * np.exp(-0.5 * sigma ** 2 * T + sigma * math.sqrt(2.0 * T) * t)
    losses = np.array([impermanent_loss(float(p), P0, Pa, Pb, x0, y0) for p in prices])
    expected_il = float(w @ losses) / math.sqrt(math.pi)  # E[IL] under the GBM
    return -expected_il / T


if __name__ == "__main__":
    P0 = 3000.0
    # 1. Pa -> 0 and Pb -> inf must collapse exactly onto the v2 formula.
    x0, y0 = position_amounts(1.0, P0, 0.0, math.inf)
    for k in (0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 4.0):
        got, want = impermanent_loss(k * P0, P0, 0.0, math.inf, x0, y0), il_full_range(k)
        assert abs(got - want) < 1e-12, f"full range k={k}: {got} vs {want}"
    # 2. Same limit approached from a wide but finite range.
    lo, hi = P0 * 1e-9, P0 * 1e9
    xw, yw = position_amounts(1.0, P0, lo, hi)
    assert abs(impermanent_loss(2 * P0, P0, lo, hi, xw, yw) - il_full_range(2.0)) < 1e-4
    # 3. No price move means no loss; tighter ranges must lose more.
    assert abs(impermanent_loss(P0, P0, 0.0, math.inf, x0, y0)) < 1e-15
    losses = []
    for width in (0.05, 0.20, 0.50):
        lo, hi = P0 * (1 - width), P0 * (1 + width)
        xa, ya = position_amounts(1.0, P0, lo, hi)
        losses.append(impermanent_loss(1.2 * P0, P0, lo, hi, xa, ya))
    assert losses[0] < losses[1] < losses[2] < il_full_range(1.2), losses
    # 4. Out of range the position collapses to a single asset.
    assert position_amounts(1.0, 1000.0, 2000.0, 4000.0)[1] == 0.0
    assert position_amounts(1.0, 5000.0, 2000.0, 4000.0)[0] == 0.0
    # 5. Breakeven fee APR must rise as the range tightens.
    aprs = [breakeven_fee_apr(P0, P0*(1-w), P0*(1+w), 0.60, 365) for w in (0.05, 0.20, 0.50)]
    assert aprs[0] > aprs[1] > aprs[2] > 0.0, aprs
    print(f"il_math: checks passed | breakeven APR +/-5/20/50% = {aprs[0]:.1%}, {aprs[1]:.1%}, {aprs[2]:.1%}")
