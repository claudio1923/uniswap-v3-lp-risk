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

    In range each leg implies its own liquidity, Lx = x0 / (1/sqrt(P0) - 1/sqrt(Pb))
    and Ly = y0 / (sqrt(P0) - sqrt(Pa)), and the two must agree: a deposit whose
    legs disagree cannot be placed in full, so this raises ValueError rather than
    quietly keeping the smaller and losing the surplus.
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
    if abs(lx - ly) > 1e-9 * max(lx, ly):
        raise ValueError(
            f"unbalanced deposit at P0={P0}: the ETH leg funds L={lx:.6g}, the USDC leg "
            f"L={ly:.6g}. A pool returns the surplus, so silently taking min(lx, ly) while "
            f"the buy-and-hold benchmark keeps the whole basket would report undeposited "
            f"capital as impermanent loss.")
    return lx


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


def position_delta(L: float, P: float, Pa: float, Pb: float) -> float:
    """dV/dP, in ETH: sensitivity of the position value to the spot price.

    Inside the range V = L*(2*sqrt(P) - P/sqrt(Pb) - sqrt(Pa)), so
        dV/dP = L*(1/sqrt(P) - 1/sqrt(Pb)) = x.
    Below Pa the position is a flat ETH holding and dV/dP = x again; above Pb it
    is pure USDC and dV/dP = 0, which is also x. Delta therefore equals the ETH
    balance in all three regimes, and is continuous at both bounds.
    """
    return position_amounts(L, P, Pa, Pb)[0]


def position_gamma(L: float, P: float, Pa: float, Pb: float) -> float:
    """d2V/dP2, in ETH per USDC: -L/(2*P**1.5) strictly inside the range, else 0.

    The value is concave in P wherever the position provides liquidity and linear
    outside, so gamma jumps at Pa and Pb, where the second derivative does not
    exist: the payoff is C1 but not C2. At the bounds this returns 0.

    Short gamma while collecting fees is the same structure as a sold straddle,
    with the fee APR playing the role of theta.
    """
    _validate_range(Pa, Pb)
    if P <= 0.0:
        raise ValueError(f"P must be strictly positive, got {P}")
    return -L / (2.0 * P ** 1.5) if Pa < P < Pb else 0.0


def lvr_rate(L: float, P: float, Pa: float, Pb: float, sigma: float) -> float:
    """Loss-versus-rebalancing, in USDC per year, at spot P.

    LVR is the rate at which the pool leaks value to arbitrageurs: the gap
    between the LP and a strategy holding the same instantaneous exposure while
    trading at market prices. Under a GBM it is driven entirely by curvature,

        LVR = (sigma^2 * P^2 / 2) * (-Gamma)

    so it inherits the gamma structure exactly: strictly positive inside the
    range, and identically zero outside it. That is the sharp statement behind
    the fee accounting - out of range the position stops bleeding to arbitrage
    at the same instant it stops collecting fees.

    Milionis, Moallemi, Roughgarden and Wan (2022), "Automated Market Making and
    Loss-Versus-Rebalancing". sigma is the annualised volatility of log returns.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    return 0.5 * sigma ** 2 * P ** 2 * -position_gamma(L, P, Pa, Pb)


def lvr_yield(L: float, P: float, Pa: float, Pb: float, sigma: float) -> float:
    """LVR as an annual fraction of position value: the fee APR that offsets it.

    For a full-range position this is exactly sigma**2/8, the constant-product
    result, which is also the small-sigma limit of breakeven_fee_apr. Narrowing
    the range leaves the absolute LVR untouched - same L, same gamma - while
    shrinking the capital it is charged against, so the yield rises. That ratio
    is the concentration multiplier.
    """
    value = position_value(L, P, Pa, Pb)
    if value <= 0.0:
        raise ValueError("position value must be strictly positive")
    return lvr_rate(L, P, Pa, Pb, sigma) / value
