"""Risk analysis of a concentrated ETH/USDC liquidity position on Uniswap v3.

Loads 2024 daily ETH prices, prices LP ranges against buy and hold and writes the
two figures used by the README. Every number is computed in il_math.
"""

import logging, math
from pathlib import Path

import matplotlib; matplotlib.use("Agg")   # headless backend: render straight to PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import il_math as ilm

# --- CONFIG -----------------------------------------------------------------
CONFIG = {
    "ticker": "ETH-USD",
    "start": "2024-01-01",
    "end": "2025-01-01",                     # exclusive, yfinance convention
    "capital_usdc": 100_000.0,
    "range_widths": (0.05, 0.20, 0.50),      # half-width around the entry price
    "reference_width": 0.20,                 # width detailed in the scenario table
    "price_scenarios": (-0.50, -0.20, 0.0, 0.20, 0.50),
    "horizon_days": 365, "days_per_year": 365,   # crypto trades every calendar day
    "fallback_sigma": 0.60,                  # annualised, used only if yfinance fails
    "fallback_p0": 2300.0, "fallback_seed": 7,
    "sweep_points": 60,                      # widths sampled for breakeven.png
    "greeks_span": (0.5, 1.8),               # price window of greeks.png, as a multiple of P0
    "figures_dir": Path(__file__).resolve().parent / "figures",
}
LOG = logging.getLogger("lp-risk")

def synthetic_prices() -> pd.Series:
    """Seeded driftless GBM daily path, used only when yfinance is unreachable."""
    dates = pd.date_range(CONFIG["start"], CONFIG["end"], freq="D", inclusive="left")
    dt, sigma = 1.0 / CONFIG["days_per_year"], CONFIG["fallback_sigma"]
    shocks = np.random.default_rng(CONFIG["fallback_seed"]).normal(
        -0.5 * sigma ** 2 * dt, sigma * math.sqrt(dt), len(dates))
    return pd.Series(CONFIG["fallback_p0"] * np.exp(np.cumsum(shocks)), index=dates)

def load_prices() -> tuple[pd.Series, str]:
    """Daily ETH close for the configured window, with a synthetic GBM fallback."""
    try:
        import yfinance as yf
        raw = yf.download(CONFIG["ticker"], start=CONFIG["start"], end=CONFIG["end"],
                          progress=False, auto_adjust=True, threads=False)
        if raw is None or raw.empty:
            raise RuntimeError("empty response")
        close = raw["Close"]                 # MultiIndex when several tickers are asked
        close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        return close.astype(float), "yfinance ETH-USD"
    except Exception as exc:                 # network, rate limit, parsing, missing dep
        LOG.warning("yfinance unavailable (%s: %s) - falling back to synthetic GBM",
                    type(exc).__name__, exc)
        return synthetic_prices(), f"synthetic GBM sigma={CONFIG['fallback_sigma']:.0%}"

def bounds(P0: float, width: float | None) -> tuple[float, float]:
    """Range bounds for a +/- width band around P0; width=None means full range."""
    return (0.0, math.inf) if width is None else (P0 * (1 - width), P0 * (1 + width))

def labelled_ranges(P0: float) -> list[tuple[str, float, float]]:
    """(label, Pa, Pb) for every configured width plus the full-range v2 benchmark."""
    return [("full (v2)" if w is None else f"+/-{100 * w:.0f}%", *bounds(P0, w))
            for w in list(CONFIG["range_widths"]) + [None]]

def deposit(P0: float, Pa: float, Pb: float, capital: float) -> tuple[float, float, float]:
    """Split `capital` USDC into (x0 ETH, y0 USDC, L); value is linear in L."""
    L = capital / ilm.position_value(1.0, P0, Pa, Pb)
    return *ilm.position_amounts(L, P0, Pa, Pb), L

def scenario_table(P0: float, width: float, capital: float) -> pd.DataFrame:
    """LP value, buy-and-hold value and IL across the configured price shocks."""
    Pa, Pb = bounds(P0, width)
    x0, y0, L = deposit(P0, Pa, Pb, capital)
    rows = []
    for shock in CONFIG["price_scenarios"]:
        P = P0 * (1 + shock)
        v_lp, v_hold = ilm.position_value(L, P, Pa, Pb), ilm.hold_value(x0, y0, P)
        rows.append({"scenario": f"{shock:+.0%}", "price": round(P), "LP": round(v_lp),
                     "hold": round(v_hold), "IL %": round(100 * (v_lp / v_hold - 1), 2),
                     "IL USDC": round(v_lp - v_hold)})
    return pd.DataFrame(rows)

def width_table(P0: float, prices: pd.Series, sigma: float, capital: float) -> pd.DataFrame:
    """Per range width: IL at each shock, share of days out of range, breakeven APR."""
    rows = []
    for label, Pa, Pb in labelled_ranges(P0):
        x0, y0, _ = deposit(P0, Pa, Pb, capital)
        row = {"range": label}
        for shock in CONFIG["price_scenarios"]:
            row[f"IL {shock:+.0%}"] = round(
                100 * ilm.impermanent_loss(P0 * (1 + shock), P0, Pa, Pb, x0, y0), 2)
        # Out of range the position earns no fees: the metric that decides viability.
        row["out of range %"] = round(100 * float(((prices < Pa) | (prices > Pb)).mean()), 1)
        apr = ilm.breakeven_fee_apr(P0, Pa, Pb, sigma, CONFIG["horizon_days"])
        row["breakeven APR %"] = round(100 * apr, 1)
        rows.append(row)
    return pd.DataFrame(rows)

def plot_il_curves(P0: float, capital: float, path: Path) -> None:
    """IL versus price move for each width plus the full-range v2 benchmark."""
    ratios = np.linspace(0.4, 2.0, 400)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, Pa, Pb in labelled_ranges(P0):
        x0, y0, _ = deposit(P0, Pa, Pb, capital)
        ax.plot(100 * (ratios - 1),
                [100 * ilm.impermanent_loss(r * P0, P0, Pa, Pb, x0, y0) for r in ratios],
                label=label, linewidth=1.8, linestyle="--" if "full" in label else "-")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="ETH price move vs entry (%)", ylabel="Impermanent loss (%)",
           title=f"IL vs buy and hold, entry at {P0:,.0f} USDC/ETH")
    ax.grid(alpha=0.3), ax.legend(title="LP range")
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)

def plot_breakeven(P0: float, sigma: float, path: Path) -> None:
    """Breakeven fee APR as a function of range width, at the realised volatility."""
    def apr(w): return 100 * ilm.breakeven_fee_apr(P0, *bounds(P0, w), sigma, CONFIG["horizon_days"])
    widths = np.linspace(0.02, 1.0, CONFIG["sweep_points"])
    marks = [(100 * w, apr(w)) for w in CONFIG["range_widths"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(100 * widths, [apr(w) for w in widths], linewidth=2.0, color="#1f77b4")
    ax.scatter(*zip(*marks), color="#d62728", zorder=3,
               label="  ".join(f"+/-{x:.0f}% -> {y:.0f}%" for x, y in marks))
    ax.set(xlabel="Range half-width around entry price (%)", ylabel="Breakeven fee APR (%)",
           title=f"Fee APR needed to offset expected IL | sigma={sigma:.0%}, "
                 f"{CONFIG['horizon_days']}d horizon")
    ax.grid(alpha=0.3), ax.legend()
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)

def plot_greeks(P0: float, capital: float, path: Path) -> None:
    """Delta and gamma against spot for the reference range, on a shared price axis.

    The kink of the delta and the jump of the gamma both sit on Pa and Pb: the
    payoff is C1 but not C2, and the picture is the quickest way to see it.
    """
    width = CONFIG["reference_width"]
    Pa, Pb = bounds(P0, width)
    lo, hi = CONFIG["greeks_span"]
    _, _, L = deposit(P0, Pa, Pb, capital)
    prices = np.linspace(lo * P0, hi * P0, 800)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(prices, [ilm.position_delta(L, p, Pa, Pb) for p in prices], color="#1f77b4", lw=2)
    top.set(ylabel="Delta  dV/dP  (ETH)",
            title=f"Greeks of a {capital:,.0f} USDC position, range +/-{100 * width:.0f}%")
    bottom.plot(prices, [ilm.position_gamma(L, p, Pa, Pb) for p in prices], color="#d62728", lw=2)
    bottom.set(xlabel="ETH spot price (USDC)", ylabel="Gamma  d2V/dP2  (ETH per USDC)")
    for axis in (top, bottom):
        for bound, name in ((Pa, "$P_a$"), (Pb, "$P_b$")):
            axis.axvline(bound, color="grey", ls="--", lw=1)
        axis.grid(alpha=0.3)
    for bound, name in ((Pa, "$P_a$"), (Pb, "$P_b$")):   # inside the axes, clear of the title
        top.annotate(name, xy=(bound, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, -16), textcoords="offset points", ha="center", fontsize=12)
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    prices, source = load_prices()
    P0, capital = float(prices.iloc[0]), CONFIG["capital_usdc"]
    sigma = float(np.log(prices).diff().std(ddof=1) * math.sqrt(CONFIG["days_per_year"]))
    LOG.info("Source: %s | %d daily closes | entry %.0f -> last %.0f USDC/ETH | "
             "realised sigma %.1f%%", source, len(prices), P0, prices.iloc[-1], 100 * sigma)
    LOG.info("Price scenarios, range +/-%.0f%%, capital %.0f USDC\n%s",
             100 * CONFIG["reference_width"], capital,
             scenario_table(P0, CONFIG["reference_width"], capital).to_string(index=False))
    LOG.info("Range comparison: IL %% by shock, days out of range, breakeven APR\n%s",
             width_table(P0, prices, sigma, capital).to_string(index=False))
    figures: Path = CONFIG["figures_dir"]; figures.mkdir(parents=True, exist_ok=True)
    plot_il_curves(P0, capital, figures / "il_vs_price.png")
    plot_breakeven(P0, sigma, figures / "breakeven.png")
    plot_greeks(P0, capital, figures / "greeks.png")
    LOG.info("Figures written to %s", figures)

if __name__ == "__main__":
    main()
