# uniswap-v3-lp-risk

Quantitative risk analysis of a concentrated liquidity position on Uniswap v3, ETH/USDC pair.

## The question

Given an LP position in a price range $[P_a, P_b]$:

1. what is the **impermanent loss** relative to simply holding the same two assets?
2. what **minimum fee APR** makes the position worth taking versus buy & hold?

Concentrated liquidity is the defining feature of Uniswap v3: narrowing the range multiplies the fees earned per unit of capital, but it multiplies the loss just as hard when the price moves. This project quantifies that trade-off.

## Method

### Position accounting

With $P$ the spot price (USDC per ETH), $L$ the liquidity and $[P_a, P_b]$ the range, the token balances are:

$$
x = L\left(\frac{1}{\sqrt{P}} - \frac{1}{\sqrt{P_b}}\right), \qquad
y = L\left(\sqrt{P} - \sqrt{P_a}\right) \qquad \text{if } P_a \le P \le P_b
$$

$$
P < P_a: \quad x = L\left(\frac{1}{\sqrt{P_a}} - \frac{1}{\sqrt{P_b}}\right),\ y = 0
\qquad\qquad
P > P_b: \quad x = 0,\ y = L\left(\sqrt{P_b} - \sqrt{P_a}\right)
$$

Outside the range the position holds a single asset: all ETH below $P_a$, all USDC above $P_b$. That is the mechanism generating the loss — the pool sells ETH on the way up and buys it back on the way down, always on the wrong side of the move.

The position value in USDC is $V = xP + y$.

### Impermanent loss

Depositing $(x_0, y_0)$ at price $P_0$ pins down $L$; the same basket is the buy & hold benchmark:

$$
\text{IL}(P) = \frac{V_{\text{LP}}(P)}{V_{\text{hold}}(P)} - 1
= \frac{x(P)\,P + y(P)}{x_0 P + y_0} - 1
$$

The full-range Uniswap v2 case is recovered as $P_a \to 0$, $P_b \to \infty$ and has a closed form in $k = P/P_0$:

$$
\text{IL}_{v2}(k) = \frac{2\sqrt{k}}{1+k} - 1
$$

This identity is checked numerically by the tests at the bottom of `il_math.py`: with $P_a = 0$ and $P_b = \infty$ the two formulas agree to within $10^{-12}$.

### Breakeven fee APR

The price is assumed to follow a driftless geometric Brownian motion:

$$
P_T = P_0 \exp\left(-\tfrac{1}{2}\sigma^2 T + \sigma\sqrt{T}\,Z\right), \qquad Z \sim \mathcal{N}(0,1)
$$

so that $\mathbb{E}[P_T] = P_0$: no directional view, only volatility matters. The breakeven fee APR is the fee income that exactly cancels the expected loss:

$$
\text{APR}^{*} = -\frac{\mathbb{E}\left[\text{IL}(P_T)\right]}{T}, \qquad T = \frac{\text{horizon days}}{365}
$$

The concentrated payoff has no tractable closed-form integral, so $\mathbb{E}[\text{IL}]$ is evaluated by 128-node Gauss-Hermite quadrature — deterministic, no Monte Carlo sampling.

## Data

Daily ETH-USD closes for 2024 via `yfinance`, 366 observations. If the source is unreachable, `analysis.py` falls back to a synthetic GBM series at $\sigma = 60$% and says so in the log.

| | |
|---|---|
| Entry price (2024-01-01) | 2,352 USDC/ETH |
| Final price (2024-12-31) | 3,333 USDC/ETH |
| Period return | +41.7% |
| Low / high | 2,211 / 4,066 |
| Realised annualised volatility | 64.4% |
| Simulated capital | 100,000 USDC |

## Results

### Price scenarios, ±20% range

| scenario | price | LP value | hold value | IL % | IL USDC |
|---|---|---|---|---|---|
| −50% | 1,176 | 53,233 | 77,393 | −31.22% | −24,160 |
| −20% | 1,882 | 85,173 | 90,957 | −6.36% | −5,784 |
| 0% | 2,352 | 100,000 | 100,000 | 0.00% | 0 |
| +20% | 2,823 | 104,315 | 109,043 | −4.34% | −4,727 |
| +50% | 3,528 | 104,315 | 122,607 | −14.92% | −18,292 |

The LP value is **identical** at +20% and +50%: at or above $P_b$ the position is entirely USDC and stops participating in the upside. The cap on gains is structural, not an artefact.

### Range width comparison

| range | IL −50% | IL −20% | IL +20% | IL +50% | days out of range | breakeven fee APR |
|---|---|---|---|---|---|---|
| ±5% | −33.00% | −10.17% | −7.79% | −18.63% | 82.8% | 24.5% |
| ±20% | −31.22% | −6.36% | −4.34% | −14.92% | 61.2% | 22.0% |
| ±50% | −22.30% | −2.53% | −1.78% | −8.89% | 18.6% | 16.7% |
| full range (v2) | −5.72% | −0.62% | −0.41% | −2.02% | 0.0% | 5.0% |

![IL by range width](figures/il_vs_price.png)

Concentration multiplies the loss: on a +50% move the ±5% range loses 18.6% against 2.0% for the full range, **nine times as much**. The dashed red curve is the v2 benchmark and is the lower envelope of all the others.

### Breakeven fee APR

![Breakeven fee APR](figures/breakeven.png)

The breakeven rises as the range tightens, but **far less than the loss does**. Going from ±50% to ±5%, the IL at +50% worsens by a factor of 2.1 while the breakeven only moves from 16.7% to 24.5%. The reason is that at $\sigma = 64$% over a year the price almost certainly leaves any range: the three curves converge towards the same fate.

### The correction that changes the conclusion

The breakeven above assumes fees accrue on the full position value for the entire horizon. That is false: **a concentrated position earns nothing while out of range**. Correcting for the fraction of 2024 actually spent in range:

| range | days in range | nominal breakeven | APR required **while in range** |
|---|---|---|---|
| ±5% | 17.2% | 24.5% | **142.2%** |
| ±20% | 38.8% | 22.0% | **56.7%** |
| ±50% | 81.4% | 16.7% | **20.5%** |
| full range (v2) | 100.0% | 5.0% | **5.0%** |

This is the practical answer to the research question. A ±5% ETH/USDC range in 2024 would have needed to yield over 140% annualised on its active days alone just to match buy & hold. The ±50% range gets away with 20.5%, the full range with 5%.

The correction is approximate: it uses the realised out-of-range frequency of 2024 as an estimate of the future, and it ignores that fee density is higher near the spot price — a narrow range, while in range, earns more than its share of capital. The two effects pull in opposite directions and do not obviously cancel; measuring them requires pool volume data, outside the scope of this work.

## LIMITATIONS

The model does **not** capture:

- **Gas costs.** Opening, closing and collecting fees costs gas on Ethereum L1. On a small position it can exceed the impermanent loss itself. None of the figures above account for it.
- **Range rebalancing.** The analysis assumes a static position held for all of 2024. In practice an LP moves the range when the price runs away, crystallising the loss and paying gas every time. A ±5% range rebalanced weekly is a completely different product from the one modelled here.
- **Different fee tiers.** ETH/USDC exists across several pools (0.01%, 0.05%, 0.3%, 1%) with different depth and volume. The breakeven is expressed as a generic APR and is not tied to any specific tier.
- **MEV and transaction ordering.** Arbitrageurs realign the pool price to the market price and capture part of the value; the LP is on the losing side of every realignment (*loss-versus-rebalancing*). The classic IL model used here understates that cost.
- **Other LPs' liquidity concentration.** Fees earned depend on your share of the liquidity active at that tick, which changes continuously. Here the fee APR is an exogenous parameter, not an output of the model.
- **Path dependency.** The out-of-range percentages come from **a single** price path, 2024, a year in which ETH returned +41.7%. A sideways year would produce numbers far more favourable to tight ranges. This is not a distribution, it is a sample of size one.
- **GBM as a price model.** Constant volatility, no jumps, no fat tails, no clustering. The expected breakeven is computed under this assumption and is optimistic to the extent that real ETH returns are leptokurtic.
- **Fee reinvestment and compounding.** The breakeven is a simple annualised return, with no compounding.

## Layout

```
uniswap-v3-lp-risk/
├── il_math.py     closed-form formulas, pure functions, no I/O
├── analysis.py    data, scenarios, tables, figures
├── README.md
└── figures/       il_vs_price.png, breakeven.png
```

## Running it

Self-consistency tests of the math, including the full-range limit:

```bash
python il_math.py
```

Full analysis: tables to the terminal, figures regenerated:

```bash
python analysis.py
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `yfinance` (optional — without it the synthetic fallback is used).
