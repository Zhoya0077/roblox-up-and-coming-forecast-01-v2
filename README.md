# Roblox Up-and-Coming Forecast — v2 (transition-matrix model)

Second entry for `roblox-up-and-coming-forecast-01`. A different model from
the v1 Monte Carlo entry: an empirical rank-transition (Markov) model over
rank states, estimated directly from the history's read pairs.

## How it works

- The chart is a Markov chain over 21 states: exact ranks 1-10, fine rank
  bands beyond (11-12, 13-15, ..., 86+), plus one ABSENT state for games
  that left the chart.
- One transition matrix T(dt) is estimated per lag bucket from ALL read
  pairs in the history, each row shrunk toward a diffusion prior fitted to
  the same pairs (empirical mean/std of rank change and survival vs lag).
- For each game and each scored slot, the row of T(h) for the game's
  last-known state — drift-shifted by shrunk recent rank velocity — gives
  the place distribution; averaging over the 72 slots gives the forecast.
- Games never seen before are covered by an empirical newcomer-share
  reserve (never-seen share curve vs horizon, scaled 1.4), left as unnamed
  share rather than named rows.
- Place probabilities get a mild calibration power (1.15, renormalized per
  place); games are named only while their probability beats twice the
  unnamed per-game value, and naming stops at the reserve cap.

## Why this beats v1

The empirical matrices capture the chart's asymmetric, rank-dependent
stickiness (rank 1 stays ~68% at short lags; rank-change spread grows then
saturates; games drift down over long lags) instead of imposing symmetric
mean reversion. Backtested with the exact RULES scorer on the same held-out
cuts as v1:

- Pooled cuts: v2 -38.52 vs v1 -39.66 (+1.14), v2 wins all 5 cuts.
- Practice window: v2 -36.65 vs v1 -37.49 and the house baseline -43.51.

Tuning used held-out cuts only; the practice window was validation.

## Files

- `main.py` — the forecaster. Python 3.13, stdlib only, no network, fully
  deterministic (no RNG). Reads the window and place count from task.json,
  never hardcodes dates. Handles null details (archive reads) gracefully.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Writes `forecast.json` with exactly two keys: `probabilities` (rows of
universe_id / place / p, at most 100 rows per place, each place sums to at
most 1) and `top10` (exactly 10 distinct universe_ids).

## Determinism

Two runs on the same input produce byte-identical output. Runtime is about
0.3 seconds on the practice input (limit 300s); dense histories are kept
fast by subsampling read pairs above 6,000.
