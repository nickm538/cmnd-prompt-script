# Financial Live Premium Scanner

Command-line Python scanner for live U.S. stock and ETF analysis. The active
entry point is:

```bash
new_stock_scanner_pipeline_claude_opus_41426.py
```

The scanner dynamically discovers the live universe (Massive first, then
official NASDAQ Trader listing files, then Finnhub -- never a preset
basket), pulls current market data and headlines, checks macro regime
context against today's world events and live market status (VIX, yields,
USD, gold, oil, session open/holiday), applies execution guards and hard
buy rules, ranks survivors, evaluates fundamentals/options, and writes
auditable outputs. It does not use preset ticker baskets, watchlists,
yesterday's CSV, mock data, or a pre-selected benchmark ETF. News and the
earnings calendar annotate names that already survived; they never choose
the universe. Relative strength is measured against the live S&P 500
index fetched that run.

The output limit is seven; it is not a quota. If no ticker passes every hard
rule, the scanner abstains and writes a non-actionable near-miss report. It
never promotes an 8/10 or 9/10 setup to `BUY` just to fill the table.

## Local setup

Use Python 3.12 or another current Python 3 release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experimental informational LSTM is disabled by default and never
participates in ranking. To inspect it locally:

```bash
python -m pip install -r requirements-optional.txt
ENABLE_EXPERIMENTAL_LSTM=1 python new_stock_scanner_pipeline_claude_opus_41426.py
```

## API configuration

The script supports these environment variable overrides:

- `MASSIVE_API_KEY` (preferred for universe discovery and OHLCV; official
  NASDAQ listing files are the no-key universe fallback)
- `MBOUM_API_KEY` (primary OHLCV and fundamentals source when credits remain)
- `MBOUM_OPTIONS_KEY` (primary options chains when present)
- `TWELVEDATA_API_KEY` (fallback OHLCV and fundamentals)
- `FINNHUB_API_KEY` (fallback fundamentals; OHLCV if the plan includes candles)
- `ALPHAVANTAGE_API_KEY`

MBOUM stays the primary market-data source. If the MBOUM plan is out of
credits, unauthorized, or repeatedly fails at the provider level, the engine
trips a process-local circuit and continues the same scan through Massive, then
TwelveData, then Finnhub, then Yahoo v8 / yfinance. Restored MBOUM credits
are used first again on the next run. No bars or fundamentals are fabricated.
Ticker-specific no-data responses do not trip a provider for the rest of the
universe.

Massive, TwelveData, and Finnhub API keys must be provided via environment
variables or GitHub Actions secrets. Each provider is skipped in the fallback
chain if its key is absent. MBOUM keys (both tiers) are also read from
environment variables or GitHub Actions secrets only.

Scheduled-event context prefers Finnhub's economic calendar (CPI, NFP,
FOMC). If that endpoint is plan-blocked, the engine loads the official
Federal Reserve calendar for FOMC/Beige Book/Fed releases and keeps
position sizing conservative because BLS/BEA prints are still unverified.

## Run locally

Run regression checks first:

```bash
source .venv/bin/activate
python -m unittest -v
```

Run the live scanner:

```bash
source .venv/bin/activate
python new_stock_scanner_pipeline_claude_opus_41426.py
```

The scanner calls live financial data APIs and can take several minutes.
Runtime depends on universe size, provider latency/rate limits, and how far the
fallback chain has to travel. The in-process budget reserves time to write an
auditable status artifact before the GitHub Actions timeout.

## GitHub scheduled run

The checked-in workflow is:

```bash
.github/workflows/run-scanner.yml
```

It runs the scanner on weekdays at 08:45 ET using GitHub Actions. Because GitHub
cron uses UTC, the workflow has both daylight-saving and standard-time triggers
and skips the duplicate off-hour run with an ET-hour guard.

Each scheduled run:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs `requirements.txt`.
4. Runs `python -m unittest -v`.
5. Runs `python new_stock_scanner_pipeline_claude_opus_41426.py`.
6. Uploads scan logs/results as workflow artifacts.

The workflow timeout is 120 minutes to allow for live data-provider latency.

## Outputs

Runtime outputs are ignored by git and written to:

- `scan_pipeline.log`
- `scan_results/scan_<timestamp>.csv`
- `scan_results/scan_<timestamp>.json`
- `scan_results/near_misses_<timestamp>.csv` when no ticker passes all hard
  rules
- `scan_results/near_misses_<timestamp>.json` when no ticker passes all hard
  rules
- `scan_results/status_<timestamp>.json` when a run stops incomplete

The JSON report includes macro regime snapshots, headline/economic/earnings
feed status, pipeline funnel counts, provider provenance, feature importances,
date-purged walk-forward diagnostics, calibration metrics, and explicit model
risk limitations.

## Model-risk boundaries

- Tree-model labels use the next session's open as the feasible entry and the
  close 20 sessions later as the exit. Validation is grouped by trading date
  with a 20-session purge, and out-of-sample probabilities are calibrated. A
  model that does not beat the chronological base-rate forecast abstains at
  `0.5`.
- The broad training pool is not selected on today's return. It is still built
  from securities listed today because the configured providers do not supply
  a point-in-time delisted universe. The report therefore marks survivorship
  bias as uncontrolled.
- `setup_quality_score` (and its backward-compatible
  `overall_confidence_score` alias) is a transparent heuristic, not a win
  probability. No scanner can guarantee predictive accuracy or eliminate
  earnings-gap, liquidity, macro, or geopolitical risk.
- An earnings date alone is treated as event risk, not a bullish catalyst.
  Candidates with earnings inside the strategy holding window are held at
  `WAIT`, and options spanning that event are rejected unless a separately
  validated event model is added in the future.

## Operational notes

- Run before the market opens; the scanner uses the latest fully closed daily
  bars.
- The official XNYS calendar (including holidays, special closures, and
  half-days) determines the latest vendor-finalized session. Same-day partial
  bars are removed even if they appear outside regular trading hours.
- A day with zero qualifying buys is valid behavior. In that case, the scanner
  writes a near-miss report instead of fabricating candidates.
- Options candidates require live bid/ask quotes and liquidity checks; otherwise
  the equity candidate remains equity-only.
