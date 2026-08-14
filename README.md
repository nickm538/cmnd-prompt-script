# Financial Live Premium Scanner

Command-line Python scanner for live U.S. stock and ETF analysis. The active
entry point is:

```bash
new_stock_scanner_pipeline_claude_opus_41426.py
```

The scanner dynamically discovers the live universe, pulls current market data
and headlines, checks macro regime context against today's world events,
applies execution guards and hard buy rules, ranks survivors, evaluates
fundamentals/options, and writes auditable outputs. It does not use preset
ticker baskets, watchlists, yesterday's CSV, or mock data. News and the
earnings calendar annotate names that already survived; they never choose
the universe.

## Local setup

Use Python 3.12 or another current Python 3 release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## API configuration

The script supports these environment variable overrides:

- `MASSIVE_API_KEY` (required for universe discovery)
- `MBOUM_API_KEY` (primary OHLCV and fundamentals source when credits remain)
- `MBOUM_OPTIONS_KEY` (primary options chains when present)
- `TWELVEDATA_API_KEY` (fallback OHLCV and fundamentals)
- `FINNHUB_API_KEY` (fallback fundamentals; OHLCV if the plan includes candles)
- `ALPHAVANTAGE_API_KEY`

MBOUM stays the primary market-data source. If the MBOUM plan is out of
credits, unauthorized, or returning empty history, the engine trips a
process-local circuit and continues the same scan through Massive, then
TwelveData, then Finnhub, then Yahoo v8 / yfinance. Restored MBOUM credits
are used first again on the next run. No bars or fundamentals are fabricated.

Environment variables and GitHub Actions secrets override the committed
fallback keys for Massive, TwelveData, and Finnhub. If those secrets are
empty, the engine uses the keys checked in on this branch so the scan can
still run. MBOUM keys are not committed and still come from secrets.

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

The scanner calls live financial data APIs and can take several minutes. A full
live run on the current universe has been verified end-to-end at about 13.5
minutes, but runtime depends on API latency and universe size.

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

The JSON report includes macro regime snapshots, pipeline funnel counts,
feature importances, and an attestation that live data sources were used.

## Operational notes

- Run before the market opens; the scanner uses the latest fully closed daily
  bars.
- If run during regular trading hours, intraday partial bars are trimmed before
  screening so volume and freshness checks do not use incomplete data.
- A day with zero qualifying buys is valid behavior. In that case, the scanner
  writes a near-miss report instead of fabricating candidates.
- Options candidates require live bid/ask quotes and liquidity checks; otherwise
  the equity candidate remains equity-only.
