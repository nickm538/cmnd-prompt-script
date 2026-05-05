# Financial Live Premium Scanner

Command-line Python scanner for live U.S. stock and ETF analysis. The active
entry point is:

```bash
new_stock_scanner_pipeline_claude_opus_41426.py
```

The scanner dynamically discovers the live universe, pulls current market data,
checks macro regime context, applies execution guards and hard buy rules, ranks
survivors, evaluates fundamentals/options, and writes auditable outputs. It does
not use preset ticker baskets or mock data.

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

- `MASSIVE_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `MBOUM_API_KEY`
- `MBOUM_OPTIONS_KEY`

If they are not set, the scanner falls back to the keys embedded in the source
file. For GitHub Actions, configure these as repository secrets so scheduled
runs do not depend on local machine state.

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
