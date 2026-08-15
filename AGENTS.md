# AGENTS.md

## Cursor Cloud specific instructions

This repository is a single Python 3.12 command-line product: a live U.S. stock/ETF
scanner. There is no web server or frontend. The two things to run are the
regression tests and the scanner CLI. See `README.md` for the full product
description and API-key details.

### Services / commands

- Regression tests: `python3 -m unittest -v test_stock_scanner_pipeline.py`
  (fast, fully mocked, no network). Run these before the scanner.
- Live scanner (the product): `python3 new_stock_scanner_pipeline_claude_opus_41426.py`
  Optional env: `SCAN_BUDGET_MINUTES` (wall-clock budget, default `100`) and
  `LOG_LEVEL` (`INFO`/`DEBUG`).
- There is no configured linter (no ruff/flake8/pylint config, and CI only runs
  the tests + scanner). Use `python3 -m py_compile <file>` for a syntax check.

### Non-obvious caveats

- Dependencies are installed into the **system** Python (not a venv). `python3 -m
  venv` fails here unless `python3.12-venv` is installed; the update script uses
  system `pip3` instead, so just call `python3`/`pip3` directly. Do not expect a
  `.venv/` to exist.
- The scanner needs no secrets to run: it ships committed fallback API keys for
  Massive, TwelveData, and Finnhub. `MBOUM_API_KEY` / `MBOUM_OPTIONS_KEY` are the
  only keys that are never committed; without them MBOUM (the preferred market-data
  source) is skipped and the engine falls back to Massive -> TwelveData -> Finnhub
  -> Yahoo/yfinance for the same run. Set those env vars to prefer MBOUM.
- Some fallback providers may be rate/plan limited on any given run (e.g. Finnhub
  candles returning HTTP 403). This is expected: the engine trips a process-local
  circuit and continues down the fallback chain rather than failing the scan.
- A full live run walks the entire live universe (~10k+ tickers) and can take on
  the order of ~13 minutes depending on API latency. It is network-heavy by
  design (no preset ticker baskets or mock data).
- Outputs are git-ignored and written to `scan_pipeline.log` and
  `scan_results/scan_<timestamp>.{csv,json}` (or `near_misses_<timestamp>.*` when
  zero tickers pass every hard rule, which is a valid outcome).
