# Financial Live Premium Scanner

Command-line Python scanner for live market-data analysis.

## Setup

Use Python 3.12 or another current Python 3 release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scanner supports these environment variable overrides:

- `MASSIVE_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `MBOUM_API_KEY`

If they are not set, the script falls back to the keys already embedded in the
source file.

## Run

```bash
source .venv/bin/activate
python stock_scanner_pipeline_claude_opus_41426.py
```

The script calls live financial data APIs and can take several minutes. Runtime
outputs are written to:

- `scan_pipeline.log`
- `scan_results/`
