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

## Troubleshooting

### `ModuleNotFoundError: No module named 'websockets.asyncio'`

The `massive` SDK (>= 2.6) imports `websockets.asyncio`, which only exists in
`websockets` >= 13.0. If an older `websockets` (often pulled in by another
package such as `yfinance`) is already installed in your Python, `pip install
massive` will not always upgrade it, and `from massive import RESTClient`
fails on import.

Fix it by upgrading `websockets` in the same Python interpreter you used to
run the script:

```bash
python -m pip install --upgrade "websockets>=14.0" "massive>=2.6"
```

On Windows, run that command from the same Command Prompt / PowerShell where
`python ...stock_scanner_pipeline_claude_opus_41426.py` failed, so the upgrade
lands in the same `Python312\Lib\site-packages` directory shown in the
traceback. After the upgrade, verify with:

```bash
python -c "from websockets.asyncio.client import connect; print('ok')"
python -c "from massive import RESTClient; print('ok')"
```

Both should print `ok`.
