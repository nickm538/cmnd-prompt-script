# Financial Live Premium Scanner — 6.1.0

A portable Python 3.12 command-line scanner. The product is one script and its
console, CSV, and JSON outputs. There is no server, frontend, hosting service,
or broker execution.

```bash
python new_stock_scanner_pipeline_claude_opus_41426.py
```

The scanner discovers current U.S. exchange listings across sectors, fetches
live provider histories and context, removes unfinished daily bars using the
official XNYS calendar, applies liquidity/execution guards and ten technical
rules, fits chronological ML models, evaluates fundamentals and options, then
writes an auditable report. Prior scan outputs never choose the universe.
Relative strength uses the live S&P 500 index, rather than a preset ETF basket.

**Its mandate is liquid bullish breakouts over about 20 trading sessions.**
Broad discovery does not mean coverage of every investment style: short sales,
mean reversion, early reversals, illiquid stocks, sub-$5 stocks, and short-history
IPOs are not actionable strategies in this version. Reports expose exclusions,
unavailable symbols, overlapping rule failures, and alternative setup archetypes.
Seven results is a maximum, not a quota. Near misses are never promoted to BUY.

## Setup and credentials

Use Python 3.12 and the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v -p 'test_*.py'
python new_stock_scanner_pipeline_claude_opus_41426.py
```

Store keys in local environment variables or **repository Actions secrets**:

| Secret | Purpose |
| --- | --- |
| `MASSIVE_API_KEY` | Universe discovery, history fallback, news, options |
| `MBOUM_API_KEY` | Preferred history and fundamentals when credits remain |
| `MBOUM_OPTIONS_KEY` | Preferred options chains |
| `TWELVEDATA_API_KEY` | History and fundamentals fallback |
| `FINNHUB_API_KEY` | Listings, history if entitled, fundamentals, dated insiders, news and calendars |
| `ALPHAVANTAGE_API_KEY` | Reserved configuration; no active Alpha Vantage fetch path |

There are no working fallback keys in source. **Rotate the Massive, TwelveData,
and Finnhub keys that were previously public in git history** and save replacements
only as secrets. Deleting current literals does not revoke exposed credentials.
Logs redact configured keys and credential query parameters.

Discovery prefers Massive, then both official NASDAQ Trader listing files, then
Finnhub. Incomplete pagination or a single successful listing file cannot
masquerade as a complete universe. Common `.A`/`.B` share classes and ADR classes
use provider-specific symbol aliases. History prefers MBOUM, then Massive,
TwelveData, Finnhub, and public Yahoo/yfinance. Provider-level plan, credit, or
authorization failures trip a process-local circuit; ticker-specific misses do
not disable a provider for other securities. Circuits reset on the next run.

Without paid keys, public discovery and histories remain available, but context
and fundamentals may be incomplete. Missing quick-screen quotes are retained for
the full history cascade. If fewer than 80% of retained symbols have sufficient,
fresh history, the scanner writes an **incomplete** report instead of presenting
a partial market scan as complete. Even higher coverage can miss winners; every
unavailable ticker is exported. Provider adjustment policies are disclosed and
are not assumed identical.

## Run in GitHub Actions

Open **Actions → Run Stock Scanner → Run workflow**. Manual runs start immediately
with the selected branch. Weekday scheduling at 08:45 ET remains enabled; the two
UTC triggers handle daylight saving and the off-season trigger is skipped.

The workflow uses Python 3.12, installs pinned requirements, runs all regression
files by default, executes the same single script, writes an Actions summary, and
uploads `stock-scan-results` with the log and CSV/JSON reports. Artifacts expire
after 14 days. Scheduled and manual scans share a concurrency group so their
provider requests do not overlap. Regression CI also runs for pull requests and
changes to `main` or `codex/**` branches.

`SCAN_BUDGET_MINUTES` defaults to 100. Workflow input must be positive, finite,
and at most 110 because the job timeout is 120 minutes. The scanner bounds
queued work and emits diagnostics on interruption. Active requests still have
finite provider timeouts, so this is not an instantaneous hard kill deadline.
`LOG_LEVEL` accepts `INFO` or `DEBUG`. Live runtime depends on coverage, provider
latency, rate limits, and entitlements; recent audited runs took around 18–21 minutes.

## Scoring and action gates

The equity setup score uses these **heuristic, unoptimized** weights:

| Component | Weight |
| --- | ---: |
| Coded investor-style panel | 34% |
| Validated ML probability lift above its target base rate | 26% |
| Ten-rule eligibility | 14% |
| Additional trend/momentum measurements | 8% |
| Verified, dated open-market insider evidence | 8% |
| Liquidity | 10% |

The report exports each contribution and all exhaustion, risk-flag, binary-event,
and fundamental-data penalties. The panel contains correlated coded heuristics,
not five independent investors. Its consensus is a gate, not independent proof.
Missing insider evidence earns no insider bonus. Short interest and price/volume
“hype” remain context; they do not add another equity-score confirmation.
Contract quality is scored separately so option availability cannot improve an
equity's quality. `setup_quality_score` and legacy `overall_confidence_score` are
quality scores, **not probabilities of profit or expected returns**.

BUY eligibility requires every hard rule, panel qualification, usable ML evidence
with positive lift, and event/data checks. Unknown upcoming issuer earnings dates
produce `WAIT: CALENDAR GAP`; verified funds do not require an issuer earnings
date. Earnings in the holding window, binary events, daily-reset leveraged funds,
or failed fundamentals cannot be bypassed by a high score. A legitimate scan can
produce only WATCH/WAIT results or zero strict candidates.

## What ML and options mean

The tree-model target is **more than +5% from the next XNYS session's open to the
close 20 sessions after the signal**. It is not the probability of any positive
return, reaching a displayed target, surviving an ATR stop, or profiting on a call.
Missing required entry/exit bars have unknown labels and are dropped. All symbols
on a date stay in the same expanding validation fold, with a 20-session purge.
Calibration and evaluation use separate date blocks with their own purge.
Evaluation includes base rates, Brier skill, AUC, log loss, and reliability bins.
The final estimator configuration matches its out-of-fold configuration.

Model skill also requires a positive lower 95% percentile bound from 400
resamples of whole-session loss totals in circular 20-session blocks. At least
40 held-out sessions are required. This is an approximate stability filter;
longer dependence, regime changes and selection effects remain unvalidated.
It does not establish the actual trade policy's net returns.

Failed/unskilled models cannot dilute a usable model through a neutral 0.5
placeholder. The actual active raw ensemble is independently calibrated and
evaluated before its probabilities authorize actions. Candidate
`ml_probability_lift` is **probability minus historical target base rate**, not a
ratio. Broad-panel model skill has not established calibration within the strict
ten-rule subset or net trading profitability. Histories from today's listings
still contain survivorship bias; delisted point-in-time constituents are absent.

Options require finite executable bid/ask quotes, identified quote timing and
delay, a synchronized underlying quote, verified standard 100-share deliverables,
liquidity/spread checks, and expiry after the actual holding window plus a buffer.
Ask-based breakeven and maximum premium loss are exported. Last trade is not a
current quote. Closed-session references remain WATCH rather than actionable
calls. An unavailable provider field is reported as unknown, not synthesized.
Any fallback estimated delta is labeled as an approximate European-model Greek.
It does not model American exercise, dividends fully, or option profitability.

The experimental LSTM is disabled by default and never participates in ranking.
It remains unvalidated informational output. Optional local inspection:

```bash
python -m pip install -r requirements-optional.txt
ENABLE_EXPERIMENTAL_LSTM=1 python new_stock_scanner_pipeline_claude_opus_41426.py
```

## Outputs and risk interpretation

Runtime files are ignored by git:

- `scan_pipeline.log`
- `scan_results/scan_<timestamp>.csv` and `.json` for strict survivors
- `scan_results/near_misses_<timestamp>.csv` and `.json` for valid no-action scans
- `scan_results/status_<timestamp>.json` for incomplete scans

Reports include universe/history coverage, rejected/unavailable symbols, funnel
counts, exact model specifications, chronological validation, component scores,
fundamental/insider provenance, macro/event feed status, and options rejections.
JSON is strict: missing or infinite numerical values are exported as `null`.
The legacy JSON key `top_25` remains for compatibility even though the display
limit is seven.

Stops and 2.5R targets are **planning levels at the last finalized signal close**,
not price forecasts. A standalone size per $10,000 limits planned stop risk to at
most 1% scaled by macro conditions and notional exposure to 20% per name. Refresh
prices and sizes at actual entry. Simultaneously applying every standalone size
does not form a validated portfolio allocation. Gaps can exceed intended stop
losses. The helper `SellMonitor` is not called by the CLI and does not monitor a
broker account or execute trades.

News is a bounded provider sample. General macro headlines exclude repeated
shareholder-lawyer solicitations while company-specific legal risk remains
context. Price/volume hype is not observed social sentiment. A Federal Reserve
calendar fallback covers Fed events, not a verified BLS/BEA release calendar;
those gaps keep sizing conservative. No report is a complete census of current
events or a guarantee of future returns.

See [the detailed audit](SCANNER_AUDIT_2026-09-29.md) for defects, research,
validation evidence, and the remaining requirements for credible strategy testing.
