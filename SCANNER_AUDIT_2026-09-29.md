# Scanner integrity and methodology audit — September 29, 2026

Engine 6.1.0 fixes defects that could silently omit securities, distort indicators,
overstate model validity, or authorize trades from insufficient event/quote data.
It preserves the portable script and existing manual/scheduled Actions workflow.
There is no host, UI, Railway service, broker connection, or trading deployment.

The implemented weights are transparent heuristics. Neither this review nor a
successful live scan establishes maximum returns, optimal weights, or profitable
trading. The most consequential remaining gap is validation of the actual
selected strategy after execution costs, rather than only a broad-panel classifier.

## Scope and baseline evidence

Reviewed the original 8,649-line script by subsystem, its 64-test suite, both
requirements files, Actions workflow, summary script, README, agent instructions,
ignore rules, and editor configuration. Traced discovery, provider circuits,
session handling, technical calculations, admission rules, ML labels/validation,
fundamental mappings, panel scoring, options selection, risk planning, exits,
reporting, and exception paths together. Reviewed the complete JSON and relevant
failure/coverage entries from the September 28 live run and its CSV/log artifacts.

Baseline commit: `2cebe915284dfbfcc9b68cffe02fdf6b07a0b3d3`.
[Baseline run](https://github.com/nickm538/cmnd-prompt-script/actions/runs/36454456954).

| September 28 baseline observation | Evidence / implication |
| --- | --- |
| Runtime | 668.7 seconds, about 11.1 minutes |
| Discovered listings | 10,811 |
| Finalized histories | 5,086; the old report did not explain the entire missing-history denominator |
| Passed execution guards | 1,563 |
| Passed all ten rules | Two: CERT and PFBC |
| Final actions | CERT WATCH; PFBC WAIT: EARNINGS |
| Broad evaluation target base rate | About 20.4%, not 50% |
| Reported XGB / RF AUC | About 0.621 / 0.659; these are classification metrics, not realized trading returns |
| News sample | Dominated by shareholder-lawyer solicitations; macro relevance was weak |
| Context gaps | Finnhub unavailable and Fed-only calendar coverage did not verify BLS/BEA events |
| Original offline checks | 64 tests passed with one optional Torch test skipped, despite untested runtime defects |

The baseline and new live scan occur on different days and may use different
providers. Changes in candidate counts or scores are not a controlled performance
comparison. The original calibration defects also prevent comparing old and new
probabilities as if their validity were equivalent.

## How the single script works

| Stage | Inputs, output, and dependency relationships |
| --- | --- |
| Runtime configuration | Environment-only credentials, redacted logs, pinned Python dependencies, finite scan budget |
| Macro/world context | Current index/risk gauges, bounded news, earnings/economic calendars and market status; gaps affect risk sizing |
| Discovery and histories | Current listings → quick price/liquidity screen → bounded full provider cascade → normalized, finalized XNYS daily bars |
| Technical and execution checks | Histories → indicators → price/history/liquidity/freshness/anti-chase guards; failures are counted |
| Strict selection | One shared ten-rule evaluator drives both admission and reported diagnostics; other archetypes remain non-actionable |
| ML ranking | Broad liquid historical panel → exact forward labels → date-grouped purged folds → separate calibration/evaluation → active ensemble |
| Fundamentals and style panel | Survivors → provider fields, dated insider evidence, verified instrument type, earnings risk, date-aligned benchmark comparisons |
| Options and risk plan | Survivors → verified contract/quote eligibility and separate contract score → standalone equity stop/size plan |
| Output verification | Finite values, duplicate removal, final risk flags and reconciled score → actions, strict JSON, CSV, console, Actions summary |

`SellMonitor` is a helper, not part of the CLI's execution path. The scanner does
not read positions from a broker, monitor orders, or execute its stops. Optional
LSTM output is disabled and excluded from ranking. CI and documentation do not
introduce another production application or a second scanner entry point.

## Defects corrected

| Area | Defect / misleading behavior | Implemented correction |
| --- | --- | --- |
| Scalar arithmetic | `safe_div` had no function declaration; its body was stranded under another helper | Restored the function and tested real callers |
| Credentials | Working fallback keys were public; shared-session authentication could reach an unrelated listing host | Removed fallback literals, scoped authentication, and redacted configured keys/query parameters |
| Discovery | Incomplete pages or one listing file could become a seemingly complete universe | Reject incomplete discovery and continue to a complete alternative |
| Symbol coverage | Valid common/ADR share classes and IEX listings could be omitted | Added supported listing types and provider-specific class aliases; preserved verified security-type exclusions |
| Quick quotes | An unavailable quick quote could eliminate a valid ticker | Retain unverified names for the full history cascade; export quote misses and price/liquidity exclusions |
| Provider resilience | Ticker misses could disable a whole provider; plan-history constraints stopped usable shorter histories | Separate symbol/provider failures; retry Massive's entitled shorter window; retry stale/insufficient histories downstream |
| History integrity | UTC candle labels, mixed multi-index responses, malformed OHLC, and unfinished bars could corrupt signals | Correct dates/symbol selection; reject nonfinite or inconsistent bars; normalize duplicates/order; apply XNYS finalization |
| Incomplete scanning | A numerical minimum history count could hide weak market coverage | Export attempted/loaded/unavailable lists and denominator; stop below 80% retained-history coverage |
| Runtime work | Unbounded submitted futures and invalid budgets undermined predictable completion | Bound queued futures, cancel pending work on interruption, and validate a finite positive budget |
| False quota exhaustion | Valid MBOUM fields containing `Month`, including security names, fundamentals and option quotes, disabled whole providers | Classify explicit error envelopes and status codes; successful data fields cannot trip credit circuits |
| Technical formulas | RSI/ATR/ADX initialization deviated from Wilder; flat RSI and missing values could mislead | Arithmetic-seeded Wilder smoothing, flat RSI 50, explicit missingness and OHLC integrity |
| Strict admission | Separate reporting and admission evaluators disagreed, including volume/VWAP edge cases | One evaluator; complete rule outcomes, failure counts, and setup archetypes |
| Fundamentals | Unit errors, stale quality flags and optional endpoint failure could distort valuation/growth or discard good base fields | Provider-specific percent/ratio/field normalization, quality recovery, endpoint-scoped circuits and coverage provenance |
| Instruments | Funds could masquerade as ordinary issuer equities | Verified provider instrument type and explicit daily-reset leveraged/inverse fund risk |
| Insider evidence | Undated, duplicate or non-market transactions could add bullish weight | Dated, classified, deduplicated open-market evidence; unknown aggregates earn no equity bonus |
| Benchmark panel | Mismatched date windows, ties and stale percentile caches distorted relative strength | Exact-date/horizon comparisons, neutral tie handling, cache reset and excess-return exports |
| Panel claims | Analyst count, daily-sign Kelly sizing and fabricated upside at new highs suggested unsupported evidence | Removed those rewards; new-high asymmetry stays neutral without observed resistance; panel labeled heuristic |
| ML labels | Missing sessions could stretch a nominal 20-session outcome; invalid tails could become negatives | Exact XNYS next-open/session+20-close target; missing required prices become unknown labels |
| ML folds | Row splits could mix dates or leak overlapping labels | Whole-date expanding folds with a 20-session purge; no row-based fallback |
| Calibration | Calibration/evaluation overlap and changed final model configuration invalidated probability claims | Purged date-block separation, same OOF/final estimators, reliability and base-rate skill diagnostics |
| Fragile model skill | Seeded random labels could clear a point-estimate skill gate through a tiny chance improvement | Require positive lower 95% Brier-skill percentile bound from 400 circular 20-session date-block resamples; at least 40 evaluation sessions |
| Ensemble | Failed 0.5 placeholders diluted valid models; averaging calibrated outputs was mislabeled as calibrated | Exclude failed models; independently calibrate/evaluate the actual aligned raw active ensemble |
| Equity scoring | Option availability and repeated hype/squeeze momentum rewarded correlated evidence | Separate contract quality; export equity components/weights; compare ML to its own target base rate |
| Action labels | Strong scores could survive missing issuer calendar or unvalidated model context | Explicit WAIT/WATCH gates for unverified earnings, daily-reset funds and invalid/no-lift ML; missing current features cannot inherit ensemble validation |
| Options execution | Last prices, missing quote timing/deliverables, or stale underlying prices could look executable | Require bid/ask, timing/delay, synchronized underlying and standard deliverable; unknown metadata rejects; closed-session references WATCH |
| Options horizon | Calendar-day expiry could precede the trading-session holding window | Expiry after 20 XNYS sessions plus buffer; ask-based breakeven and 100-share premium loss |
| Exits | Horizon and peak-memory errors; missing data could produce a sale request | Trading-session horizon, retained peak reference and explicit review for absent data |
| Equity sizing | Tight support could create a nominal position larger than the account | Standalone 20% notional cap and at most 1% macro-scaled planned stop risk per $10,000 |
| Reporting | Nonstandard NaN/Infinity JSON, mutation on repeat exports, empty no-action output and stale descriptions | Strict null-safe JSON, repeatable exports, readable empty reports, accurate model/risk/strategy metadata |
| Workflow | New tests were not reliably discovered; unsafe input budgets and obsolete fallback-key instructions | All-file regression discovery, budget validation, maintained action releases, secret-presence-only checks |

## Scoring policy and its practical meaning

| Equity component | Weight | Interpretation |
| --- | ---: | --- |
| Coded style panel | 34% | Correlated technical/fundamental heuristics, with field coverage |
| Validated ML lift | 26% | Calibrated target probability minus its historical target base rate |
| Eligibility | 14% | Ten-rule gate; constant among strict survivors, not independent alpha |
| Additional momentum | 8% | Heuristic; overlaps panel/model features and is not independent evidence |
| Dated insider evidence | 8% | Bonus only for supported open-market transactions |
| Liquidity | 10% | Tradability proxy, not expected return |

Contributions sum to 100% before disclosed penalties and clipping. The insider
bonus is zero when evidence is unavailable. ML with no usable validation receives
a neutral score component but cannot authorize a BUY. Short interest and hype
remain visible context. Estimated maximum option loss is separate from equity
stop sizing. Verification-time flags are included before final ranking.

The weights were not fitted to maximize historical returns. They reduce specific
interpretation and double-counting problems; they remain policy choices. Promoting
more names, relaxing rules, or adding more correlated factors without testing
would not demonstrate better expected returns. The existing ten-rule strategy is
preserved while its narrow mandate and exclusions are made explicit.

## Research and why it changed the implementation

The following are primary papers or official technical/market sources. Their
results inform engineering choices; none validates this exact scanner.

- [Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning](https://dachxiu.chicagobooth.edu/download/ML.pdf): nonlinear predictors and out-of-sample discipline motivate consistent estimators and honest historical evaluation. Their asset-pricing task/horizon is different from this scanner's 20-session event and cannot supply its optimal weights.
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): calibration requires predictions independent of model training; evaluation must also remain separate. Reliability bins accompany Brier/AUC because a single metric is not proof of calibration.
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): chronological splitting and a gap are useful concepts, but a row gap on a multi-stock panel is not a trading-date purge. This implementation groups all rows by session.
- [Politis and Romano, The Stationary Bootstrap](https://statistics.stanford.edu/technical-reports/stationary-bootstrap) and [official circular block bootstrap documentation](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html): dependent observations require dependence-aware resampling. The scanner implements a fixed-length circular date-block filter in NumPy, without adding another runtime package. Twenty sessions follow the outcome horizon; this block length is a disclosed policy, not an optimal estimate. Percentile ranges are approximate and do not repair regime changes or historical model selection.
- [Daniel and Moskowitz, Momentum Crashes](https://www.nber.org/papers/w20439): momentum can suffer persistent losses during stress/rebound states. Retain macro/anti-chase context, but do not claim that heuristic risk controls eliminate those losses.
- [Baker and Wurgler, Investor Sentiment in the Stock Market](https://www.nber.org/papers/w13189): sentiment can affect prices, but measurement matters. The current script observes price/volume activity, not a direct social sentiment series, so it must not label hype as measured social evidence.
- [NASDAQ Trader symbol definitions](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs): official listing fields and exchange codes inform fallback completeness, fund identification and symbol handling.
- [Massive options-chain schema](https://massive.com/docs/rest/options/snapshots/option-chain-snapshot): quote/underlying timing, deliverable and provider-delay fields determine whether a contract can be actionable; absent fields remain unknown.
- [TwelveData API documentation](https://twelvedata.com/docs): provider field definitions and units drive explicit normalization rather than guessing by numeric magnitude.
- [Options Industry Council, theta](https://www.optionseducation.org/advancedconcepts/theta): option time decay varies with expiry, implied volatility and calendar time. An equity terminal-return classifier is insufficient to estimate call profitability.
- [SEC-filed leveraged-fund prospectus](https://www.sec.gov/Archives/edgar/data/1424958/000119312526192282/d927286d485bpos.htm): daily-reset leverage creates holding-period risk that differs from ordinary common-stock breakouts; verified fund characteristics require separate flags.
- Official [checkout](https://github.com/actions/checkout/releases), [setup-python](https://github.com/actions/setup-python/releases), and [upload-artifact](https://github.com/actions/upload-artifact/releases) releases informed the workflow runtime updates.

## Validation evidence

- Local Python 3.12: **143 tests run: 142 passed, one optional Torch test skipped**. This includes 79 additional regression cases across data, signals, models and output, plus corrections to existing expectations.
- Real installed XGBoost/Random Forest integration: both estimators fit seeded 1,200-row, 240-session synthetic panels. Random-label models must abstain under the stability gate; an engineered predictor exercises both active models and independent ensemble calibration. This verifies execution and gate behavior, not financial predictive power.
- Compiled the scanner and summary script; `git diff --check` passed.
- Offline cases cover provider circuits, class aliases, incomplete discovery, missing quick quotes, malformed bars, session labels, Wilder calculations, fundamental units, insider dates, date purges, missing labels, ensemble calibration/failures, option timing/deliverables, action vetoes, sizing, strict JSON and empty/repeated exports.
- Full live GitHub Actions scan: **pending in this audit draft**. Its results will be recorded before delivery. The temporary branch-only validation workflow will be removed after verification.
- Railway: the connected tools cannot execute a standalone one-time script without deploying a service. No Railway resource was created or changed. GitHub Actions supplies the already-authorized ephemeral runtime and existing secrets.

## Remaining limits and next validation requirements

1. **Actual trading policy:** measure walk-forward net returns for the ten-rule subset with next-session execution, spreads/slippage, overnight gaps, dividends, fees, realistic stop/target order ambiguity, time exits and portfolio overlap. Compare against simple momentum and index baselines with date-block uncertainty. Classification skill alone cannot replace this.
2. **Historical data:** obtain point-in-time listed/delisted constituents and as-of fundamentals/events. Current listings introduce survivorship bias. Provider split/dividend adjustment policies still differ and are now reported, not silently harmonized or claimed identical.
3. **Model selection:** weights, thresholds and archetypes require nested chronological selection and a final untouched period. Broad-panel calibration has not been established inside the strict selected subset. Changing the strategy on the same evaluation period would invalidate its evidence.
4. **Coverage:** 80% is an operational completeness floor, not a claim of total market coverage. Unsupported share-class conventions, data outages, illiquid/sub-$5/short-history names and alternate strategies can still be excluded. Exported counts make these boundaries reviewable.
5. **Context:** bounded news is not exhaustive current-event research. Company legal headlines remain risk context. BLS/BEA calendar coverage may be unverified under current API plans, and no direct social-source sentiment series is collected.
6. **Options:** standard-quote metadata is mandatory. Some providers may supply useful chains without enough metadata to authorize a call. Approximate European delta is labeled; option-return validation needs historical contracts, IV dynamics, corporate actions and American exercise/dividend treatment.
7. **Risk and exits:** targets are illustrative 2.5R levels, not forecasts. Standalone per-name sizing is not a simultaneous portfolio allocation. Stops do not cap overnight gaps. The CLI does not run a broker position monitor.
8. **Heuristics:** panel overlap remains; ATR snapshots do not establish a full VCP pattern. Experimental LSTM remains disabled and unvalidated. There is no substantiated promise of maximum returns.

**User action required for exposed credentials:** revoke/rotate the previously
public Massive, TwelveData and Finnhub keys, then store replacements as repository
Actions secrets. Current-source removal and log redaction are complete; historical
exposure cannot be repaired by changing scoring code.
