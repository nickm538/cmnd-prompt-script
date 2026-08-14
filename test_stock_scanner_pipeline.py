from datetime import datetime, timedelta
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import new_stock_scanner_pipeline_claude_opus_41426 as scanner


def _guard_ready_df(last_day, unique_prices=True):
    idx = pd.bdate_range(end=pd.Timestamp(last_day), periods=scanner.MIN_TRADING_DAYS)
    closes = np.linspace(100.0, 130.0, scanner.MIN_TRADING_DAYS)
    if not unique_prices:
        closes[:] = 100.0
    return pd.DataFrame(
        {
            "Close": closes,
            "Volume": np.full(scanner.MIN_TRADING_DAYS, 100_000),
            "Avg_Dollar_Vol_20": np.full(scanner.MIN_TRADING_DAYS, 12_000_000),
            "Return_63d": np.full(scanner.MIN_TRADING_DAYS, 0.12),
            "ATR_pct": np.full(scanner.MIN_TRADING_DAYS, 0.03),
        },
        index=idx,
    )


class ScannerRegressionTests(unittest.TestCase):
    def setUp(self):
        scanner.reset_market_router()

    def test_missing_macro_series_degrades_instead_of_stopping_the_scan(self):
        # The live failure: 9 of 10 macro series loaded, ^IRX did not, and the
        # whole universe scan was discarded at Stage 0. Macro is context -- it
        # tunes sizing and the panel's market-direction score, it does not pick
        # or price a candidate -- so a dead endpoint must not cost the run.
        macro = scanner.MacroRegime()
        real_series = macro._series

        def drop_irx(symbol, *args, **kwargs):
            return None if symbol == macro.SYMBOLS["irx"] else real_series(
                symbol, *args, **kwargs
            )

        idx = pd.bdate_range(end="2026-08-13", periods=260)
        frame = pd.DataFrame({"Close": np.linspace(100.0, 130.0, 260)}, index=idx)

        def fake_series(symbol, *args, **kwargs):
            return None if symbol == macro.SYMBOLS["irx"] else frame.copy()

        with patch.object(macro, "_series", side_effect=fake_series):
            with patch.object(macro.world, "load"):
                with patch.object(scanner.log, "warning"):
                    macro.load()   # must not raise

        self.assertTrue(macro.degraded)
        self.assertEqual(macro.missing_symbols, ["irx"])
        self.assertEqual(macro.missing_required, ["irx"])
        self.assertIsNotNone(macro.regime_score)
        self.assertNotEqual(macro.regime_label, "UNAVAILABLE")
        # The absent series contributes nothing -- no yield-curve note is
        # invented from a guessed short rate.
        self.assertFalse(any("Yield curve" in n for n in macro.notes))
        payload = macro.to_dict()
        self.assertTrue(payload["macro_degraded"])
        self.assertEqual(payload["inputs_missing_required"], ["irx"])

    def test_total_macro_outage_still_yields_a_neutral_regime(self):
        macro = scanner.MacroRegime()
        with patch.object(macro, "_series", return_value=None):
            with patch.object(macro.world, "load"):
                with patch.object(scanner.log, "warning"):
                    macro.load()   # must not raise

        self.assertTrue(macro.degraded)
        self.assertEqual(macro.regime_score, 50.0)
        self.assertEqual(macro.regime_label, "NEUTRAL")
        self.assertEqual(macro.notes, [])
        self.assertEqual(len(macro.missing_symbols), len(macro.SYMBOLS))

    def test_partial_macro_never_sizes_above_neutral(self):
        macro = scanner.MacroRegime()
        macro.regime_score = 90.0          # would normally scale to 1.20
        macro.world.event_risk = 10.0
        self.assertEqual(macro.position_sizing_scalar(), 1.20)

        macro.missing_required = ["irx"]
        # Read the regime from what loaded, but do not lever up on a partial
        # picture: above 1.0 asserts confirmed risk-on.
        self.assertEqual(macro.position_sizing_scalar(), 1.00)

    def test_expected_last_closed_trading_day_before_close_uses_prior_session(self):
        now = datetime(2026, 4, 30, 12, 0, tzinfo=scanner.ET_TZ)

        self.assertEqual(
            scanner.expected_last_closed_trading_day(now),
            datetime(2026, 4, 29).date(),
        )

    def test_execution_guard_rejects_one_session_stale_daily_data(self):
        now = datetime(2026, 4, 30, 12, 0, tzinfo=scanner.ET_TZ)
        stale_day = datetime(2026, 4, 28).date()
        df = _guard_ready_df(stale_day)

        reason = scanner.ExecutionGuards._check("TEST", df, now=now)

        self.assertEqual(
            reason,
            "GUARD_A: Stale data (1 trading sessions behind expected 2026-04-29)",
        )

    def test_execution_guard_accepts_expected_prior_session_before_close(self):
        now = datetime(2026, 4, 30, 12, 0, tzinfo=scanner.ET_TZ)
        expected_day = datetime(2026, 4, 29).date()
        df = _guard_ready_df(expected_day)

        self.assertIsNone(scanner.ExecutionGuards._check("TEST", df, now=now))

    def test_missing_fundamentals_are_not_neutral_boosted(self):
        panel = scanner.InvestorPanel()

        unknown_score = panel._score_lynch("ETF", pd.DataFrame(), {"sector": "Unknown"})
        error_score = panel._score_lynch(
            "BAD", pd.DataFrame(), {"fundamentals_quality": "failed", "sector": "Unknown"}
        )

        self.assertGreater(unknown_score, error_score)
        self.assertEqual(error_score, 35.0)

    def test_options_require_live_bid_ask_for_candidate(self):
        candidate = {
            "ticker": "TEST",
            "price": 100.0,
            "ml_ensemble_score": 0.8,
            "panel_composite_score": 80,
            "return_20d": 0.08,
            "rsi_14": 55,
        }
        exp = (datetime.now().date() + timedelta(days=35)).isoformat()
        chain = [{
            "expiry": exp,
            "strike": 100.0,
            "bid": None,
            "ask": None,
            "mid": 5.0,
            "oi": 1_000,
            "volume": 100,
            "iv": 0.35,
            "delta": 0.55,
            "theta": -0.02,
            "break_even": 105.0,
            "underlying_price": 100.0,
        }]

        with patch.object(scanner.OptionsEvaluator, "_fetch_chain_mboum", return_value=chain):
            with patch.object(scanner.OptionsEvaluator, "_fetch_chain_massive", return_value=[]):
                with patch.object(scanner.OptionsEvaluator, "_fetch_chain_yahoo", return_value=[]):
                    result = scanner.OptionsEvaluator._find_best_option(candidate, {})

        self.assertEqual(result["option_candidate"], "N")

    def test_rank_pool_backfills_top7_from_live_near_misses(self):
        data = {
            f"T{i}": _guard_ready_df(datetime(2026, 4, 29).date())
            for i in range(10)
        }
        strict = [
            scanner.HardBuyRules._candidate_record(
                "T0",
                data["T0"],
                rule_result={
                    "rules_passed": 10,
                    "rules_failed": 0,
                    "passed_rules": [f"BUY_{i:02d}" for i in range(1, 11)],
                    "failed_rules": [],
                },
                hard_buy_pass=True,
            )
        ]
        near_misses = [
            {
                "ticker": f"T{i}",
                "price": 100 + i,
                "rules_passed": 9 if i < 8 else 7,
                "rules_failed": 1 if i < 8 else 3,
                "passed_rules": [f"BUY_{j:02d}" for j in range(1, 10)],
                "failed_rules": ["BUY_05:Crossover"],
                "return_20d": 0.08,
                "volume_ratio": 1.5,
                "rsi_14": 55,
            }
            for i in range(10)
        ]

        with patch.object(scanner.HardBuyRules, "near_misses", return_value=near_misses):
            pool, _ = scanner.HardBuyRules.build_rank_pool(
                data, strict, target_size=7, max_pool_size=7
            )

        self.assertEqual(len(pool), 7)
        self.assertEqual(pool[0]["ticker"], "T0")
        self.assertTrue(pool[0]["hard_buy_pass"])
        self.assertTrue(all(p["rules_passed"] >= scanner.MIN_NEAR_MISS_RULES for p in pool))
        self.assertTrue(any("NEAR_MISS_9_OF_10" in p["flags"] for p in pool[1:]))

    def test_panel_can_score_without_filtering_backfill_pool(self):
        panel = scanner.InvestorPanel()
        survivor = {"ticker": "TEST"}
        data = {"TEST": _guard_ready_df(datetime(2026, 4, 29).date())}

        with patch.object(panel, "load_benchmark", return_value=None):
            with patch.object(panel, "_score_livermore", return_value=50.0):
                with patch.object(panel, "_score_druckenmiller", return_value=50.0):
                    with patch.object(panel, "_score_lynch", return_value=50.0):
                        with patch.object(panel, "_score_minervini", return_value=50.0):
                            with patch.object(panel, "_score_oneil", return_value=50.0):
                                unfiltered = panel.score_all(
                                    [survivor.copy()], data, {}, apply_filter=False
                                )
                                filtered = panel.score_all(
                                    [survivor.copy()], data, {}, apply_filter=True
                                )

        self.assertEqual(len(unfiltered), 1)
        self.assertEqual(len(filtered), 0)

    def test_trade_setup_score_uses_equity_quality_without_options(self):
        candidate = {
            "ticker": "TEST",
            "ml_ensemble_score": 0.65,
            "panel_composite_score": 72,
            "rules_passed": 9,
            "return_20d": 0.10,
            "close_vs_sma50": 0.07,
            "ema20_vs_ema50": 0.04,
            "avg_dollar_volume": 50_000_000,
            "option_candidate": "N",
            "option_score": 0.0,
            "flags": [],
        }

        score = scanner.OptionsEvaluator._trade_setup_score(candidate)

        self.assertGreaterEqual(score, 60.0)
        self.assertEqual(candidate["overall_confidence_score"], round(score, 1))

    def test_credit_errors_trip_mboum_but_minute_limits_do_not(self):
        self.assertEqual(scanner.classify_http_error(402, ""), "credit")
        self.assertEqual(
            scanner.classify_http_error(429, "You have run out of API credits for the current month"),
            "credit",
        )
        self.assertEqual(
            scanner.classify_http_error(429, "You have run out of API credits for the current minute"),
            "rate",
        )
        self.assertEqual(scanner.classify_http_error(403, "You don't have access"), "auth")

        circuit = scanner.ProviderCircuit.get("MBOUM-OHLCV", fail_limit=4)
        self.assertTrue(callable(getattr(circuit, "record_failure", None)))
        self.assertTrue(circuit.available())
        circuit.record_failure("credits exhausted", credit=True)
        self.assertFalse(circuit.available())
        scanner.ProviderCircuit.reset_all()
        self.assertTrue(scanner.ProviderCircuit.get("MBOUM-OHLCV", fail_limit=4).available())

    def test_missing_mboum_key_is_optional_when_massive_is_present(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": "massive-test-key"}, clear=True):
            with patch.object(scanner.log, "warning"), patch.object(scanner.log, "info"):
                status = scanner.verify_api_credentials()
        self.assertEqual(status["MASSIVE_API_KEY"], "env")
        self.assertEqual(status["MBOUM_API_KEY"], "absent")
        self.assertEqual(status["TWELVEDATA_API_KEY"], "embedded")
        self.assertEqual(status["FINNHUB_API_KEY"], "embedded")

    def test_committed_fallback_keys_are_used_when_secrets_are_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(scanner.log, "warning"), patch.object(scanner.log, "info"):
                status = scanner.verify_api_credentials()
            massive = scanner._resolve_api_key("MASSIVE_API_KEY")
            twelve = scanner._resolve_api_key("TWELVEDATA_API_KEY")
            finnhub = scanner._resolve_api_key("FINNHUB_API_KEY")
        self.assertEqual(status["MASSIVE_API_KEY"], "embedded")
        self.assertEqual(status["TWELVEDATA_API_KEY"], "embedded")
        self.assertEqual(status["FINNHUB_API_KEY"], "embedded")
        self.assertTrue(massive)
        self.assertTrue(twelve)
        self.assertTrue(finnhub)

    def test_credential_status_ignores_import_time_key_snapshot(self):
        # On Actions every secret is exported for the whole job, so the module
        # constants hold live keys. Status must report where the key comes from
        # now, not replay that snapshot -- otherwise a real MBOUM secret is
        # labelled "embedded" even though MBOUM has no committed fallback.
        with patch.object(scanner, "MBOUM_API_KEY", "live-mboum-secret"):
            with patch.object(scanner, "MBOUM_OPTIONS_KEY", "live-mboum-options-secret"):
                with patch.dict(os.environ, {"MASSIVE_API_KEY": "massive-test-key"}, clear=True):
                    with patch.object(scanner.log, "warning"), patch.object(scanner.log, "info"):
                        status = scanner.verify_api_credentials()
        self.assertEqual(status["MBOUM_API_KEY"], "absent")
        self.assertEqual(status["MBOUM_OPTIONS_KEY"], "absent")

    def test_env_secrets_win_over_committed_fallback_keys(self):
        secrets = {
            "MASSIVE_API_KEY": "massive-secret",
            "MBOUM_API_KEY": "mboum-secret",
            "MBOUM_OPTIONS_KEY": "mboum-options-secret",
            "TWELVEDATA_API_KEY": "twelvedata-secret",
            "FINNHUB_API_KEY": "finnhub-secret",
        }
        with patch.dict(os.environ, secrets, clear=True):
            with patch.object(scanner.log, "warning"), patch.object(scanner.log, "info"):
                status = scanner.verify_api_credentials()
            for name, value in secrets.items():
                self.assertEqual(scanner._resolve_api_key(name), value)
        for name in secrets:
            self.assertEqual(status[name], "env")

    def test_mboum_keys_have_no_committed_fallback(self):
        # MBOUM is the credit-metered primary; a committed key would spend the
        # very plan the Massive -> TwelveData -> Finnhub -> Yahoo chain backs up.
        self.assertNotIn("MBOUM_API_KEY", scanner.EMBEDDED_FALLBACK_KEYS)
        self.assertNotIn("MBOUM_OPTIONS_KEY", scanner.EMBEDDED_FALLBACK_KEYS)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scanner._resolve_api_key("MBOUM_API_KEY"), "")
            self.assertEqual(scanner._resolve_api_key("MBOUM_OPTIONS_KEY"), "")

    def test_ohlcv_router_keeps_mboum_primary_when_it_returns_history(self):
        history = _guard_ready_df(datetime(2026, 4, 29).date())
        router = scanner.MarketDataRouter()
        with patch.object(router, "_provider_enabled", return_value=True):
            with patch.object(router, "_history_mboum", return_value=history) as mboum:
                with patch.object(router, "_history_massive") as massive:
                    df, source = router.get_history("AAPL")
        self.assertEqual(source, "MBOUM")
        self.assertIs(df, history)
        mboum.assert_called_once_with("AAPL")
        massive.assert_not_called()

    def test_ohlcv_router_falls_back_to_massive_when_mboum_is_out_of_credit(self):
        history = _guard_ready_df(datetime(2026, 4, 29).date())
        router = scanner.MarketDataRouter()
        with patch.object(router, "_provider_enabled", return_value=True):
            with patch.object(
                router,
                "_history_mboum",
                side_effect=scanner.ProviderExhausted("MBOUM", "credits exhausted"),
            ):
                with patch.object(router, "_history_massive", return_value=history) as massive:
                    with patch.object(router, "_history_twelvedata") as twelve:
                        df, source = router.get_history("AAPL")
        self.assertEqual(source, "Massive")
        self.assertIs(df, history)
        massive.assert_called_once_with("AAPL")
        twelve.assert_not_called()

    def test_ohlcv_router_skips_tripped_mboum_circuit_on_later_tickers(self):
        history = _guard_ready_df(datetime(2026, 4, 29).date())
        router = scanner.MarketDataRouter()
        scanner.ProviderCircuit.get("MBOUM-OHLCV", fail_limit=4).trip("credits exhausted")
        with patch.object(router, "_provider_enabled", return_value=True):
            with patch.object(router, "_history_mboum") as mboum:
                with patch.object(router, "_history_massive", return_value=history):
                    df, source = router.get_history("MSFT")
        self.assertEqual(source, "Massive")
        self.assertIsNotNone(df)
        mboum.assert_not_called()

    def test_world_context_never_adds_tickers(self):
        ctx = scanner.WorldContext()
        ctx.earnings_soon = {"TEST": "2026-08-18"}
        incoming = [{"ticker": "TEST", "flags": []}, {"ticker": "OTHER", "flags": []}]
        with patch.object(ctx, "_company_headlines", return_value=["Live geopolitics headline"]):
            out = ctx.annotate(incoming)
        self.assertEqual([row["ticker"] for row in out], ["TEST", "OTHER"])
        self.assertIn("LIVE_EARNINGS_WINDOW", out[0]["flags"])
        self.assertIn("LIVE_NEWS", out[0]["flags"])
        self.assertNotIn("LIVE_EARNINGS_WINDOW", out[1]["flags"])
        self.assertFalse(ctx.to_dict()["seeds_universe"])

    def test_high_live_event_risk_tightens_position_sizing(self):
        macro = scanner.MacroRegime()
        macro.regime_score = 80.0
        macro.world.event_risk = 80.0
        self.assertLess(macro.position_sizing_scalar(), 1.20)
        macro.world.event_risk = 40.0
        self.assertEqual(macro.position_sizing_scalar(), 1.20)

    def test_universe_cleaner_is_a_filter_not_a_basket(self):
        cleaned = scanner.UniverseDiscovery._clean_listed_equities(
            [
                {"ticker": "TEST", "primary_exchange": "XNAS", "type": "CS"},
                {"ticker": "BRK.A", "primary_exchange": "XNYS", "type": "CS"},
                {"symbol": "FAKE", "mic": "XNAS", "type": "Warrant"},
            ]
        )
        self.assertEqual(cleaned, ["TEST"])

    def test_engine_has_no_preset_screener_or_spy_etf_fetch(self):
        source = inspect.getsource(scanner)
        self.assertNotIn('get_history("SPY")', source)
        self.assertNotIn("most_actives", source)
        self.assertNotIn("SCAN_TICKERS", source)
        self.assertNotIn("TICKER_LIST", source)
        self.assertFalse(hasattr(scanner.MboumAPI, "get_screener"))

    def test_benchmark_uses_live_spx_snapshot_not_spy(self):
        idx = pd.bdate_range(end="2026-04-29", periods=120)
        spx = pd.DataFrame({"Close": np.linspace(4000.0, 5200.0, 120)}, index=idx)
        macro = scanner.MacroRegime()
        macro.snapshot["spx"] = {"df": spx, "last": 5200.0}
        panel = scanner.InvestorPanel(macro=macro)
        with patch.object(scanner, "get_market_router") as router:
            with patch.object(macro, "_series") as series:
                panel.load_benchmark()
        router.assert_not_called()
        series.assert_not_called()
        self.assertIs(panel.benchmark_data, spx)

    def test_benchmark_fallback_fetches_live_spx_index_not_spy(self):
        idx = pd.bdate_range(end="2026-04-29", periods=120)
        spx = pd.DataFrame({"Close": np.linspace(4000.0, 5200.0, 120)}, index=idx)
        macro = scanner.MacroRegime()
        panel = scanner.InvestorPanel(macro=macro)
        with patch.object(macro, "_series", return_value=spx) as series:
            with patch.object(scanner, "get_market_router") as router:
                panel.load_benchmark()
        series.assert_called_once_with("^GSPC", range_="2y")
        router.assert_not_called()
        self.assertIs(panel.benchmark_data, spx)

    def test_live_vix_raises_event_risk(self):
        ctx = scanner.WorldContext()
        ctx._score_event_risk(snapshot={"vix": {"last": 40.0}})
        stressed = ctx.event_risk
        ctx._score_event_risk(snapshot={"vix": {"last": 12.0}})
        self.assertGreater(stressed, ctx.event_risk)

    def test_anti_chase_limits_reject_already_run_moves(self):
        base = {
            "Close_vs_SMA50": 0.10, "RSI_14": 55.0,
            "Return_5d": 0.03, "Return_20d": 0.10, "Stretch_ATR": 1.5,
        }
        self.assertIsNone(scanner.ExecutionGuards._exhaustion_reason(pd.Series(base)))

        # Each limit rejects on its own. The values that motivated tightening:
        # 30% over the 50DMA, RSI 80, +35% in a week and +80% in a month all
        # cleared the previous thresholds.
        for field, value, expect in [
            ("Close_vs_SMA50", 0.30, "Over-extended"),
            ("RSI_14", 80.0, "Blow-off RSI"),
            ("Return_5d", 0.35, "Vertical 5-day spike"),
            ("Return_20d", 0.80, "Parabolic 20-day run"),
            ("Stretch_ATR", 6.0, "Climax extension"),
        ]:
            reason = scanner.ExecutionGuards._exhaustion_reason(
                pd.Series({**base, field: value})
            )
            self.assertIsNotNone(reason, f"{field}={value} should be rejected")
            self.assertIn(expect, reason)

    def test_climax_extension_is_volatility_aware_not_percentage_only(self):
        # A name only 12% over its 50DMA but 6 ATR above its 20-day mean has
        # gone vertical *for what it is*. The percentage checks cannot see that.
        quiet_but_vertical = pd.Series({
            "Close_vs_SMA50": 0.12, "RSI_14": 62.0,
            "Return_5d": 0.08, "Return_20d": 0.15, "Stretch_ATR": 6.0,
        })
        reason = scanner.ExecutionGuards._exhaustion_reason(quiet_but_vertical)
        self.assertIn("Climax extension", reason)

        # A high-beta name at the same 3 ATR is an ordinary breakout, not a
        # climax, and must survive.
        high_beta_breakout = pd.Series({
            "Close_vs_SMA50": 0.22, "RSI_14": 66.0,
            "Return_5d": 0.11, "Return_20d": 0.30, "Stretch_ATR": 3.0,
        })
        self.assertIsNone(
            scanner.ExecutionGuards._exhaustion_reason(high_beta_breakout)
        )

    def test_ml_features_are_cross_sectionally_comparable(self):
        # No raw price-unit or raw-dollar magnitudes: the model ranks a $9 name
        # against a $900 one, so every feature has to be a ratio or log scale.
        self.assertNotIn("MACD_histogram", scanner.MLRanker.FEATURE_COLS)
        self.assertNotIn("Avg_Dollar_Vol_20", scanner.MLRanker.FEATURE_COLS)
        self.assertIn("MACD_hist_pct", scanner.MLRanker.FEATURE_COLS)
        self.assertIn("Log_Dollar_Vol_20", scanner.MLRanker.FEATURE_COLS)

        idx = pd.bdate_range(end="2026-04-29", periods=300)
        rng = np.random.default_rng(5)
        steps = rng.normal(0, 0.01, 300)
        cheap = pd.DataFrame({"Close": 9.0 * np.exp(np.cumsum(steps))}, index=idx)
        rich = pd.DataFrame({"Close": 900.0 * np.exp(np.cumsum(steps))}, index=idx)
        for frame in (cheap, rich):
            frame["Open"] = frame["Close"]
            frame["High"] = frame["Close"] * 1.01
            frame["Low"] = frame["Close"] * 0.99
            frame["Volume"] = 1_000_000.0

        cheap_out = scanner.TechnicalEngine.compute_all(cheap)
        rich_out = scanner.TechnicalEngine.compute_all(rich)

        # Identical price *path*, 100x different level: the normalised MACD
        # must agree, while the raw one differs by roughly that same factor.
        self.assertAlmostEqual(
            cheap_out["MACD_hist_pct"].iloc[-1],
            rich_out["MACD_hist_pct"].iloc[-1],
            places=6,
        )
        self.assertGreater(
            abs(rich_out["MACD_histogram"].iloc[-1]),
            abs(cheap_out["MACD_histogram"].iloc[-1]) * 50,
        )
        # Log liquidity separates decades, not just the mega-cap tail.
        self.assertAlmostEqual(
            rich_out["Log_Dollar_Vol_20"].iloc[-1],
            np.log10(rich_out["Avg_Dollar_Vol_20"].iloc[-1]),
            places=6,
        )

    def test_lstm_sequence_norm_cannot_see_the_future(self):
        idx = pd.bdate_range(end="2026-04-29", periods=120)
        rng = np.random.default_rng(11)
        feats = pd.DataFrame(
            {"a": rng.normal(100, 5, 120), "b": rng.normal(0, 1, 120)}, index=idx
        )

        normed = scanner.MLRanker._causal_sequence_norm(feats, 20)

        # Rewriting the tail must not move a single earlier normalised value.
        tampered = feats.copy()
        tampered.iloc[90:] *= 50.0
        tampered_normed = scanner.MLRanker._causal_sequence_norm(tampered, 20)
        pd.testing.assert_frame_equal(normed.iloc[:90], tampered_normed.iloc[:90])

        # And the transform is a real trailing z-score, not a passthrough.
        window = feats["a"].iloc[80:100]
        expected = (feats["a"].iloc[99] - window.mean()) / window.std()
        self.assertAlmostEqual(normed["a"].iloc[99], expected, places=9)
        self.assertTrue(np.isfinite(normed.values).all())

    def test_backfill_pool_rejects_broken_trend_and_late_rsi(self):
        data = {
            f"T{i}": _guard_ready_df(datetime(2026, 4, 29).date()) for i in range(4)
        }
        # T1 is below its own 200DMA, T2 arrives with RSI already run up, T3 is
        # clean. All three pass the 8-of-10 near-miss bar.
        near_misses = [
            {
                "ticker": "T1",
                "rules_passed": 8,
                "rules_failed": 2,
                "passed_rules": [f"BUY_{j:02d}" for j in (2, 3, 4, 5, 6, 7, 8, 9)],
                "failed_rules": ["BUY_01:Trend", "BUY_10:Penny"],
                "return_20d": 0.09, "volume_ratio": 1.6, "rsi_14": 55,
            },
            {
                "ticker": "T2",
                "rules_passed": 8,
                "rules_failed": 2,
                "passed_rules": [f"BUY_{j:02d}" for j in (1, 2, 3, 4, 5, 6, 7, 9)],
                "failed_rules": ["BUY_08:RSI", "BUY_10:Penny"],
                "return_20d": 0.09, "volume_ratio": 1.6, "rsi_14": 78,
            },
            {
                "ticker": "T3",
                "rules_passed": 8,
                "rules_failed": 2,
                "passed_rules": [f"BUY_{j:02d}" for j in (1, 2, 4, 6, 7, 8, 9, 10)],
                "failed_rules": ["BUY_03:BB/High", "BUY_05:Crossover"],
                "return_20d": 0.09, "volume_ratio": 1.6, "rsi_14": 55,
            },
        ]

        with patch.object(scanner.HardBuyRules, "near_misses", return_value=near_misses):
            pool, _ = scanner.HardBuyRules.build_rank_pool(
                data, [], target_size=7, max_pool_size=7
            )

        tickers = [p["ticker"] for p in pool]
        self.assertIn("T3", tickers)
        # Failing the trend filter or the RSI sweet spot is not a timing miss,
        # it is the "already ran / broken chart" entry the strategy refuses.
        self.assertNotIn("T1", tickers)
        self.assertNotIn("T2", tickers)

    def test_backfill_pool_rejects_already_extended_move(self):
        df = _guard_ready_df(datetime(2026, 4, 29).date())
        nm = {
            "ticker": "T0",
            "rules_passed": 9,
            "rules_failed": 1,
            "passed_rules": [f"BUY_{j:02d}" for j in range(1, 10)],
            "failed_rules": ["BUY_05:Crossover"],
            "return_20d": 0.09, "volume_ratio": 1.6, "rsi_14": 55,
        }
        record = scanner.HardBuyRules._candidate_record(
            "T0", df, rule_result=nm, hard_buy_pass=False
        )

        self.assertIsNone(scanner.HardBuyRules._backfill_block_reason(nm, record))

        extended = dict(record, flags=list(record["flags"]) + ["EXTENDED_MOVE"])
        reason = scanner.HardBuyRules._backfill_block_reason(nm, extended)
        self.assertIsNotNone(reason)
        self.assertIn("extended", reason)

    def test_ml_training_pool_is_not_conditioned_on_recent_performance(self):
        # A name whose 63-day return is negative is excluded by the execution
        # guards but must still train the model: filtering the training set on
        # performance measured at the end of the window leaks that outcome back
        # into every historical row.
        loser = _guard_ready_df(datetime(2026, 4, 29).date())
        loser["Return_63d"] = -0.25
        winner = _guard_ready_df(datetime(2026, 4, 29).date())
        illiquid = _guard_ready_df(datetime(2026, 4, 29).date())
        illiquid["Avg_Dollar_Vol_20"] = 100_000
        penny = _guard_ready_df(datetime(2026, 4, 29).date())
        penny["Close"] = 2.0

        pool = scanner.ExecutionGuards.ml_training_pool(
            {"LOSER": loser, "WINNER": winner, "ILLIQUID": illiquid, "PENNY": penny}
        )

        self.assertEqual(sorted(pool), ["ILLIQUID", "LOSER", "WINNER"])
        # ...while the guards themselves still reject the loser for trading.
        self.assertIsNotNone(
            scanner.ExecutionGuards._check(
                "LOSER", loser, now=datetime(2026, 4, 30, 12, 0, tzinfo=scanner.ET_TZ)
            )
        )

    def test_training_cap_keeps_the_chronological_tail(self):
        rows = 200_000
        X = np.arange(rows, dtype=float).reshape(-1, 1)
        y = (np.arange(rows) % 3 == 0).astype(int)

        X_cap, y_cap = scanner.MLRanker._cap_training_rows(X, y, max_rows=80_000)

        self.assertEqual(len(X_cap), 80_000)
        # Date-sorted rows in, contiguous most-recent block out. Sampling each
        # class separately used to reorder them and strand a minority-only
        # block at the front.
        np.testing.assert_array_equal(X_cap, X[-80_000:])
        np.testing.assert_array_equal(y_cap, y[-80_000:])

    def test_training_cap_leaves_no_single_class_cv_fold(self):
        # The live failure: 511,649 samples at 30.6% positive produced a capped
        # set whose oldest ~22,300 rows were all class 1, so TimeSeriesSplit
        # handed XGBoost a single-class first fold and the scan died with
        # "Invalid classes inferred from unique values of `y`".
        rng = np.random.default_rng(0)
        y = (rng.random(511_649) < 0.306).astype(int)
        X = np.zeros((len(y), 1))

        _, y_cap = scanner.MLRanker._cap_training_rows(X, y, max_rows=80_000)

        gap = 20
        folds = 0
        for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(y_cap):
            train_idx = train_idx[train_idx < (val_idx[0] - gap)]
            if train_idx.size == 0:
                continue
            folds += 1
            self.assertEqual(
                np.unique(y_cap[train_idx]).size, 2,
                "TimeSeriesSplit fold is single-class -- XGBoost will raise",
            )
        self.assertGreater(folds, 0)

    def test_scale_pos_weight_rebalances_the_minority_class(self):
        y = np.array([0] * 700 + [1] * 300)

        self.assertAlmostEqual(scanner.MLRanker._scale_pos_weight(y), 700 / 300)
        self.assertEqual(scanner.MLRanker._class_counts(y), (700, 300))
        # Degenerate targets must not produce a divide-by-zero weight.
        self.assertEqual(scanner.MLRanker._scale_pos_weight(np.ones(10, dtype=int)), 1.0)
        self.assertEqual(scanner.MLRanker._scale_pos_weight(np.zeros(10, dtype=int)), 1.0)

    def test_single_class_labels_degrade_gracefully_instead_of_crashing(self):
        # Contract: the raw trainers signal an impossible fit by raising the
        # typed _ModelDegradedError -- never XGBoost's ValueError or a
        # one-column predict_proba IndexError -- and _safe_scores converts
        # that into neutral scores plus a recorded degradation, so Stage 4
        # keeps the scan alive and the report says the ensemble was degraded.
        n_feat = len(scanner.MLRanker.FEATURE_COLS)
        rng = np.random.default_rng(1)
        X_train = rng.normal(size=(500, n_feat))
        X_current = rng.normal(size=(3, n_feat))

        for labels in (np.ones(500, dtype=int), np.zeros(500, dtype=int)):
            ranker = scanner.MLRanker()
            with self.assertRaises(scanner._ModelDegradedError):
                ranker._train_xgboost(X_train, labels, X_current)
            with self.assertRaises(scanner._ModelDegradedError):
                ranker._train_rf(X_train, labels, X_current)

            with patch.object(scanner.log, "warning"):
                xgb_scores = ranker._safe_scores(
                    "XGBoost", ranker._train_xgboost, X_train, labels, X_current
                )
                rf_scores = ranker._safe_scores(
                    "RandomForest", ranker._train_rf, X_train, labels, X_current
                )
            np.testing.assert_array_equal(xgb_scores, np.full(3, 0.5))
            np.testing.assert_array_equal(rf_scores, np.full(3, 0.5))
            self.assertEqual(ranker.degraded_models, ["XGBoost", "RandomForest"])

    def test_optional_lstm_failure_cannot_end_the_scan(self):
        # torch is installed in CI but not locally, so this path first executes
        # on a live run. The LSTM is purely informational -- lstm_score is
        # reported, never ranked on -- so a failure there must cost the score,
        # not the scan.
        ranker = scanner.MLRanker()
        n_feat = len(scanner.MLRanker.FEATURE_COLS)
        rng = np.random.default_rng(4)
        X_train = rng.normal(size=(600, n_feat))
        y_train = (rng.random(600) < 0.35).astype(int)
        X_current = rng.normal(size=(1, n_feat))
        survivors = [{"ticker": "TEST", "flags": []}]

        with patch.object(
            ranker, "_build_dataset",
            return_value=(X_train, y_train, X_current, ["TEST"]),
        ):
            with patch.object(ranker, "_train_xgboost", return_value=np.array([0.7])):
                with patch.object(ranker, "_train_rf", return_value=np.array([0.9])):
                    with patch.object(
                        ranker, "_train_lstm", side_effect=RuntimeError("torch blew up")
                    ):
                        with patch.object(scanner.log, "error"):
                            out = ranker.rank(survivors, {}, training_universe={})

        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["lstm_score"])
        self.assertIn("LSTM", ranker.degraded_models)
        # The tree ensemble is untouched -- only the optional layer is lost.
        self.assertEqual(out[0]["ml_score_xgb"], 0.7)
        self.assertEqual(out[0]["ml_score_rf"], 0.9)
        self.assertAlmostEqual(out[0]["ml_ensemble_score"], 0.8)
        self.assertIn("ML_DEGRADED:LSTM", out[0]["flags"])

    @unittest.skipUnless(scanner.LSTM_AVAILABLE, "torch not installed")
    def test_optional_lstm_internal_training_failure_propagates_to_degraded(self):
        # This test verifies that when an exception occurs INSIDE _train_lstm's
        # training code (not mocking _train_lstm itself), it propagates up to
        # MLRanker.rank()'s try/except, which then correctly adds "LSTM" to
        # degraded_models and applies the ML_DEGRADED:LSTM flag.
        ranker = scanner.MLRanker()
        n_feat = len(scanner.MLRanker.FEATURE_COLS)
        rng = np.random.default_rng(5)
        X_train = rng.normal(size=(600, n_feat))
        y_train = (rng.random(600) < 0.35).astype(int)
        X_current = rng.normal(size=(1, n_feat))
        survivors = [{"ticker": "TEST", "flags": []}]

        # Build a DataFrame with all required LSTM SEQ_FEATURES columns and enough rows
        n_days = 252  # MIN_TRADING_DAYS
        dates = pd.bdate_range(end=pd.Timestamp("2026-01-15"), periods=n_days)
        df_data = {
            "Close": np.linspace(100.0, 120.0, n_days),
            "Volume": np.full(n_days, 1_000_000),
            "RSI_14": np.full(n_days, 55.0),
            "MACD_histogram": np.full(n_days, 0.1),
            "EMA_20": np.linspace(100.0, 120.0, n_days),
            "EMA_50": np.linspace(98.0, 118.0, n_days),
            "SMA_50": np.linspace(99.0, 119.0, n_days),
            "SMA_200": np.linspace(95.0, 115.0, n_days),
            "Avg_Dollar_Vol_20": np.full(n_days, 12_000_000),
            "Vol_SMA_20": np.full(n_days, 1_000_000),
        }
        all_data = {"TEST": pd.DataFrame(df_data, index=dates)}

        with patch.object(
            ranker, "_build_dataset",
            return_value=(X_train, y_train, X_current, ["TEST"]),
        ):
            with patch.object(ranker, "_train_xgboost", return_value=np.array([0.7])):
                with patch.object(ranker, "_train_rf", return_value=np.array([0.9])):
                    # Ensure LSTM_AVAILABLE is True so the method enters the training path
                    with patch.object(scanner, "LSTM_AVAILABLE", True):
                        # Inject a failure inside the training code by patching torch.optim.Adam
                        with patch("torch.optim.Adam", side_effect=RuntimeError("Adam optimizer failed")):
                            with patch.object(scanner.log, "error"):
                                with patch.object(scanner.log, "warning"):
                                    out = ranker.rank(survivors, all_data, training_universe={})

        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["lstm_score"])
        self.assertIn("LSTM", ranker.degraded_models)
        # The tree ensemble is untouched -- only the optional layer is lost.
        self.assertEqual(out[0]["ml_score_xgb"], 0.7)
        self.assertEqual(out[0]["ml_score_rf"], 0.9)
        self.assertAlmostEqual(out[0]["ml_ensemble_score"], 0.8)
        self.assertIn("ML_DEGRADED:LSTM", out[0]["flags"])

    def test_model_failure_degrades_and_flags_instead_of_ending_the_scan(self):
        ranker = scanner.MLRanker()
        n_feat = len(scanner.MLRanker.FEATURE_COLS)
        rng = np.random.default_rng(2)
        X_train = rng.normal(size=(600, n_feat))
        y_train = (rng.random(600) < 0.3).astype(int)
        X_current = rng.normal(size=(1, n_feat))
        survivors = [{"ticker": "TEST", "flags": []}]

        with patch.object(
            ranker, "_build_dataset",
            return_value=(X_train, y_train, X_current, ["TEST"]),
        ):
            with patch.object(
                ranker, "_train_xgboost", side_effect=RuntimeError("model blew up")
            ):
                with patch.object(ranker, "_train_rf", return_value=np.array([0.8])):
                    with patch.object(ranker, "_train_lstm", return_value=None):
                        with patch.object(scanner.log, "error"):
                            out = ranker.rank(survivors, {}, training_universe={})

        self.assertEqual(ranker.degraded_models, ["XGBoost"])
        self.assertEqual(out[0]["ml_score_xgb"], 0.5)
        self.assertEqual(out[0]["ml_score_rf"], 0.8)
        self.assertAlmostEqual(out[0]["ml_ensemble_score"], 0.65)
        # A 0.5 from a dead model is the absence of a signal, so it is flagged
        # rather than presented as a neutral read on the name.
        self.assertIn("ML_DEGRADED:XGBoost", out[0]["flags"])

    def test_world_headlines_drop_vendor_related_tickers(self):
        ctx = scanner.WorldContext()
        ctx.headlines = [
            {"headline": "Fed holds rates amid tariff risk", "source": "wire"}
        ]
        dumped = ctx.to_dict()["headlines"][0]
        self.assertNotIn("related", dumped)
        self.assertFalse(dumped.get("related"))


if __name__ == "__main__":
    unittest.main()
