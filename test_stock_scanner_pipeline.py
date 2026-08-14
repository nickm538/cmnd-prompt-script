from datetime import datetime, timedelta
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

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

    def test_macro_regime_requires_live_required_symbols(self):
        macro = scanner.MacroRegime()
        with patch.object(macro, "_series", return_value=None):
            with self.assertRaisesRegex(scanner.PipelineError, "Macro regime unavailable"):
                macro.load()

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
        self.assertTrue(circuit.available())
        circuit.record_failure("credits exhausted", credit=True)
        self.assertFalse(circuit.available())
        scanner.ProviderCircuit.reset_all()
        self.assertTrue(scanner.ProviderCircuit.get("MBOUM-OHLCV", fail_limit=4).available())

    def test_missing_mboum_key_is_optional_when_massive_is_present(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": "massive-test-key"}, clear=True):
            status = scanner.verify_api_credentials()
        self.assertEqual(status["MASSIVE_API_KEY"], "env")
        self.assertEqual(status["MBOUM_API_KEY"], "absent")
        self.assertEqual(status["TWELVEDATA_API_KEY"], "embedded")
        self.assertEqual(status["FINNHUB_API_KEY"], "embedded")

    def test_committed_fallback_keys_are_used_when_secrets_are_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            status = scanner.verify_api_credentials()
            massive = scanner._env_or_default(
                "MASSIVE_API_KEY", "yGJVMwH5maQwB5mTKqvEpiJpsz5t7g4H"
            )
            twelve = scanner._env_or_default(
                "TWELVEDATA_API_KEY", "5e7a5daaf41d46a8966963106ebef210"
            )
            finnhub = scanner._env_or_default(
                "FINNHUB_API_KEY", "d55b3ohr01qljfdeghm0d55b3ohr01qljfdeghmg"
            )
        self.assertEqual(status["MASSIVE_API_KEY"], "embedded")
        self.assertEqual(status["TWELVEDATA_API_KEY"], "embedded")
        self.assertEqual(status["FINNHUB_API_KEY"], "embedded")
        self.assertTrue(massive)
        self.assertTrue(twelve)
        self.assertTrue(finnhub)

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
