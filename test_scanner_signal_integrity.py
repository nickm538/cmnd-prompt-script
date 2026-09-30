"""Numerical, provider-schema, and admission invariants; no live API requests."""
from datetime import date, datetime, timedelta
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import new_stock_scanner_pipeline_claude_opus_41426 as scanner


def _response(payload, status=200):
    response = Mock(status_code=status, text="plan unavailable" if status == 403 else "")
    response.json.return_value = payload
    return response


def _rule_frame():
    frame = pd.DataFrame({
        "Open": [100.] * 260, "High": [101.] * 260,
        "Low": [99.] * 260, "Close": [100.] * 260,
        "Volume": [200.] * 260, "Vol_SMA_20": [100.] * 260,
        "SMA_50": np.linspace(85., 90., 260), "SMA_200": [80.] * 260,
        "SMA_10": [98.] * 260, "SMA_30": [95.] * 260,
        "BB_upper": [102.] * 260, "High_Close_20": [99.] * 260,
        "MACD_line": [.2] * 260, "MACD_signal": [.1] * 260,
        "MACD_histogram": [.1] * 260, "MACD_hist_pct": [.001] * 260,
        "VWAP": [90.] * 260, "EMA_20": [94.] * 260,
        "EMA_50": [93.] * 260, "RSI_14": [55.] * 260,
        "Return_1d": [.01] * 260,
    }, index=pd.bdate_range(end="2026-09-28", periods=260))
    frame.iloc[-3, frame.columns.get_loc("SMA_10")] = 94.
    return frame


class SignalIntegrityTests(unittest.TestCase):
    def setUp(self):
        scanner.reset_market_router()

    def test_wilder_rsi_matches_published_reference_seed_and_recursion(self):
        # Wilder's common worked price series, checked against the defining
        # arithmetic seed + recursive smoothing rather than a second library.
        prices = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
                            45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                            46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21])
        delta = prices.diff().iloc[1:]
        gain = delta.clip(lower=0).iloc[:14].mean()
        loss = -delta.clip(upper=0).iloc[:14].mean()
        expected = [100 - 100 / (1 + gain / loss)]
        for change in delta.iloc[14:]:
            gain = (gain * 13 + max(change, 0)) / 14
            loss = (loss * 13 + max(-change, 0)) / 14
            expected.append(100 - 100 / (1 + gain / loss))
        actual = scanner.TechnicalEngine.rsi(prices)
        self.assertTrue(actual.iloc[:14].isna().all())
        np.testing.assert_allclose(actual.iloc[14:], expected, atol=1e-10)

    def test_rsi_flat_rising_and_falling_have_defined_distinct_values(self):
        for prices, expected in (([100.] * 40, 50.),
                                 (list(range(1, 41)), 100.),
                                 (list(range(40, 0, -1)), 0.)):
            self.assertEqual(scanner.TechnicalEngine.rsi(pd.Series(prices)).iloc[-1], expected)

    def test_atr_uses_wilder_smoothing_after_initial_seed(self):
        values = pd.Series([1.] * 14 + [15.])
        smoothed = scanner.TechnicalEngine.wilder_average(values)
        self.assertEqual(smoothed.iloc[13], 1.)
        self.assertAlmostEqual(smoothed.iloc[14], 2.)

    def test_strict_rules_and_report_cannot_disagree_about_missing_volume(self):
        frame = _rule_frame()
        self.assertTrue(scanner.HardBuyRules._evaluate("TEST", frame)[0])
        for value in (np.nan, np.inf, -1.):
            altered = frame.copy()
            altered.iloc[-1, altered.columns.get_loc("Volume")] = value
            result = scanner.HardBuyRules._evaluate_all_rules("TEST", altered)
            self.assertIn("BUY_09:Volume", result["failed_rules"])
            self.assertFalse(scanner.HardBuyRules._evaluate("TEST", altered)[0])

    def test_missing_daily_vwap_proxy_cannot_be_replaced_by_single_bar_price(self):
        frame = _rule_frame()
        frame.iloc[-1, frame.columns.get_loc("VWAP")] = np.nan
        frame.iloc[-2, frame.columns.get_loc("High")] = 90.
        frame.iloc[-2, frame.columns.get_loc("Low")] = 90.
        frame.iloc[-2, frame.columns.get_loc("Close")] = 90.
        self.assertIn("BUY_06:VWAP", scanner.HardBuyRules._evaluate_all_rules("TEST", frame)["failed_rules"])
        self.assertFalse(scanner.HardBuyRules._evaluate("TEST", frame)[0])

    def test_continuation_breakout_is_explained_without_loosening_buy_gate(self):
        frame = _rule_frame()
        frame["SMA_10"] = 98.
        result = scanner.HardBuyRules._evaluate_all_rules("TEST", frame)
        self.assertEqual(result["setup_archetype"], "TREND_CONTINUATION_BREAKOUT")
        self.assertTrue(result["strict_policy_excluded"])
        self.assertFalse(scanner.HardBuyRules._evaluate("TEST", frame)[0])

    def test_training_pool_does_not_delete_good_old_history_using_future_flat_tail(self):
        frame = _rule_frame()
        frame["Close"] = np.linspace(50., 100., len(frame))
        frame.loc[frame.index[-30:], "Close"] = 100.
        self.assertIn("TEST", scanner.ExecutionGuards.ml_training_pool({"TEST": frame}))

    def test_enriched_fundamentals_recover_from_stale_partial_quality(self):
        fetcher = scanner.FundamentalsFetcher("test-key")
        extra = {"name": "Example Corp", "sector": "Technology", "market_cap": 1e9,
                 "earnings_growth": .2, "revenue_growth": .15, "return_on_equity": .1,
                 "debt_to_equity": .5, "free_cash_flow": 1e6,
                 "insider_buy_transactions": 0., "insider_sell_transactions": 0.}
        with patch.object(fetcher, "_fundamentals_massive", return_value=extra):
            with patch.object(fetcher, "_fundamentals_twelvedata") as td:
                result = fetcher._enrich_fundamentals("TEST", {"name": "TEST", "sector": "Unknown", "fundamentals_quality": "partial"})
        self.assertEqual(result["name"], "Example Corp")
        self.assertEqual(result["fundamentals_quality"], "complete")
        self.assertGreater(result["fundamental_data_coverage"], .5)
        td.assert_not_called()

    def test_optional_finnhub_plan_failure_preserves_metrics_and_basics_circuit(self):
        session = Mock()
        session.get.side_effect = [
            _response({"name": "Example Corp", "finnhubIndustry": "Technology", "marketCapitalization": 1_000}),
            _response({"metric": {"netProfitMarginTTM": .5, "roeTTM": 1., "epsGrowthQuarterlyYoy": .5, "revenueGrowthQuarterlyYoy": 1.4}}),
            _response({}, 403),
        ]
        router = Mock(session=session)
        fetcher = scanner.FundamentalsFetcher("test-key")
        with patch.object(scanner, "FINNHUB_API_KEY", "test-key"):
            with patch.object(scanner, "get_market_router", return_value=router):
                result = fetcher._fundamentals_finnhub("TEST")
        self.assertEqual(result["market_cap"], 1e9)
        self.assertAlmostEqual(result["earnings_growth"], .005)
        self.assertAlmostEqual(result["revenue_growth"], .014)
        self.assertAlmostEqual(result["return_on_equity"], .01)
        self.assertAlmostEqual(result["profit_margin"], .005)
        self.assertTrue(scanner.ProviderCircuit.get("Finnhub-fundamentals").available())
        self.assertFalse(scanner.ProviderCircuit.get("Finnhub-insider").available())

    def test_growth_basis_follows_the_provider_that_supplied_the_value(self):
        fetcher = scanner.FundamentalsFetcher("test-key")
        result = fetcher._fill_missing_fundamentals(
            {"earnings_growth": np.nan, "earnings_growth_basis": "quarterly_yoy"},
            {"earnings_growth": .2, "earnings_growth_basis": "ttm_yoy"},
        )
        self.assertEqual(result["earnings_growth_basis"], "ttm_yoy")

    def test_twelvedata_documented_fields_and_float_denominator_are_used(self):
        session = Mock()
        session.get.side_effect = [
            _response({"name": "Example Corp", "sector": "Technology"}),
            _response({"statistics": {
                "financials": {"income_statement": {"quarterly_earnings_growth_yoy": .932},
                               "balance_sheet": {"total_debt_to_equity_mrq": 210.782},
                               "cash_flow": {"levered_free_cash_flow_ttm": 80625876992}},
                "stock_statistics": {"shares_short": 100, "float_shares": 1000,
                                     "shares_outstanding": 2000, "short_percent_of_shares_outstanding": .05},
            }}),
        ]
        fetcher = scanner.FundamentalsFetcher("test-key")
        with patch.object(scanner, "TWELVEDATA_API_KEY", "test-key"):
            with patch.object(scanner, "get_market_router", return_value=Mock(session=session)):
                result = fetcher._fundamentals_twelvedata("TEST")
        self.assertAlmostEqual(result["earnings_growth"], .932)
        self.assertAlmostEqual(result["debt_to_equity"], 2.10782)
        self.assertEqual(result["free_cash_flow"], 80625876992)
        self.assertAlmostEqual(result["short_pct_float"], .1)
        self.assertAlmostEqual(result["short_pct_outstanding"], .05)

    def test_insider_epoch_dates_exclude_old_future_and_nonmarket_trades(self):
        today = date(2026, 9, 29)
        epoch = int(datetime(2026, 9, 28).timestamp())
        rows = [
            {"startDate": {"raw": epoch}, "transactionText": "Purchase", "shares": 100},
            {"transactionDate": "2026-09-27", "transactionCode": "S", "change": -50},
            {"transactionDate": "2024-01-01", "transactionCode": "P", "change": 999},
            {"transactionDate": "2026-09-30", "transactionCode": "P", "change": 999},
            {"transactionDate": "2026-09-28", "transactionCode": "A", "transactionText": "Award buy", "change": 999},
        ]
        with patch.object(scanner, "today_et", return_value=today):
            result = scanner.FundamentalsFetcher._summarize_insider_transactions(rows)
        self.assertEqual(result["insider_buy_transactions"], 1.)
        self.assertEqual(result["insider_sell_transactions"], 1.)
        self.assertEqual(result["insider_net_shares"], 50.)
        self.assertEqual(result["insider_data_asof"], "2026-09-28")

    def test_single_insider_trade_does_not_manufacture_full_conviction(self):
        self.assertAlmostEqual(scanner.MomentumQuality.insider_score({"insider_buy_transactions": 1, "insider_sell_transactions": 0}), 100 * 2 / 3)
        self.assertFalse(scanner.MomentumQuality.evidence(pd.Series(dtype=float), {})["insider_evidence_available"])
        self.assertEqual(scanner.MomentumQuality.squeeze_score({"short_pct_float": .005}), 2.5)

    def test_verified_fund_product_risk_excludes_short_duration_false_positive(self):
        self.assertEqual(scanner.FundamentalsFetcher.instrument_risk_flags({"name": "Daily 3x Bull ETF", "quote_type": "ETF"}), ["DAILY_RESET_LEVERAGE_RISK"])
        self.assertEqual(scanner.FundamentalsFetcher.instrument_risk_flags({"name": "UltraShort Treasury ETF", "quote_type": "ETF"}), ["DAILY_RESET_LEVERAGE_RISK"])
        self.assertEqual(scanner.FundamentalsFetcher.instrument_risk_flags({"name": "Short Duration Bond ETF", "quote_type": "ETF"}), [])
        self.assertEqual(scanner.FundamentalsFetcher.instrument_risk_flags({"name": "Ultra Company", "quote_type": "EQUITY"}), [])

    def test_yfinance_verified_quote_type_and_shareclass_alias_survive_enrichment(self):
        tk = Mock()
        tk.info = {"longName": "Example ETF", "quoteType": "ETF"}
        tk.calendar = {}
        tk.insider_transactions = pd.DataFrame()
        with patch.object(scanner.yf, "Ticker", return_value=tk) as ticker:
            result = scanner.FundamentalsFetcher("test-key")._fundamentals_yfinance("BRK.B")
        ticker.assert_called_once_with("BRK-B")
        self.assertEqual(result["quote_type"], "ETF")
        self.assertEqual(result["quote_type_source"], "yfinance")

    def test_benchmark_return_matches_dates_and_exact_63_session_horizon(self):
        panel = scanner.InvestorPanel()
        index = pd.bdate_range(end="2026-09-28", periods=130)
        frame = pd.DataFrame({"Close": np.linspace(100., 130., 130)}, index=index)
        panel.benchmark_data = frame.copy()
        panel.benchmark_data.loc[pd.Timestamp("2026-09-29"), "Close"] = 10_000.
        self.assertAlmostEqual(panel._benchmark_excess(frame, 63), 0.)
        panel.benchmark_data = frame.iloc[:-1].copy()
        self.assertIsNone(panel._benchmark_excess(frame, 63))

    def test_cross_sectional_percentile_ties_are_neutral_and_state_is_reset(self):
        panel = scanner.InvestorPanel()
        panel._rs_universe_returns = np.full(100, .1)
        self.assertEqual(panel.rs_percentile(.1), 50.)
        panel.set_universe_returns({})
        self.assertIsNone(panel._rs_universe_returns)


if __name__ == "__main__":
    unittest.main()
