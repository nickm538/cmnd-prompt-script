from datetime import datetime, timedelta
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


if __name__ == "__main__":
    unittest.main()
