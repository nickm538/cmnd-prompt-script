"""Temporal validation, ensemble, execution-quote, and horizon regressions."""
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import new_stock_scanner_pipeline_claude_opus_41426 as scanner


class ModelIntegrityTests(unittest.TestCase):
    def test_missing_bars_do_not_extend_horizon_or_invent_next_open(self):
        sessions = scanner.XNYS_CALENDAR.sessions_in_range("2026-01-02", "2026-05-29")[:60]
        frame = pd.DataFrame({"Open": 100., "Close": 100.}, index=sessions)
        signal_day = sessions[5]
        frame.loc[sessions[25], "Close"] = 110.
        frame.loc[sessions[26], "Close"] = 200.
        # A missing intermediate observation must not move exit to the next bar.
        missing_middle = frame.drop(sessions[15])
        outcome = scanner.MLRanker._forward_session_returns(missing_middle, 20)
        self.assertAlmostEqual(outcome.loc[signal_day], .10)
        # No exact next-session open means no feasible historical outcome.
        missing_entry = frame.drop(sessions[6])
        outcome = scanner.MLRanker._forward_session_returns(missing_entry, 20)
        self.assertTrue(pd.isna(outcome.loc[signal_day]))

    def test_short_calendar_panel_cannot_fall_back_to_row_validation(self):
        ranker = scanner.MLRanker()
        ranker.training_dates_ = np.repeat(
            np.array(pd.bdate_range("2025-01-02", periods=60), dtype="datetime64[ns]"), 100
        )
        self.assertEqual(ranker._purged_walk_forward_splits(6000), [])
        ranker.training_dates_ = np.array([], dtype="datetime64[ns]")
        self.assertEqual(ranker._purged_walk_forward_splits(6000), [])

    def test_calibration_requires_dates_and_purges_whole_sessions(self):
        ranker = scanner.MLRanker()
        dates = np.repeat(np.array(pd.bdate_range("2025-01-02", periods=120), dtype="datetime64[ns]"), 7)
        labels = np.tile([0, 1, 0, 1, 0, 1, 0], 120)
        probabilities = np.where(labels, 0.8, 0.2)
        ranker._calibrate_probabilities("MissingDates", [probabilities], [labels], np.array([0.7]))
        self.assertEqual(ranker.validation_metrics["MissingDates"]["validation_status"], "unavailable")
        result = ranker._calibrate_probabilities("TEST", [probabilities], [labels], np.array([0.7]), [dates])
        metrics = ranker.validation_metrics["TEST"]
        self.assertEqual(metrics["validation_status"], "passed")
        unique = np.unique(dates)
        last_fit = np.searchsorted(unique, np.datetime64(metrics["calibration_last_signal_date"]))
        first_eval = np.searchsorted(unique, np.datetime64(metrics["evaluation_first_signal_date"]))
        self.assertGreaterEqual(first_eval - last_fit - 1, scanner.HOLDING_HORIZON_DAYS)
        self.assertEqual(metrics["evaluation_samples"], metrics["evaluation_sessions"] * 7)
        bins = metrics["reliability_bins"]
        self.assertEqual(sum(item["samples"] for item in bins), metrics["evaluation_samples"])
        self.assertTrue(all(0 <= item["mean_prediction"] <= 1 for item in bins))
        observed_positives = sum(item["samples"] * item["observed_frequency"] for item in bins)
        self.assertAlmostEqual(observed_positives, metrics["evaluation_samples"] * 3 / 7, places=3)
        self.assertTrue(0 <= result[0] <= 1)
        self.assertFalse(metrics["realized_trade_profitability_validated"])

    def test_many_rows_do_not_substitute_for_calibration_history(self):
        ranker = scanner.MLRanker()
        dates = np.repeat(np.array(pd.bdate_range("2025-01-02", periods=30), dtype="datetime64[ns]"), 100)
        labels = np.arange(len(dates)) % 2
        ranker._calibrate_probabilities("TEST", [np.where(labels, .8, .2)], [labels], np.array([.7]), [dates])
        self.assertEqual(ranker.validation_metrics["TEST"]["validation_status"], "unavailable")

    def test_invalid_model_outputs_degrade_instead_of_becoming_evidence(self):
        ranker = scanner.MLRanker()
        scores = ranker._safe_scores("Bad", lambda *_: np.array([np.inf]), np.zeros((300, 1)), np.zeros(300), np.zeros((1, 1)))
        np.testing.assert_array_equal(scores, [.5])
        self.assertIn("Bad", ranker.degraded_models)

    def test_exact_active_ensemble_has_its_own_oof_validation(self):
        ranker = scanner.MLRanker()
        dates = np.repeat(np.array(pd.bdate_range("2025-01-02", periods=160), dtype="datetime64[ns]"), 5)
        labels = np.arange(len(dates)) % 2
        ranker.training_dates_ = dates
        ranker.training_tickers_ = np.tile(np.array(list("ABCDE"), dtype=object), 160)

        def train(name, reverse=False):
            def run(*_):
                indices = np.arange(len(dates))
                if reverse:
                    indices = indices[::-1]
                ranker._oof_by_model[name] = {
                    "indices": indices, "dates": dates[indices], "labels": labels[indices],
                    "probabilities": np.where(labels[indices], .8, .2),
                }
                ranker._current_raw_by_model[name] = np.array([.8])
                ranker.validation_metrics[name] = {"validation_status": "passed", "calibrated": True, "reference_base_rate": .5}
                return np.array([.8])
            return run

        with patch.object(ranker, "_build_dataset", return_value=(np.zeros((len(dates), 10)), labels, np.zeros((1, 10)), ["TEST"])):
            with patch.object(ranker, "_train_xgboost", side_effect=train("XGBoost")):
                with patch.object(ranker, "_train_rf", side_effect=train("RandomForest", True)):
                    out = ranker.rank([{"ticker": "TEST", "flags": []}], {})
        self.assertEqual(ranker.validation_metrics["Ensemble"]["validation_status"], "passed")
        self.assertTrue(ranker.probabilities_calibrated)
        self.assertTrue(out[0]["ml_probability_usable"])
        self.assertGreater(out[0]["ml_probability_lift"], 0)

    def test_quality_uses_validated_lift_not_absolute_fifty_percent(self):
        base = {"panel_composite_score": 75, "rules_passed": 10, "return_20d": .1, "close_vs_sma50": .05, "ema20_vs_ema50": .04, "avg_dollar_volume": 50_000_000, "flags": []}
        useful = {**base, "ml_ensemble_score": .267, "ml_probability_usable": True, "ml_reference_base_rate": .204, "ml_probability_lift": .063}
        unavailable = {**base, "ml_ensemble_score": .5, "ml_probability_usable": False}
        useful_score = scanner.OptionsEvaluator._trade_setup_score(useful)
        unavailable_score = scanner.OptionsEvaluator._trade_setup_score(unavailable)
        self.assertGreater(useful_score, unavailable_score)
        self.assertEqual(unavailable["score_components"]["validated_ml_lift"], 13)
        self.assertAlmostEqual(sum(useful["score_components"].values()), useful_score, places=4)

    def test_missing_current_features_cannot_inherit_ensemble_validity(self):
        ranker = scanner.MLRanker()
        def valid_rf(*_):
            ranker.validation_metrics["RandomForest"] = {
                "validation_status": "passed", "calibrated": True,
                "reference_base_rate": .2,
            }
            return np.array([.8])
        candidates = [{"ticker": "GOOD", "flags": []}, {"ticker": "MISSING", "flags": []}]
        dataset = (np.zeros((300, 10)), np.arange(300) % 2, np.zeros((1, 10)), ["GOOD"])
        with patch.object(ranker, "_build_dataset", return_value=dataset), patch.object(ranker, "_train_xgboost", side_effect=scanner._ModelDegradedError("unavailable")), patch.object(ranker, "_train_rf", side_effect=valid_rf), patch.object(scanner, "ENABLE_EXPERIMENTAL_LSTM", False):
            result = ranker.rank(candidates, {})
        self.assertTrue(result[0]["ml_probability_usable"])
        self.assertFalse(result[1]["ml_probability_usable"])
        self.assertFalse(result[1]["ml_probability_calibrated"])
        self.assertIsNone(result[1]["ml_probability_lift"])
        self.assertIn("ML_CURRENT_FEATURES_UNAVAILABLE", result[1]["flags"])

    def test_option_availability_and_unsupported_hype_cannot_reorder_equities(self):
        base = {"panel_composite_score": 75, "rules_passed": 10, "ml_probability_usable": False, "flags": []}
        ordinary = scanner.OptionsEvaluator._trade_setup_score(dict(base))
        hyped = scanner.OptionsEvaluator._trade_setup_score({**base, "option_candidate": "Y", "option_score": 100, "hype_score": 100, "squeeze_score": 100, "insider_score": 100})
        self.assertEqual(ordinary, hyped)


class OptionIntegrityTests(unittest.TestCase):
    def setUp(self):
        scanner.reset_market_router()
        self.now = datetime(2026, 9, 29, 11, 0, tzinfo=scanner.ET_TZ)
        self.stamp = pd.Timestamp(self.now - timedelta(minutes=1)).tz_convert("UTC").value
        self.candidate = {"ticker": "TEST", "price": 100., "panel_composite_score": 80, "return_20d": .08, "rsi_14": 55}
        self.contract = {"expiry": "2026-11-13", "strike": 100., "bid": 4.9, "ask": 5.1, "mid": 999., "oi": 1000, "volume": 100, "iv": .15, "delta": .55, "theta": -.02, "break_even": 1., "underlying_price": 100., "underlying_timestamp": self.stamp, "underlying_timeframe": "REAL-TIME", "quote_timestamp": self.stamp, "quote_timeframe": "REAL-TIME", "contract_type": "call", "contract_symbol": "TEST261113C00100000", "shares_per_contract": 100, "adjusted": False}

    def evaluate(self, chain, now=None):
        with patch.object(scanner, "now_et_dt", return_value=now or self.now):
            with patch.object(scanner, "MBOUM_OPTIONS_KEY", ""):
                with patch.object(scanner.OptionsEvaluator, "_fetch_chain_massive", return_value=chain):
                    with patch.object(scanner.OptionsEvaluator, "_fetch_chain_yahoo", return_value=[]):
                        return scanner.OptionsEvaluator._find_best_option(self.candidate, {})

    def test_verified_call_uses_ask_cost_without_overwriting_real_iv(self):
        result = self.evaluate([self.contract])
        self.assertEqual(result["option_candidate"], "Y")
        self.assertEqual(result["option_mid"], 5.)
        self.assertEqual(result["option_break_even"], 105.1)
        self.assertEqual(result["option_max_loss_per_contract"], 510.)
        self.assertEqual(result["option_iv"], .15)

    def test_quote_metadata_is_required_even_with_both_sides(self):
        for changes in ({"quote_timestamp": None}, {"underlying_timestamp": None}, {"shares_per_contract": 50}, {"adjusted": True}, {"quote_timeframe": "UNKNOWN"}):
            with self.subTest(changes=changes):
                result = self.evaluate([{**self.contract, **changes}])
                self.assertNotEqual(result["option_candidate"], "Y")
                self.assertTrue(result["option_rejection_reason"])

    def test_expiration_must_outlive_trading_horizon(self):
        result = self.evaluate([{**self.contract, "expiry": "2026-10-23"}])
        self.assertEqual(result["option_candidate"], "N")
        self.assertIn("expires_before_thesis_and_buffer", result["option_rejection_counts"])

    def test_individually_fresh_mixed_delay_quotes_must_also_be_synchronized(self):
        delayed_underlying = pd.Timestamp(self.now - timedelta(minutes=16)).value
        mismatched = {**self.contract, "underlying_timestamp": delayed_underlying, "underlying_timeframe": "DELAYED"}
        self.assertEqual(scanner.OptionsEvaluator._quote_context(mismatched, self.now)["status"], "unsynchronized_quote_and_underlying")
        result = self.evaluate([mismatched])
        self.assertNotEqual(result["option_candidate"], "Y")
        self.assertTrue(result["option_rejection_reason"])
        synchronized = self.evaluate([{**self.contract, "quote_timestamp": delayed_underlying, "quote_timeframe": "DELAYED", "underlying_timestamp": delayed_underlying, "underlying_timeframe": "DELAYED"}])
        self.assertEqual(synchronized["option_candidate"], "Y")

    def test_stale_and_future_quotes_never_become_call_buys(self):
        for stamp in (pd.Timestamp(self.now - timedelta(days=7)).value, pd.Timestamp(self.now + timedelta(hours=1)).value):
            result = self.evaluate([{**self.contract, "quote_timestamp": stamp}])
            self.assertNotEqual(result["option_candidate"], "Y")

    def test_last_closed_session_quote_is_watch_only_before_open(self):
        now = datetime(2026, 9, 30, 8, 45, tzinfo=scanner.ET_TZ)
        stamp = pd.Timestamp(datetime(2026, 9, 29, 15, 59, tzinfo=scanner.ET_TZ)).value
        result = self.evaluate([{**self.contract, "quote_timestamp": stamp, "underlying_timestamp": stamp}], now)
        self.assertEqual(result["option_candidate"], "WATCH")
        self.assertFalse(result["option_actionable"])
        self.assertIn("requote_required", result["option_quote_status"])

    def test_one_malformed_contract_does_not_hide_later_valid_contract(self):
        result = self.evaluate([{**self.contract, "strike": "bad-price"}, self.contract])
        self.assertEqual(result["option_candidate"], "Y")

    def test_epoch_precision_is_not_confused(self):
        for unit, factor in (("s", 10**9), ("ms", 10**6), ("us", 10**3), ("ns", 1)):
            with self.subTest(unit=unit):
                stamp = scanner.OptionsEvaluator._timestamp_utc(self.stamp // factor)
                self.assertEqual(stamp.date(), self.now.date())


class ExitIntegrityTests(unittest.TestCase):
    def test_time_stop_matches_twentieth_entry_session_even_with_missing_bars(self):
        sessions = scanner.XNYS_CALENDAR.sessions_in_range("2026-08-03", "2026-09-30")[:20]
        frame = pd.DataFrame({"Close": 100., "EMA_20": 95., "EMA_10": 95., "RSI_14": 55.}, index=sessions[[0, -1]])
        result = scanner.SellMonitor.check_exits([{"ticker": "TEST", "entry_price": 100., "entry_date": str(sessions[0].date())}], {"TEST": frame})
        self.assertIn("SELL_03", result[0]["reason"])

    def test_profit_lock_remembers_peak_after_gain_falls_below_ten_percent(self):
        frame = pd.DataFrame({"Close": [112., 108.], "EMA_20": [100., 100.], "EMA_10": [110., 109.], "RSI_14": [55., 55.]}, index=pd.to_datetime(["2026-09-28", "2026-09-29"]))
        result = scanner.SellMonitor.check_exits([{"ticker": "TEST", "entry_price": 100., "entry_date": "2026-09-28"}], {"TEST": frame})
        self.assertIn("SELL_04", result[0]["reason"])

    def test_empty_data_requests_review_instead_of_crashing_or_ordering_sale(self):
        result = scanner.SellMonitor.check_exits([{"ticker": "TEST"}], {"TEST": pd.DataFrame()})
        self.assertEqual(result[0]["action"], "REVIEW DATA")


if __name__ == "__main__":
    unittest.main()
