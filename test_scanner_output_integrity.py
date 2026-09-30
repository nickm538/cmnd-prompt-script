"""Regression cases for output/action safety, all offline."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import new_stock_scanner_pipeline_claude_opus_41426 as scanner


class OutputIntegrityTests(unittest.TestCase):
    def candidate(self):
        return {
            "ticker": "TEST", "price": 100.0, "flags": [],
            "hard_buy_pass": True, "rules_passed": 10, "rules_failed": 0,
            "panel_composite_score": 85.0, "panel_consensus": 5,
            "trade_setup_score": 85.0, "setup_quality_score": 85.0,
            "overall_confidence_score": 85.0, "ml_ensemble_score": 0.7,
            "ml_probability_usable": True, "ml_probability_lift": 0.15,
            "fundamentals_quality": "complete", "option_candidate": "N",
        }

    def test_high_score_cannot_override_unknown_earnings_or_daily_reset_risk(self):
        for flag, expected in [
            ("EARNINGS_DATE_UNVERIFIED", "WAIT: CALENDAR GAP"),
            ("DAILY_RESET_LEVERAGE_RISK", "WATCH: DAILY RESET FUND"),
        ]:
            row = self.candidate()
            row.update(flags=[flag], option_candidate="Y", option_score=99)
            self.assertEqual(scanner.OutputFormatter.recommended_action(row), expected)

    def test_unvalidated_or_below_base_rate_ml_cannot_be_actionable(self):
        row = self.candidate()
        row["ml_probability_usable"] = False
        self.assertEqual(scanner.OutputFormatter.recommended_action(row), "WATCH: ML UNVALIDATED")
        row["ml_probability_usable"] = True
        for lift in (-0.05, 0.0):
            row["ml_probability_lift"] = lift
            self.assertEqual(scanner.OutputFormatter.recommended_action(row), "WATCH: NO ML LIFT")

    def test_missing_or_nonfinite_risk_inputs_never_create_shares(self):
        for close, atr, scalar in [(np.nan, 3, .5), (100, np.inf, .5), (100, 0, .5), (100, 3, np.nan)]:
            self.assertEqual(scanner.build_risk_plan(close, atr, 95, scalar), {})

    def test_position_plan_is_fundable_and_explicit_about_gap_risk(self):
        # Tight support previously allowed sizing larger than the account.
        plan = scanner.build_risk_plan(100, 3, 99.99, .8)
        self.assertLessEqual(plan["position_value_per_10k"], 10000 * scanner.MAX_POSITION_PCT)
        self.assertFalse(plan["risk_limit_is_guaranteed"])
        self.assertIn("not_forecast", plan["target_kind"])
        self.assertIn("actual_entry", plan["sizing_basis"])

    def test_no_trade_with_no_diagnostics_still_produces_readable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(scanner, "OUTPUT_DIR", Path(directory)), contextlib.redirect_stdout(io.StringIO()):
            scanner.OutputFormatter.save_near_misses([], {}, diagnostics={"data_coverage": {"complete": True}})
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
            csv = pd.read_csv(next(Path(directory).glob("*.csv")))
            self.assertEqual(report["report_kind"], "no_action")
            self.assertFalse(report["near_misses"])
            self.assertIn("ticker", csv.columns)
            self.assertTrue(report["scan_diagnostics"]["data_coverage"]["complete"])

    def test_reporting_is_repeatable_and_retains_numeric_missingness(self):
        candidate = self.candidate()
        candidate.update(flags=["INFO"], insider_score=np.nan, option_iv=np.inf,
                         score_components={"panel": np.float64(28.9)})
        with tempfile.TemporaryDirectory() as directory, patch.object(scanner, "OUTPUT_DIR", Path(directory)), patch.object(scanner.OutputFormatter, "_print_results"):
            for _ in range(2):
                scanner.OutputFormatter.format_and_save([candidate], {}, {}, {}, diagnostics={"data_coverage": {"unavailable": 3}})
            text = next(Path(directory).glob("*.json")).read_text()
            report = json.loads(text, parse_constant=lambda value: self.fail("Invalid JSON numeric literal: " + value))
        self.assertEqual(candidate["flags"], ["INFO"])
        self.assertNotIn("recommended_action", candidate)
        self.assertIsNone(report["top_25"][0]["insider_score"])
        self.assertEqual(report["top_25"][0]["score_components"]["panel"], 28.9)
        self.assertEqual(report["scan_diagnostics"]["data_coverage"]["unavailable"], 3)

    def test_json_export_handles_numpy_arrays_and_missing_scalars(self):
        value = scanner.json_safe({"values": np.array([1., np.inf, np.nan]), "missing": pd.NA, "boolean": np.bool_(True)})
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), {"values": [1., None, None], "missing": None, "boolean": True})

    def test_new_high_has_no_fabricated_panel_upside(self):
        frame = pd.DataFrame({"Close": np.linspace(90, 100, 70), "SMA_50": 95., "OBV": np.arange(70)})
        panel = scanner.InvestorPanel()
        distant_support = panel._score_druckenmiller("TEST", frame, {})
        frame["SMA_50"] = 99.99
        tight_support = panel._score_druckenmiller("TEST", frame, {})
        self.assertEqual(distant_support, tight_support)


if __name__ == "__main__":
    unittest.main()
