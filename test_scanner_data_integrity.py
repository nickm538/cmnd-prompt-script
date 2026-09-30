"""Regression cases for provider recovery, coverage and credential containment."""

from datetime import date, timedelta
import json
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import new_stock_scanner_pipeline_claude_opus_41426 as scanner


SESSION = date(2026, 4, 29)


def history(end=SESSION, count=260):
    idx = pd.bdate_range(end=end, periods=count)
    close = np.linspace(50.0, 70.0, count)
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": np.full(count, 100_000),
    }, index=idx)


class DataIntegrityTests(TestCase):
    def setUp(self):
        scanner.reset_market_router()

    def test_successful_month_fields_and_quota_words_are_not_api_errors(self):
        payloads = [
            {"meta": {"status": 200, "shortName": "1-3 Month Treasury"}, "body": {}},
            {"body": {"priceToSalesTrailing12Months": 3.5}},
            {"body": [{"averageDailyVolume3Month": 100_000}]},
            {"results": [{"description": "Credit Limit and Quota Solutions"}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(scanner.classify_http_error(200, json.dumps(payload)), "other")

    def test_mboum_valid_data_never_trips_history_fundamental_or_option_circuit(self):
        payload = {"meta": {"status": 200}, "body": {"averageDailyVolume3Month": 100_000}}
        response = Mock(status_code=200, text=json.dumps(payload))
        response.json.return_value = payload
        api = scanner.MboumAPI("test-key")
        for name in ["MBOUM-OHLCV", "MBOUM-Fundamentals", "MBOUM-Options"]:
            circuit = scanner.ProviderCircuit.get(name)
            with self.subTest(provider=name):
                self.assertEqual(api._decode_payload(response, circuit, "test"), payload)
                self.assertTrue(circuit.available())

    def test_explicit_quota_rate_and_auth_envelopes_remain_provider_errors(self):
        cases = [
            ({"error": "API credits exhausted this month"}, "credit"),
            ({"meta": {"status": 429, "message": "Limit reached for the current minute"}}, "rate"),
            ({"meta": {"status": 403}}, "auth"),
            ({"body": "API credits exhausted this month"}, "credit"),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(scanner.classify_http_error(200, json.dumps(payload)), expected)

    def test_month_word_in_symbol_miss_does_not_mean_exhausted_credits(self):
        for status in [200, 404]:
            self.assertEqual(scanner.classify_http_error(status, "No data for the current month"), "other")

    def test_safe_div_is_defined_and_rejects_missing_or_nonfinite_values(self):
        self.assertEqual(scanner.safe_div(6, 3), 2)
        for numerator, denominator in [(None, 3), (3, 0), (3, np.nan), (np.inf, 3), ("x", 3)]:
            self.assertEqual(scanner.safe_div(numerator, denominator, default=1), 1)

    def test_ohlcv_normalizer_rejects_impossible_and_infinite_bars(self):
        frame = history(count=6)
        frame.iloc[1, frame.columns.get_loc("High")] = 1
        frame.iloc[2, frame.columns.get_loc("Open")] = np.inf
        frame.iloc[3, frame.columns.get_loc("Low")] = -1
        frame.iloc[4, frame.columns.get_loc("Volume")] = np.nan
        normalized = scanner._normalize_ohlcv_frame(frame)
        self.assertEqual(list(normalized.index), [frame.index[0], frame.index[-1]])

    @patch.object(scanner, "expected_last_closed_trading_day", return_value=SESSION)
    def test_stale_primary_history_falls_through_to_current_secondary(self, _expected):
        router = scanner.MarketDataRouter()
        fresh = history()
        with patch.object(router, "_provider_enabled", return_value=True), \
             patch.object(router, "_history_mboum", return_value=history(SESSION - timedelta(days=1))), \
             patch.object(router, "_history_massive", return_value=fresh) as secondary:
            frame, source = router.get_history("TEST")
        self.assertIs(frame, fresh)
        self.assertEqual(source, "Massive")
        secondary.assert_called_once_with("TEST")

    @patch.object(scanner, "expected_last_closed_trading_day", return_value=SESSION)
    def test_unfinalized_primary_bar_is_trimmed_before_provider_acceptance(self, _expected):
        router = scanner.MarketDataRouter()
        frame = history(SESSION + timedelta(days=1))
        with patch.object(router, "_provider_enabled", return_value=True), \
             patch.object(router, "_history_mboum", return_value=frame), \
             patch.object(router, "_history_massive") as secondary:
            loaded, source = router.get_history("TEST")
        self.assertEqual(loaded.index[-1].date(), SESSION)
        self.assertEqual(source, "MBOUM")
        secondary.assert_not_called()

    @patch.object(scanner, "expected_last_closed_trading_day", return_value=SESSION)
    def test_short_primary_history_uses_full_secondary_without_provider_trip(self, _expected):
        router = scanner.MarketDataRouter()
        with patch.object(router, "_provider_enabled", return_value=True), \
             patch.object(router, "_history_mboum", return_value=history(count=100)), \
             patch.object(router, "_history_massive", return_value=history()):
            _, source = router.get_history("TEST")
        self.assertEqual(source, "Massive")
        self.assertTrue(scanner.ProviderCircuit.get("MBOUM-OHLCV").available())

    def test_yahoo_quote_misses_continue_to_history_chain(self):
        fetcher = scanner.DataFetcher()
        liquid = {"last_close": 20.0, "avg_volume_20d": 200_000}
        quotes = {"GOOD": liquid, "MISSING": None, "CHEAP": {**liquid, "last_close": 2.0},
                  "THIN": {**liquid, "avg_volume_20d": 10}}
        with patch.object(fetcher.yahoo, "quick_quote", side_effect=lambda symbol: quotes[symbol]):
            result = fetcher._quick_screen(list(quotes))
        self.assertEqual(result, ["GOOD", "MISSING"])
        coverage = fetcher.coverage["quick_screen"]
        self.assertEqual(coverage["retained_unverified"], 1)
        self.assertEqual(coverage["excluded_price"], 1)
        self.assertEqual(coverage["excluded_liquidity"], 1)
        self.assertEqual(coverage["unavailable_tickers"], ["MISSING"])

    def test_total_quick_quote_outage_preserves_every_discovered_symbol(self):
        fetcher = scanner.DataFetcher()
        symbols = [f"T{i}" for i in range(600)]
        with patch.object(fetcher.yahoo, "quick_quote", return_value=None):
            retained = fetcher._quick_screen(symbols)
        self.assertEqual(set(retained), set(symbols))
        self.assertEqual(fetcher.coverage["quick_screen"]["quote_coverage_ratio"], 0)

    def test_actual_average_dollar_volume_controls_prescreen(self):
        fetcher = scanner.DataFetcher()
        quote = {"last_close": 100.0, "avg_volume_20d": 100_000,
                 "avg_dollar_volume_20d": 900_000.0}
        with patch.object(fetcher.yahoo, "quick_quote", return_value=quote):
            self.assertEqual(fetcher._quick_screen(["TEST"]), [])
        self.assertEqual(fetcher.coverage["quick_screen"]["excluded_liquidity"], 1)

    @patch.object(scanner, "expected_last_closed_trading_day", return_value=SESSION)
    def test_large_universe_cannot_bypass_full_history_coverage_gate(self, _expected):
        fetcher = scanner.DataFetcher()
        frame = history()
        symbols = [f"T{i}" for i in range(1_000)]
        with patch.object(fetcher.router, "get_history", side_effect=lambda symbol: (
            (frame, "Mock") if int(symbol[1:]) < 500 else (None, "")
        )), patch.object(fetcher.router, "yfinance_batch", return_value={}):
            with self.assertRaisesRegex(scanner.PipelineError, "coverage too low"):
                fetcher._full_download(symbols)
        coverage = fetcher.coverage["full_history"]
        self.assertEqual(coverage["loaded"], 500)
        self.assertEqual(coverage["coverage_ratio"], 0.5)
        self.assertEqual(coverage["unavailable"], 500)
        self.assertTrue(coverage["complete"])

    def test_budget_abort_cancels_bounded_queue_without_waiting_for_universe(self):
        fetcher = scanner.DataFetcher()
        executor = Mock()
        futures = []

        def submit(*args):
            future = Mock()
            futures.append(future)
            return future

        executor.submit.side_effect = submit
        # One initial check, four submissions (2 * two workers), then abort
        # before waiting. No 10,000-item queue is ever submitted or drained.
        checks = [None] * 5 + [scanner.PipelineBudgetExceeded("test deadline")]
        with patch.object(scanner, "ThreadPoolExecutor", return_value=executor), \
             patch.object(fetcher, "_check_budget_reserve", side_effect=checks):
            with self.assertRaises(scanner.PipelineBudgetExceeded):
                list(fetcher._parallel_results(list(range(10_000)), Mock(), 2, "test"))
        self.assertEqual(len(futures), 4)
        self.assertTrue(all(future.cancel.called for future in futures))
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_pipeline_clock_rejects_nonfinite_or_nonpositive_budgets(self):
        for budget in [float("inf"), float("nan"), 0, -1]:
            with self.assertRaises(ValueError):
                scanner.PipelineClock(budget)

    def test_partial_nasdaq_listing_files_are_not_a_complete_universe(self):
        discovery = scanner.UniverseDiscovery("")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        body = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        body += "\n".join(
            f"{alphabet[i // 26]}{alphabet[i % 26]}Q|Example - Common Stock|Q|N|N|100|N|N"
            for i in range(600)
        )
        good = Mock(text=body)
        good.raise_for_status.return_value = None
        bad = Mock()
        bad.raise_for_status.side_effect = RuntimeError("other exchange file unavailable")
        with patch.object(discovery.session, "get", side_effect=[good, bad]):
            self.assertEqual(discovery._discover_nasdaq_listings(), [])

    def test_massive_pagination_loop_discards_incomplete_prefix(self):
        discovery = scanner.UniverseDiscovery("test-key")
        first = Mock()
        first.raise_for_status.return_value = None
        first.json.return_value = {
            "results": [{"ticker": "TEST", "type": "CS", "primary_exchange": "XNAS"}],
            "next_url": "https://api.massive.com/v3/reference/tickers",
        }
        with patch.object(discovery.session, "get", return_value=first), \
             patch.object(discovery, "_discover_without_massive", return_value=["RECOVERED"]) as fallback:
            self.assertEqual(discovery.discover(), ["RECOVERED"])
        fallback.assert_called_once()

    def test_adr_and_iex_equities_are_part_of_discovered_universe(self):
        cleaned = scanner.UniverseDiscovery._clean_listed_equities([
            {"ticker": "TM", "type": "ADRC", "primary_exchange": "XNYS"},
            {"ticker": "TEST", "type": "CS", "primary_exchange": "IEXG"},
        ])
        self.assertEqual(cleaned, ["TEST", "TM"])
        text = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\nTEST|Example Common Stock|V|TEST|N|100|N|TEST"
        rows = scanner.UniverseDiscovery._parse_nasdaq_listing_text(text, "other")
        self.assertEqual(rows[0]["primary_exchange"], "IEXG")

    def test_universe_session_does_not_forward_provider_key_to_listing_host(self):
        discovery = scanner.UniverseDiscovery("secret-for-massive")
        self.assertNotIn("X-Massive-Token", discovery.session.headers)
        self.assertNotIn("Authorization", discovery.session.headers)

    def test_provider_error_text_redacts_values_and_query_credentials(self):
        with patch.object(scanner, "MASSIVE_API_KEY", "private-key-value"):
            result = scanner.redact_sensitive_text(
                "HTTP 403 https://api.test/path?apiKey=private-key-value&token=other-value "
                "MASSIVE_API_KEY absent"
            )
        self.assertNotIn("private-key-value", result)
        self.assertNotIn("other-value", result)
        self.assertIn("MASSIVE_API_KEY absent", result)

    @patch.object(scanner, "_history_window", return_value=(date(2021, 4, 29), SESSION))
    def test_massive_denied_deep_history_retries_entitled_recent_window(self, _window):
        router = scanner.MarketDataRouter()
        denied = Mock(status_code=403, text="You are not authorized for this timeframe")
        current = Mock(status_code=200, text="")
        current.json.return_value = {"results": [{
            "t": pd.Timestamp("2026-04-29", tz="America/New_York").value // 1_000_000,
            "o": 50, "h": 51, "l": 49, "c": 50, "v": 100_000,
        }]}
        with patch.object(router.session, "get", side_effect=[denied, current]) as request:
            frame = router._history_massive("TEST")
        self.assertIsNotNone(frame)
        self.assertEqual(request.call_count, 2)
        self.assertIn("2021-04-29", request.call_args_list[0].args[0])
        self.assertNotIn("2021-04-29", request.call_args_list[1].args[0])
        self.assertTrue(scanner.ProviderCircuit.get("Massive-OHLCV").available())

    @patch.object(scanner, "_history_window", return_value=(date(2021, 4, 29), SESSION))
    def test_finnhub_utc_midnight_daily_label_does_not_shift_to_previous_day(self, _window):
        router = scanner.MarketDataRouter()
        idx = pd.bdate_range(end=SESSION, periods=260, tz="UTC")
        response = Mock(status_code=200, text="")
        response.json.return_value = {
            "s": "ok", "t": [int(stamp.timestamp()) for stamp in idx],
            "o": [50] * 260, "h": [51] * 260, "l": [49] * 260,
            "c": [50] * 260, "v": [100_000] * 260,
        }
        with patch.object(router.session, "get", return_value=response):
            frame = router._history_finnhub("TEST")
        self.assertEqual(frame.index[-1].date(), SESSION)

    def test_yfinance_price_first_batch_selects_only_requested_symbol(self):
        first = history(count=3)
        second = first.copy()
        second.loc[:, ["Open", "High", "Low", "Close"]] *= 2
        batch = pd.concat({"FIRST": first, "SECOND": second}, axis=1)
        batch = batch.swaplevel(0, 1, axis=1)
        loaded = scanner._frame_from_yfinance(batch, "SECOND")
        pd.testing.assert_frame_equal(loaded, second, check_freq=False)
        self.assertIsNone(scanner._frame_from_yfinance(batch, "ABSENT"))

    def test_mboum_history_adjusts_full_ohlc_on_the_same_price_basis(self):
        api = scanner.MboumAPI("test-key")
        frame = history()
        body = {
            str(index): {
                "date": stamp.isoformat(), "open": row.Open,
                "high": row.High, "low": row.Low,
                "close": row.Close, "adjclose": row.Close * 0.5,
                "volume": row.Volume,
            }
            for index, (stamp, row) in enumerate(frame.iterrows())
        }
        response = Mock(status_code=200, text="")
        response.json.return_value = {"body": body}
        with patch.object(api.session, "get", return_value=response):
            loaded = api.get_history("TEST")
        self.assertEqual(len(loaded), len(frame))
        for column in ["Open", "High", "Low", "Close"]:
            np.testing.assert_allclose(loaded[column], frame[column] * 0.5)
        np.testing.assert_allclose(loaded["Volume"], frame["Volume"])

    def test_empty_provider_payload_is_not_a_verified_empty_event_calendar(self):
        response = Mock(status_code=200, text="{}")
        response.json.return_value = {}
        session = Mock()
        session.get.return_value = response
        ctx = scanner.WorldContext()
        with patch.object(scanner, "FINNHUB_API_KEY", "test-key"):
            ctx._load_earnings(session)
            success = ctx._load_economic_calendar_finnhub(session)
            ctx._load_market_status(session)
        self.assertNotEqual(ctx.feed_status["earnings_calendar"], "ok")
        self.assertFalse(success)
        self.assertNotEqual(ctx.feed_status["economic_calendar"], "ok")
        self.assertNotEqual(ctx.feed_status["market_status"], "ok")

    def test_share_classes_keep_canonical_identity_and_provider_aliases(self):
        rows = [
            {"ticker": "BRK.A", "type": "CS", "primary_exchange": "XNYS"},
            {"ticker": "BRK-B", "type": "Common Stock", "primary_exchange": "XNYS"},
            {"ticker": "TEST.A", "type": "Warrant", "primary_exchange": "XNYS"},
            {"ticker": "TEST.B", "type": "CS", "primary_exchange": "XNYS", "name": "Example - Warrants"},
            {"ticker": "PREF.A", "type": "PS", "primary_exchange": "XNYS"},
            {"ticker": "UNKN.A", "primary_exchange": "XNYS"},
        ]
        self.assertEqual(scanner.UniverseDiscovery._clean_listed_equities(rows), ["BRK.A", "BRK.B"])
        self.assertEqual(scanner.provider_symbol("BRK.B", "Yahoo"), "BRK-B")
        self.assertEqual(scanner.provider_symbol("BRK.A", "yfinance"), "BRK-A")
        self.assertEqual(scanner.provider_symbol("BRK.B", "Massive"), "BRK.B")
        self.assertEqual(scanner.provider_symbol("DX-Y.NYB", "Yahoo"), "DX-Y.NYB")

    def test_yahoo_class_quote_uses_vendor_alias(self):
        api = scanner.YahooDirectAPI()
        response = Mock(status_code=404, text="not found")
        with patch.object(api.session, "get", return_value=response) as request:
            api.quick_quote("BRK.B")
        self.assertTrue(request.call_args.args[0].endswith("/BRK-B"))

    def test_yfinance_class_batch_maps_back_to_canonical_symbol(self):
        router = scanner.MarketDataRouter()
        frame = history()
        raw = pd.concat({"BRK-B": frame}, axis=1)
        with patch.object(scanner, "expected_last_closed_trading_day", return_value=SESSION), \
             patch.object(scanner.yf, "download", return_value=raw) as download:
            result = router.yfinance_batch(["BRK.B"])
        self.assertEqual(list(result), ["BRK.B"])
        self.assertEqual(download.call_args.args[0], ["BRK-B"])

    def test_preferred_bank_name_is_not_misclassified_as_preferred_stock(self):
        text = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\nPFBC|Preferred Bank - Common Stock|Q|N|N|100|N|N"
        rows = scanner.UniverseDiscovery._parse_nasdaq_listing_text(text, "nasdaq")
        self.assertEqual(scanner.UniverseDiscovery._clean_listed_equities(rows), ["PFBC"])

    def test_macro_news_omits_law_firm_solicitation_but_keeps_legal_catalyst(self):
        advertising = "SHAREHOLDER ALERT: Law Offices invites investors to join class action"
        factual = "Company settles antitrust lawsuit for $50 million"
        self.assertTrue(scanner.WorldContext._is_promotional_solicitation(advertising))
        self.assertFalse(scanner.WorldContext._is_promotional_solicitation(factual))
        self.assertIn("LEGAL", scanner.WorldContext._classify_catalysts([factual]))
        ctx = scanner.WorldContext()
        self.assertFalse(ctx.to_dict()["news_is_comprehensive"])
