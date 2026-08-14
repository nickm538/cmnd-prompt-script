#!/usr/bin/env python3
"""
===============================================================================
  Stock Universe Scan Pipeline -- Nick's Live Trading System
  ─────────────────────────────────────────────────────────
  Owner:   Nick -- Data Analyst, real-capital trader
  Purpose: Identify the strongest buy opportunities before they happen
  Capital: Real money -- zero tolerance for hallucinated data or shortcuts

  STAGES:
    1. Universe Discovery   (live exchange listings -- never a preset basket)
    2. Execution Guards     (data integrity, liquidity, spread, 3mo perf)
    3. Hard Buy Rules       (ALL 10 must pass)
    4. ML Ranking           (XGBoost + Random Forest + optional LSTM)
    5. 5-Investor Panel     (Livermore, Druckenmiller, Lynch, Minervini, O'Neil)
    6. Options Evaluation   (long calls, 0.35-0.50 delta, 14-45 DTE)

  World context (live headlines + earnings calendar) annotates the regime and
  survivors. It does not choose the universe.

  Run:  python new_stock_scanner_pipeline_claude_opus_41426.py
===============================================================================
"""

import os
import sys
import time
import json
import logging
import math
import warnings
import traceback
import threading
from collections import Counter
from datetime import datetime, date, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import yfinance as yf

ET_TZ = ZoneInfo("US/Eastern")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE IDENTITY -- surfaced in logs, JSON report and the Actions job summary so
# a manual "Run workflow" from the Actions tab proves which engine executed.
# ═══════════════════════════════════════════════════════════════════════════════

ENGINE_NAME = "Claude Opus 5 Live Scanner Engine"
ENGINE_VERSION = "5.3.1"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION -- API credentials prefer the environment, then committed
# fallbacks so GitHub Actions can run when a secret is unset. A non-empty
# env var / repository secret always wins.
# ═══════════════════════════════════════════════════════════════════════════════

def _env_or_default(name: str, default: str = "") -> str:
    """Use a live env/secret when present; otherwise the committed fallback."""
    value = os.environ.get(name, "").strip()
    return value if value else default


# Committed fallback keys, in one table so credential reporting can tell a live
# secret apart from a fallback. MBOUM (both tiers) and AlphaVantage have no
# fallback on purpose -- MBOUM is the credit-metered primary, and running it on
# a shared committed key would burn the plan the whole cascade depends on.
EMBEDDED_FALLBACK_KEYS: Dict[str, str] = {
    "MASSIVE_API_KEY": "yGJVMwH5maQwB5mTKqvEpiJpsz5t7g4H",
    "FINNHUB_API_KEY": "d55b3ohr01qljfdeghm0d55b3ohr01qljfdeghmg",
    "TWELVEDATA_API_KEY": "5e7a5daaf41d46a8966963106ebef210",
}


def _resolve_api_key(name: str) -> str:
    """Live env/secret first, then the committed fallback for that key."""
    return _env_or_default(name, EMBEDDED_FALLBACK_KEYS.get(name, ""))


MASSIVE_API_KEY = _resolve_api_key("MASSIVE_API_KEY")
ALPHAVANTAGE_API_KEY = _resolve_api_key("ALPHAVANTAGE_API_KEY")
MBOUM_API_KEY = _resolve_api_key("MBOUM_API_KEY")
# Options-tier MBOUM key (different plan that includes the /v1/markets/options
# endpoint), kept separate from the standard MBOUM key.
MBOUM_OPTIONS_KEY = _resolve_api_key("MBOUM_OPTIONS_KEY")
FINNHUB_API_KEY = _resolve_api_key("FINNHUB_API_KEY")
TWELVEDATA_API_KEY = _resolve_api_key("TWELVEDATA_API_KEY")
MBOUM_BASE_URL = "https://api.mboum.com"
MASSIVE_BASE_URL = "https://api.massive.com/v2"
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Universe discovery still needs Massive. MBOUM is the primary OHLCV /
# fundamentals / options source when credits remain, but it is no longer
# required: an exhausted MBOUM plan falls through Massive -> TwelveData ->
# Finnhub -> Yahoo/yfinance without fabricating data.
REQUIRED_API_KEYS = ("MASSIVE_API_KEY",)
OPTIONAL_API_KEYS = (
    "MBOUM_API_KEY",
    "MBOUM_OPTIONS_KEY",
    "TWELVEDATA_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHAVANTAGE_API_KEY",
)

# Pipeline parameters
LOOKBACK_DAYS = 380  # calendar days to request (~252 trading days)
# Fallbacks request the same ~5y depth MBOUM Pro returns so SMA-200, RS and
# ML labels do not silently shrink when MBOUM is skipped.
HISTORY_CALENDAR_DAYS = 5 * 365 + 21
MIN_TRADING_DAYS = 252
MIN_UNIVERSE_SIZE = 500
BATCH_SIZE = 100  # yfinance download batch size
MAX_WORKERS = 8   # thread pool for fundamentals
OUTPUT_DIR = Path("scan_results")
TARGET_FINAL_CANDIDATES = 7
MIN_NEAR_MISS_RULES = 8
MAX_PANEL_CANDIDATES = 60
MAX_OPTIONS_EVAL_CANDIDATES = 20

# Wall-clock budget. GitHub Actions cancels the job at `timeout-minutes` and
# discards nothing but the log, so the pipeline enforces its own earlier
# deadline and always writes whatever it has before exiting.
_PIPELINE_BUDGET_RAW = os.environ.get("SCAN_BUDGET_MINUTES", "100")
try:
    PIPELINE_BUDGET_MINUTES = float(_PIPELINE_BUDGET_RAW)
except ValueError:
    PIPELINE_BUDGET_MINUTES = 100.0

# ── Strategy configuration ────────────────────────────────────────────────────
# Objective: maximum profit with accepted risk, short-to-mid-term holds,
# following insider buying and genuine hype/momentum, while refusing to chase
# names that have already gone parabolic and are statistically due to unwind.
HOLDING_HORIZON_DAYS = 20          # short-to-mid term; also the ML label horizon
ML_TARGET_RETURN = 0.05            # a "win" is +5% over the holding horizon

# ── Anti-chase limits ────────────────────────────────────────────────────────
# Calibrated against the actual objective: a 20-session hold for +5%. The test
# for every limit below is "if a name is already past this, is the next +5%
# still the most likely next move, or is the unwind?" The previous settings
# (35% / RSI 82 / +40% in 5d / +100% in 20d) answered that generously -- a name
# up 39% in a week and 34% over its 50DMA cleared every one of them, which is
# not an entry, it is the exit somebody else is taking.
MAX_EXTENSION_ABOVE_SMA50 = 0.25   # >25% above the 50DMA = climax-extended
MAX_EXHAUSTION_RSI = 78.0          # blow-off territory
MAX_SPIKE_RETURN_5D = 0.25         # +25% in a week is already violent
MAX_SPIKE_RETURN_20D = 0.60        # +60% in a month = late to the party
# Distance from the 20-day mean in ATRs. Volatility-aware, so it catches the
# climax on a quiet $200 name and does not punish an ordinary breakout on a
# high-beta one -- the blind spot a fixed percentage cannot cover. Ordinary
# breakouts run 1.5-3 ATR; the exhaustion score already reads 100 at 4.0.
MAX_STRETCH_ATR = 4.5

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scan_pipeline.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def now_et() -> str:
    """Current time in US/Eastern as formatted string."""
    return datetime.now(ET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")


def now_et_dt() -> datetime:
    """Current timezone-aware ET datetime."""
    return datetime.now(ET_TZ)


def today_et() -> date:
    """Current ET calendar date -- the reference for every DTE calculation.

    The runner's clock is UTC, so a naive local date rolls over to
    tomorrow at 20:00 ET and silently shifts every days-to-expiry by one.
    """
    return now_et_dt().date()


def is_market_open_now() -> bool:
    """Returns True if RTH market is currently open (Mon-Fri 09:30-16:00 ET).
    Holiday-aware-best-effort: skips obvious weekend; intra-day partial-bar
    detection still uses last-bar timestamp comparison upstream."""
    now = now_et_dt()
    if not is_trading_day(now.date()):
        return False
    open_t = dtime(9, 30)
    close_t = dtime(16, 0)
    return open_t <= now.time() <= close_t


def _observed_date(month: int, day: int, year: int) -> date:
    """Observed date for fixed NYSE holidays."""
    dt = datetime(year, month, day).date()
    if dt.weekday() == 5:
        return dt - timedelta(days=1)
    if dt.weekday() == 6:
        return dt + timedelta(days=1)
    return dt


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = datetime(year, month, 1).date()
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cur = datetime(year + 1, 1, 1).date() - timedelta(days=1)
    else:
        cur = datetime(year, month + 1, 1).date() - timedelta(days=1)
    while cur.weekday() != weekday:
        cur -= timedelta(days=1)
    return cur


def _easter_date(year: int) -> date:
    """Gregorian Easter date; Good Friday is an NYSE holiday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day).date()


def nyse_holidays(year: int) -> set:
    """Core NYSE full-day holidays used for freshness checks."""
    holidays = {
        _observed_date(1, 1, year),
        _nth_weekday(year, 1, 0, 3),   # MLK Day
        _nth_weekday(year, 2, 0, 3),   # Washington's Birthday
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),     # Memorial Day
        _observed_date(6, 19, year),   # Juneteenth
        _observed_date(7, 4, year),
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed_date(12, 25, year),
    }
    return holidays


def is_trading_day(day) -> bool:
    """Best-effort NYSE trading-day check without external dependencies."""
    day = pd.Timestamp(day).date()
    holidays = set(nyse_holidays(day.year))
    holidays.update(d for d in nyse_holidays(day.year + 1) if d.year == day.year)
    return day.weekday() < 5 and day not in holidays


def previous_trading_day(day) -> date:
    day = pd.Timestamp(day).date() - timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def expected_last_closed_trading_day(now: Optional[datetime] = None) -> date:
    """Latest daily bar the scanner should be willing to use.

    Before 18:00 ET on a trading day, vendors may not have finalized today's
    daily bar, so the expected closed session remains the previous session.
    """
    now = now or now_et_dt()
    today = now.date()
    if not is_trading_day(today):
        return previous_trading_day(today + timedelta(days=1))
    if now.time() >= dtime(18, 0):
        return today
    return previous_trading_day(today)


def trading_sessions_between(start_day, end_day) -> int:
    """Count trading sessions after start_day through end_day."""
    start = pd.Timestamp(start_day).date()
    end = pd.Timestamp(end_day).date()
    if start >= end:
        return 0
    count = 0
    cur = start + timedelta(days=1)
    while cur <= end:
        if is_trading_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count


def freshness_lag_sessions(last_day, now: Optional[datetime] = None) -> int:
    expected = expected_last_closed_trading_day(now)
    last = pd.Timestamp(last_day).date()
    return trading_sessions_between(last, expected)


def last_bar_is_partial(df: pd.DataFrame) -> bool:
    """Detect if the final OHLCV bar represents an in-progress session.
    During RTH, vendors expose today's intraday-aggregated bar whose volume is
    incomplete -- using it for volume-surge / volume-ratio checks is a
    well-known false-negative source. Returns True only if last bar's date
    matches today (ET) AND market is still open."""
    if df is None or df.empty:
        return False
    if not is_market_open_now():
        return False
    try:
        last_idx = df.index[-1]
        last_date = pd.Timestamp(last_idx).date()
        today = now_et_dt().date()
        return last_date == today
    except Exception:
        return False


def is_missing_value(val) -> bool:
    """Safe missing check that tolerates list-like API payloads."""
    if val is None:
        return True
    if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
        return len(val) == 0
    try:
        missing = pd.isna(val)
    except Exception:
        return False
    if isinstance(missing, (np.ndarray, pd.Series, list, tuple)):
        return bool(np.all(missing))
    return bool(missing)


def normalize_api_scalar(val):
    """Collapse API payload variants down to a scalar or NaN."""
    if isinstance(val, dict):
        for key in ("raw", "value", "fmt"):
            if key in val:
                return normalize_api_scalar(val.get(key))
        return np.nan

    if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
        for item in val:
            normalized = normalize_api_scalar(item)
            if not is_missing_value(normalized):
                return normalized
        return np.nan

    if isinstance(val, str):
        text = val.strip()
        if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "--"}:
            return np.nan
        try:
            return float(text.replace(",", ""))
        except ValueError:
            return text

    return val


def safe_div(a, b, default=np.nan):
    """Division with zero/nan protection."""
    if is_missing_value(b):
        return default
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def clamp(val, lo, hi):
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, val))


class PipelineError(Exception):
    """Fatal pipeline error -- stop immediately per SKILL policy."""
    pass


class PipelineBudgetExceeded(PipelineError):
    """Raised when the wall-clock budget is exhausted mid-pipeline."""
    pass


def _credential_status(name: str) -> str:
    """
    Where this run's key comes from: a live env/Actions secret ("env"), the
    committed fallback ("embedded"), or nowhere ("absent").

    Resolved from the environment and EMBEDDED_FALLBACK_KEYS on every call --
    never from the module constants above. Those freeze whatever environment
    the process started in, so on Actions (where every secret is exported for
    the job) they would report a live secret as "embedded" and claim a
    committed fallback exists for keys that have none, such as MBOUM.
    """
    if os.environ.get(name, "").strip():
        return "env"
    return "embedded" if EMBEDDED_FALLBACK_KEYS.get(name, "") else "absent"


def verify_api_credentials() -> Dict[str, str]:
    """
    Confirm every required API credential is available from the environment
    or from the committed fallback keys.
    """
    status = {name: _credential_status(name) for name in REQUIRED_API_KEYS}
    missing = [name for name, state in status.items() if state == "absent"]
    if missing:
        raise PipelineError(
            "Missing required API credentials: "
            f"{', '.join(missing)}. Set them as environment variables locally, "
            "or as repository secrets for the GitHub Actions workflow."
        )

    for name in OPTIONAL_API_KEYS:
        status[name] = _credential_status(name)

    if status.get("MBOUM_API_KEY") != "absent":
        log.info(
            "MBOUM Pro is the primary OHLCV/fundamentals source. "
            "If credits are exhausted mid-run, the engine falls back to "
            "Massive -> TwelveData -> Finnhub -> Yahoo/yfinance without "
            "fabricating bars."
        )
    else:
        log.warning(
            "MBOUM_API_KEY is not set -- OHLCV/fundamentals will use "
            "Massive -> TwelveData -> Finnhub -> Yahoo/yfinance. "
            "MBOUM remains the primary source on the next run if the key "
            "is restored with available credits."
        )
    if status.get("MBOUM_OPTIONS_KEY") == "absent":
        log.warning(
            "MBOUM_OPTIONS_KEY is not set -- options chains will fall back "
            "to Massive/Yahoo."
        )
    if status.get("TWELVEDATA_API_KEY") == "absent":
        log.info("TWELVEDATA_API_KEY is not set; TwelveData is skipped in the fallback chain.")
    elif status.get("TWELVEDATA_API_KEY") == "embedded":
        log.info(
            "TwelveData will use the committed fallback key "
            "(env/Actions secrets still win when set)."
        )
    if status.get("FINNHUB_API_KEY") == "absent":
        log.info("FINNHUB_API_KEY is not set; Finnhub is skipped in the fallback chain.")
    elif status.get("FINNHUB_API_KEY") == "embedded":
        log.info(
            "Finnhub will use the committed fallback key "
            "(env/Actions secrets still win when set)."
        )
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER CIRCUIT BREAKER -- keep MBOUM primary, fail over live
# ═══════════════════════════════════════════════════════════════════════════════

class ProviderExhausted(Exception):
    """Provider is out of credits, unauthorized, or unusable for the rest of this run."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"{provider}: {reason}")


_CREDIT_HINTS = (
    "out of credit", "out of api credits", "insufficient credit",
    "no credits", "credit limit", "not enough credit", "credits exhausted",
    "quota", "payment required", "upgrade your plan", "exceeded your",
    "monthly limit", "api call credits", "usage limit", "plan limit",
    "you have run out", "limit reached", "not entitled",
)
_MINUTE_HINTS = ("per minute", "current minute", "too many requests")


def classify_http_error(status_code: int, body: str = "") -> str:
    """Classify a provider error as credit, rate, auth, or other."""
    text = (body or "").lower()
    minute = any(hint in text for hint in _MINUTE_HINTS)
    monthly = any(hint in text for hint in ("month", "daily limit", "per day", "current day"))
    credit = any(hint in text for hint in _CREDIT_HINTS)
    if minute and not monthly:
        return "rate"
    if status_code == 402 or monthly or (credit and not minute):
        return "credit"
    if status_code in (401, 403):
        return "auth"
    if status_code == 429:
        return "rate"
    return "other"


class ProviderCircuit:
    """
    Process-wide circuit breaker.

    MBOUM is tried first on every run. Once credits/auth fail (or a short
    burst of empty responses proves the plan is dead), remaining calls skip
    MBOUM and use the next live provider. Circuits do not persist across
    runs, so restored MBOUM credits automatically become primary again.
    """

    _registry: Dict[str, "ProviderCircuit"] = {}
    _reg_lock = threading.Lock()

    def __init__(self, name: str, fail_limit: int = 6):
        self.name = name
        self.fail_limit = fail_limit
        self._lock = threading.Lock()
        self.disabled = False
        self.reason = ""
        self.consecutive_failures = 0
        self.successes = 0
        self._logged_trip = False

    @classmethod
    def get(cls, name: str, fail_limit: int = 6) -> "ProviderCircuit":
        with cls._reg_lock:
            inst = cls._registry.get(name)
            if inst is None:
                inst = cls(name, fail_limit)
                cls._registry[name] = inst
            return inst

    @classmethod
    def reset_all(cls) -> None:
        with cls._reg_lock:
            cls._registry.clear()

    def available(self) -> bool:
        return not self.disabled

    def trip(self, reason: str) -> None:
        with self._lock:
            self.disabled = True
            self.reason = reason or "unavailable"
            if not self._logged_trip:
                self._logged_trip = True
                if "MBOUM" in self.name:
                    log.warning(
                        f"{self.name} disabled for this run: {self.reason}. "
                        "Falling back to Massive -> TwelveData -> Finnhub -> "
                        "Yahoo/yfinance. MBOUM stays primary on the next run "
                        "if credits are restored."
                    )
                else:
                    log.warning(
                        f"{self.name} disabled for this run: {self.reason}. "
                        "Trying the next live provider."
                    )

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.successes += 1

    def record_failure(self, reason: str = "error", credit: bool = False) -> None:
        do_trip = False
        fail_reason = reason
        with self._lock:
            self.consecutive_failures += 1
            if credit or self.consecutive_failures >= self.fail_limit:
                if not self.disabled:
                    # Disable under the lock to avoid races with record_success().
                    self.disabled = True
                    self.reason = fail_reason or "unavailable"
                    do_trip = True
                    fail_reason = self.reason
        if do_trip:
            self.trip(fail_reason)


DATA_SOURCE_USAGE: Dict[str, Counter] = {
    "ohlcv": Counter(),
    "fundamentals": Counter(),
    "options": Counter(),
}


def reset_data_source_usage() -> None:
    for bucket in DATA_SOURCE_USAGE.values():
        bucket.clear()


def _normalize_ohlcv_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Force every provider onto the same OHLCV schema MBOUM returns."""
    if df is None or df.empty:
        return None
    required = ("Open", "High", "Low", "Close", "Volume")
    if any(col not in df.columns for col in required):
        return None
    out = df.loc[:, list(required)].copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(ET_TZ).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Volume"] > 0]
    return out if len(out) > 0 else None


def _history_window() -> Tuple[date, date]:
    end = today_et()
    start = end - timedelta(days=HISTORY_CALENDAR_DAYS)
    return start, end


class PipelineClock:
    """
    Wall-clock budget for the whole scan.

    GitHub Actions kills the job at `timeout-minutes` with no result artifact.
    The clock lets the pipeline notice it is running out of time at a stage
    boundary and unwind cleanly so the log, near-miss report and any partial
    results are still written and uploaded.
    """

    def __init__(self, budget_minutes: float = PIPELINE_BUDGET_MINUTES):
        self.start = time.time()
        self.budget_seconds = max(float(budget_minutes), 1.0) * 60.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def remaining(self) -> float:
        return self.budget_seconds - self.elapsed

    def check(self, stage: str) -> None:
        """Abort the run if the budget is exhausted at a stage boundary."""
        if self.remaining <= 0:
            raise PipelineBudgetExceeded(
                f"Wall-clock budget of {self.budget_seconds / 60:.0f} minutes "
                f"exhausted before {stage}. Writing partial output instead of "
                "letting the CI job be killed with no artifacts."
            )
        log.info(
            f"  [clock] {self.elapsed / 60:.1f} min elapsed, "
            f"{self.remaining / 60:.1f} min left -- entering {stage}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD CONTEXT -- live headlines and event risk for THIS run only
# ═══════════════════════════════════════════════════════════════════════════════

class WorldContext:
    """
    Current-world overlay. Headlines, market status and the near-term
    earnings calendar contextualize regime and annotate names that already
    survived the live scan. They never seed, replace, or shrink the equity
    universe -- no watchlist, no yesterday's CSV, no news-derived basket.
    """

    NEWS_LOOKBACK_HOURS = 36
    EARNINGS_WINDOW_DAYS = 5
    MAX_HEADLINES = 12
    # Classifiers for live headline text. These are event types, never
    # company names or a watchlist -- they only score today's tape.
    RISK_TERMS = (
        "federal reserve", "fomc", "rate hike", "rate cut", "interest rate",
        "cpi", "inflation", "pce ", "nonfarm", "payroll", "recession",
        "tariff", "sanction", "embargo", "blockade", "ceasefire",
        "invasion", "missile", "oil supply", "opec",
        "bank failure", "credit crunch", "sovereign default",
        "geopolit", "war ", "shutdown", "debt ceiling", "default risk",
        "nuclear", "hostage", "pandemic", "supply chain",
    )

    def __init__(self):
        self.headlines: List[Dict[str, Any]] = []
        self.event_risk: float = 50.0
        self.event_notes: List[str] = []
        self.earnings_soon: Dict[str, str] = {}
        self.market_status: Dict[str, Any] = {}
        self.source: str = ""

    def load(self, snapshot: Optional[Dict[str, Dict]] = None) -> None:
        session = get_market_router().session
        self._load_news(session)
        self._load_earnings(session)
        self._load_market_status(session)
        self._score_event_risk(snapshot)
        log.info(
            f"  World context: {len(self.headlines)} live headlines, "
            f"event-risk {self.event_risk:.0f}/100, "
            f"{len(self.earnings_soon)} names reporting within "
            f"{self.EARNINGS_WINDOW_DAYS}d"
        )
        for note in self.event_notes[:6]:
            log.info(f"    - {note}")

    def _load_news(self, session) -> None:
        if not FINNHUB_API_KEY:
            return
        try:
            resp = session.get(
                f"{FINNHUB_BASE_URL}/news",
                params={"category": "general", "token": FINNHUB_API_KEY},
                timeout=15,
            )
            kind = classify_http_error(resp.status_code, resp.text or "")
            if kind in ("credit", "auth"):
                log.warning(f"  World news unavailable (HTTP {resp.status_code})")
                return
            if resp.status_code != 200:
                return
            payload = resp.json()
            if not isinstance(payload, list):
                return
            cutoff = now_et_dt() - timedelta(hours=self.NEWS_LOOKBACK_HOURS)
            rows = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                headline = str(item.get("headline") or "").strip()
                if not headline:
                    continue
                ts = item.get("datetime")
                try:
                    when = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(ET_TZ)
                except Exception:
                    when = None
                if when is not None and when < cutoff:
                    continue
                rows.append({
                    "headline": headline[:240],
                    "source": str(item.get("source") or ""),
                    "datetime": when.isoformat() if when is not None else None,
                    "url": str(item.get("url") or ""),
                })
                if len(rows) >= 80:
                    break
            rows.sort(key=lambda r: r.get("datetime") or "", reverse=True)
            self.headlines = rows[:self.MAX_HEADLINES]
            if self.headlines:
                self.source = "Finnhub"
        except Exception as exc:
            log.debug(f"  World news fetch failed: {exc}")

    def _load_earnings(self, session) -> None:
        if not FINNHUB_API_KEY:
            return
        start = today_et()
        end = start + timedelta(days=self.EARNINGS_WINDOW_DAYS)
        try:
            resp = session.get(
                f"{FINNHUB_BASE_URL}/calendar/earnings",
                params={
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                    "token": FINNHUB_API_KEY,
                },
                timeout=20,
            )
            if resp.status_code != 200:
                return
            payload = resp.json() or {}
            rows = payload.get("earningsCalendar") or []
            if not isinstance(rows, list):
                return
            soon = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol") or "").upper()
                day = str(row.get("date") or "")
                if not sym or not day or not sym.isalpha() or len(sym) > 5:
                    continue
                soon.setdefault(sym, day)
            self.earnings_soon = soon
        except Exception as exc:
            log.debug(f"  Earnings calendar fetch failed: {exc}")

    def _load_market_status(self, session) -> None:
        if not FINNHUB_API_KEY:
            return
        try:
            resp = session.get(
                f"{FINNHUB_BASE_URL}/stock/market-status",
                params={"exchange": "US", "token": FINNHUB_API_KEY},
                timeout=10,
            )
            if resp.status_code != 200:
                return
            payload = resp.json()
            if isinstance(payload, dict):
                self.market_status = {
                    "exchange": payload.get("exchange"),
                    "is_open": payload.get("isOpen"),
                    "holiday": payload.get("holiday"),
                    "timezone": payload.get("timezone"),
                }
        except Exception as exc:
            log.debug(f"  Market status fetch failed: {exc}")

    def _score_event_risk(self, snapshot: Optional[Dict[str, Dict]] = None) -> None:
        score = 40.0
        notes: List[str] = []
        blob = " ".join(
            str(h.get("headline") or "").lower() for h in self.headlines
        )
        hits = [term for term in self.RISK_TERMS if term in blob]
        if hits:
            score += min(40.0, 8.0 * len(set(hits)))
            notes.append(
                "Live headlines mention: " + ", ".join(sorted(set(hits))[:8])
            )
        if self.headlines:
            notes.append(f"Lead headline: {self.headlines[0]['headline'][:160]}")
        holiday = self.market_status.get("holiday")
        if holiday:
            score += 5
            notes.append(f"US session holiday flag: {holiday}")
        is_open = self.market_status.get("is_open")
        if is_open is False:
            notes.append("US cash session closed at scan time (live market-status)")
        # Live gauges from this run's macro snapshot -- indices/commodities,
        # never a pre-selected equity.
        snap = snapshot or {}
        vix_last = (snap.get("vix") or {}).get("last")
        if vix_last is not None:
            if vix_last >= 35:
                score += 18
                notes.append(f"Live VIX {vix_last:.1f} -- panic tape")
            elif vix_last >= 25:
                score += 10
                notes.append(f"Live VIX {vix_last:.1f} -- stressed tape")
            elif vix_last >= 20:
                score += 4
                notes.append(f"Live VIX {vix_last:.1f} -- elevated tape")
        gold_ret = (snap.get("gold") or {}).get("ret_20d")
        if gold_ret is not None and gold_ret > 0.06 and (
            vix_last is None or vix_last > 20
        ):
            score += 4
            notes.append(f"Live gold +{gold_ret:.1%} 20d -- safe-haven bid")
        wti_ret = (snap.get("wti") or {}).get("ret_20d")
        if wti_ret is not None and abs(wti_ret) > 0.12:
            score += 4
            notes.append(f"Live WTI {wti_ret:+.1%} 20d -- energy shock tape")
        if self.earnings_soon:
            notes.append(
                f"Live earnings calendar: {len(self.earnings_soon)} names "
                f"print within {self.EARNINGS_WINDOW_DAYS} sessions "
                "(used only to annotate survivors, not to pick names)"
            )
        if not self.headlines and not self.earnings_soon:
            notes.append(
                "World-event feed empty this run; regime uses live prices only."
            )
        self.event_risk = float(clamp(score, 0.0, 100.0))
        self.event_notes = notes

    def annotate(self, candidates: List[Dict], session=None) -> List[Dict]:
        """Attach live news/earnings to already-selected names. Never adds tickers.

        Incoming candidate order and membership are preserved. Headlines and
        the earnings calendar cannot insert, drop, or reorder names.
        """
        incoming_tickers = [str(c.get("ticker") or "") for c in candidates]
        if not candidates:
            return candidates
        session = session or get_market_router().session
        start = (today_et() - timedelta(days=5)).isoformat()
        end = today_et().isoformat()
        for candidate in candidates:
            ticker = str(candidate.get("ticker") or "").upper()
            flags = candidate.get("flags")
            if not isinstance(flags, list):
                flags = []
                candidate["flags"] = flags
            earn = self.earnings_soon.get(ticker)
            if earn:
                candidate["live_earnings_date"] = earn
                if "LIVE_EARNINGS_WINDOW" not in flags:
                    flags.append("LIVE_EARNINGS_WINDOW")
            headlines = self._company_headlines(session, ticker, start, end)
            if headlines:
                candidate["live_headlines"] = headlines
                if "LIVE_NEWS" not in flags:
                    flags.append("LIVE_NEWS")
        outgoing = [str(c.get("ticker") or "") for c in candidates]
        if outgoing != incoming_tickers:
            raise PipelineError(
                "World context mutated the candidate set. "
                "News/earnings must never pick names."
            )
        return candidates

    def _company_headlines(self, session, ticker: str, start: str, end: str) -> List[str]:
        if not FINNHUB_API_KEY or not ticker:
            return []
        try:
            resp = session.get(
                f"{FINNHUB_BASE_URL}/company-news",
                params={
                    "symbol": ticker,
                    "from": start,
                    "to": end,
                    "token": FINNHUB_API_KEY,
                },
                timeout=12,
            )
            if resp.status_code != 200:
                return []
            payload = resp.json()
            if not isinstance(payload, list):
                return []
            out = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                headline = str(item.get("headline") or "").strip()
                if headline:
                    out.append(headline[:200])
                if len(out) >= 3:
                    break
            return out
        except Exception:
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source or None,
            "event_risk": round(self.event_risk, 1),
            "notes": list(self.event_notes),
            "market_status": dict(self.market_status),
            "headlines": list(self.headlines),
            "earnings_window_days": self.EARNINGS_WINDOW_DAYS,
            "earnings_names_in_window": len(self.earnings_soon),
            "seeds_universe": False,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO REGIME -- live geopolitical / market-context overlay
# ═══════════════════════════════════════════════════════════════════════════════

class MacroRegime:
    """
    Real-time macro context. Pulled from public Yahoo v8 chart endpoints
    (no key required). Drives the panel "M" (market direction) score and
    final position-sizing scaler.

    Tracked symbols and rationale (no hardcoded equity tickers -- only
    indices/factors used universally as macro context):
      - ^GSPC  (S&P 500)              -- broad equities trend
      - ^NDX   (Nasdaq 100)           -- risk / growth proxy
      - ^RUT   (Russell 2000)         -- small-cap risk-on gauge
      - ^VIX   (CBOE Volatility)      -- fear gauge / IV regime
      - ^TNX   (10Y UST yield)        -- duration / discount-rate
      - ^IRX   (3M UST yield)         -- short rate
      - DX-Y.NYB (DXY)                -- USD strength (geopolitical risk-off)
      - GC=F   (Gold futures)         -- safe haven / inflation hedge
      - CL=F   (WTI crude)            -- supply-shock / geopolitics
      - HG=F   (Copper)               -- global growth proxy
      - ^MOVE  (bond vol, optional)   -- credit/rate stress
    """

    SYMBOLS = {
        "spx": "^GSPC",
        "ndx": "^NDX",
        "rut": "^RUT",
        "vix": "^VIX",
        "tnx": "^TNX",
        "irx": "^IRX",
        "dxy": "DX-Y.NYB",
        "gold": "GC=F",
        "wti": "CL=F",
        "copper": "HG=F",
    }
    REQUIRED_SYMBOLS = {"spx", "vix", "tnx", "irx", "dxy", "wti"}
    MIN_SNAPSHOT_COUNT = 7

    def __init__(self):
        self.yahoo = YahooDirectAPI()
        self.snapshot: Dict[str, Dict] = {}
        self.regime_score: Optional[float] = None
        self.regime_label: str = "UNAVAILABLE"
        self.notes: List[str] = []
        self.world = WorldContext()
        # Which macro series failed to load this run. Recorded rather than
        # fatal: the regime is scored from whatever did load, and the report
        # says which inputs were missing so a thin read is never mistaken for
        # a confident one.
        self.missing_symbols: List[str] = []
        self.missing_required: List[str] = []
        self.degraded: bool = False

    def _series(self, symbol: str, range_: str = "2y") -> Optional[pd.DataFrame]:
        """Pull recent OHLCV via Yahoo v8 chart -- no key needed."""
        url = f"{self.yahoo.BASE_URL}/{symbol}"
        params = {"range": range_, "interval": "1d", "includePrePost": "false"}
        try:
            r = self.yahoo.session.get(url, params=params, timeout=12)
            if r.status_code != 200:
                return None
            data = r.json().get("chart", {}).get("result")
            if not data:
                return None
            chart = data[0]
            ts = chart.get("timestamp", []) or []
            quote = chart.get("indicators", {}).get("quote", [{}])[0]
            closes = quote.get("close", []) or []
            if len(ts) < 30 or len(closes) < 30:
                return None
            idx = pd.to_datetime(ts, unit="s").tz_localize("UTC").tz_convert(ET_TZ)
            df = pd.DataFrame({"Close": closes}, index=idx).dropna()
            return df if len(df) >= 30 else None
        except Exception:
            return None

    def load(self) -> None:
        """Fetch all macro symbols in parallel and compute regime score."""
        log.info("Loading live macro regime context (VIX, yields, DXY, gold, oil, indices)...")
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(self._series, sym): name
                       for name, sym in self.SYMBOLS.items()}
            for f in as_completed(futures):
                name = futures[f]
                try:
                    df = f.result()
                    if df is not None:
                        last_date = pd.Timestamp(df.index[-1]).date()
                        lag = freshness_lag_sessions(last_date)
                        if lag > 1:
                            log.warning(
                                f"  Macro {name} stale by {lag} trading sessions; excluding."
                            )
                            continue
                        self.snapshot[name] = {
                            "df": df,
                            "last": float(df["Close"].iloc[-1]),
                            "last_date": last_date.isoformat(),
                            "ret_1d": float(df["Close"].pct_change().iloc[-1]) if len(df) >= 2 else 0.0,
                            "ret_5d": float(df["Close"].pct_change(5).iloc[-1]) if len(df) >= 6 else 0.0,
                            "ret_20d": float(df["Close"].pct_change(20).iloc[-1]) if len(df) >= 21 else 0.0,
                            "ret_63d": float(df["Close"].pct_change(63).iloc[-1]) if len(df) >= 64 else 0.0,
                        }
                except Exception:
                    pass

        # A macro series that fails to load degrades the regime read; it does
        # not end the run. Macro is *context* -- it tunes position sizing and
        # the panel's market-direction score, it does not pick or price a
        # single candidate. Every block in _score_regime() is already guarded
        # on presence, so an absent series contributes nothing rather than a
        # guessed value: the neutral fallback is the honest answer, and it is
        # not worth discarding a full universe scan over one dead endpoint.
        self.missing_symbols = sorted(set(self.SYMBOLS) - set(self.snapshot))
        self.missing_required = sorted(self.REQUIRED_SYMBOLS - set(self.snapshot))
        self.degraded = bool(self.missing_symbols)
        if self.degraded:
            log.warning(
                f"  Macro context incomplete: {len(self.snapshot)}/"
                f"{len(self.SYMBOLS)} series loaded. Unavailable: "
                f"{', '.join(self.missing_symbols)}"
                + (
                    f" (required: {', '.join(self.missing_required)})"
                    if self.missing_required else ""
                )
                + ". Scoring the regime from what did load."
            )

        self._score_regime()
        self._apply_world_context()

    def _score_regime(self) -> None:
        """Composite 0-100 macro regime score. Higher = more risk-on."""
        score = 50.0
        notes = []

        spx = self.snapshot.get("spx")
        if spx is not None and len(spx["df"]) >= 200:
            close = spx["df"]["Close"]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]
            last = close.iloc[-1]
            if last > sma50 > sma200:
                score += 18
                notes.append("SPX in uptrend (above 50/200d SMA, golden alignment)")
            elif last > sma200:
                score += 9
                notes.append("SPX above 200d SMA")
            elif last > sma50:
                score += 4
                notes.append("SPX above 50d SMA (mixed)")
            else:
                score -= 12
                notes.append("SPX below key MAs -- defensive regime")

        # VIX regime
        vix = self.snapshot.get("vix")
        if vix is not None:
            vlast = vix["last"]
            if vlast < 14:
                score += 8
                notes.append(f"VIX {vlast:.1f} -- complacent / risk-on")
            elif vlast < 18:
                score += 4
                notes.append(f"VIX {vlast:.1f} -- benign")
            elif vlast < 25:
                score -= 2
                notes.append(f"VIX {vlast:.1f} -- elevated")
            elif vlast < 35:
                score -= 12
                notes.append(f"VIX {vlast:.1f} -- stressed")
            else:
                score -= 22
                notes.append(f"VIX {vlast:.1f} -- panic regime")

        # Yield curve (10Y - 3M proxy via ^TNX - ^IRX, in percentage points)
        # Yahoo `^TNX` is typically quoted as yield * 10 (e.g. 43.0 => 4.30%),
        # while `^IRX` is already in percent. Normalize both to percent first.
        tnx = self.snapshot.get("tnx")
        irx = self.snapshot.get("irx")
        if tnx is not None and irx is not None:
            tnx_pct = tnx["last"] / 10.0
            irx_pct = irx["last"]
            spread = tnx_pct - irx_pct
            if spread > 1.5:
                score += 6
                notes.append(f"Yield curve steep (+{spread:.2f}pp) -- pro-growth")
            elif spread > 0.25:
                score += 2
                notes.append(f"Yield curve positive (+{spread:.2f}pp)")
            elif spread > -0.25:
                score -= 2
                notes.append(f"Yield curve flat ({spread:+.2f}pp)")
            else:
                score -= 8
                notes.append(f"Yield curve INVERTED ({spread:+.2f}pp) -- recession signal")

        # DXY -- strong USD pressures multinationals/EM, mixed for domestics
        dxy = self.snapshot.get("dxy")
        if dxy is not None:
            r20 = dxy.get("ret_20d", 0.0)
            if r20 > 0.04:
                score -= 4
                notes.append(f"DXY +{r20:.1%} 20d -- USD strength = EPS headwind")
            elif r20 < -0.03:
                score += 3
                notes.append(f"DXY {r20:.1%} 20d -- USD weakness = EPS tailwind")

        # Crude (geopolitics / supply shock)
        wti = self.snapshot.get("wti")
        if wti is not None:
            r20 = wti.get("ret_20d", 0.0)
            if r20 > 0.15:
                score -= 4
                notes.append(f"WTI +{r20:.1%} 20d -- supply shock / inflation risk")
            elif r20 < -0.15:
                score += 1
                notes.append(f"WTI {r20:.1%} 20d -- demand softness or supply easing")

        # Gold (safe-haven flow, geopolitical tension)
        gold = self.snapshot.get("gold")
        if gold is not None and vix is not None:
            r20 = gold.get("ret_20d", 0.0)
            if r20 > 0.06 and vix["last"] > 20:
                score -= 3
                notes.append(f"Gold +{r20:.1%} with elevated VIX -- safe-haven bid")

        # Copper (global growth proxy, "Dr. Copper")
        copper = self.snapshot.get("copper")
        if copper is not None:
            r20 = copper.get("ret_20d", 0.0)
            if r20 > 0.05:
                score += 3
                notes.append(f"Copper +{r20:.1%} 20d -- pro-cyclical signal")
            elif r20 < -0.05:
                score -= 3
                notes.append(f"Copper {r20:.1%} 20d -- growth concern")

        # Russell vs SPX (small-cap leadership = risk-on)
        rut = self.snapshot.get("rut")
        if rut is not None and spx is not None:
            excess = rut.get("ret_20d", 0.0) - spx.get("ret_20d", 0.0)
            if excess > 0.02:
                score += 3
                notes.append(f"RUT outperforming SPX by {excess:.1%} -- breadth healthy")
            elif excess < -0.04:
                score -= 4
                notes.append(f"RUT lagging SPX by {excess:.1%} -- narrow leadership")

        score = clamp(score, 0.0, 100.0)
        if score >= 75:
            label = "RISK_ON"
        elif score >= 60:
            label = "CONSTRUCTIVE"
        elif score >= 45:
            label = "NEUTRAL"
        elif score >= 30:
            label = "DEFENSIVE"
        else:
            label = "RISK_OFF"

        self.regime_score = float(score)
        self.regime_label = label
        self.notes = notes

        log.info(f"  Macro regime: {label} (score {score:.0f}/100)")
        for n in notes:
            log.info(f"    - {n}")

    def _apply_world_context(self) -> None:
        """Fold live headlines into the already-computed price regime."""
        try:
            self.world.load(snapshot=self.snapshot)
        except Exception as exc:
            log.warning(f"  World context unavailable this run: {exc}")
            return
        if self.regime_score is None:
            return
        if self.world.event_risk >= 70:
            self.regime_score = float(clamp(self.regime_score - 8.0, 0.0, 100.0))
            self.notes.append(
                f"Live event risk {self.world.event_risk:.0f}/100 -- "
                "tightening regime until headlines cool"
            )
        elif self.world.event_risk >= 55:
            self.regime_score = float(clamp(self.regime_score - 3.0, 0.0, 100.0))
            self.notes.append(
                f"Live event risk {self.world.event_risk:.0f}/100 -- modest caution"
            )
        self.notes.extend(self.world.event_notes[:4])
        if self.regime_score >= 75:
            self.regime_label = "RISK_ON"
        elif self.regime_score >= 60:
            self.regime_label = "CONSTRUCTIVE"
        elif self.regime_score >= 45:
            self.regime_label = "NEUTRAL"
        elif self.regime_score >= 30:
            self.regime_label = "DEFENSIVE"
        else:
            self.regime_label = "RISK_OFF"
        log.info(
            f"  Macro regime after world context: {self.regime_label} "
            f"(score {self.regime_score:.0f}/100)"
        )

    def position_sizing_scalar(self) -> float:
        """Multiplier in [0.4, 1.2] for downstream position sizing.
        Used to scale the recommended dollar exposure based on regime."""
        if self.regime_score is None:
            # load() always scores the regime now, so this only fires if
            # sizing is asked for before the macro layer ran. Size neutrally
            # and say so rather than ending the run at the last step.
            log.warning(
                "  Macro regime score unavailable for position sizing -- "
                "using neutral 0.80."
            )
            return 0.80
        s = self.regime_score
        if s >= 75:
            scalar = 1.20
        elif s >= 60:
            scalar = 1.00
        elif s >= 45:
            scalar = 0.80
        elif s >= 30:
            scalar = 0.55
        else:
            scalar = 0.40
        if self.world.event_risk >= 70:
            scalar *= 0.80
        elif self.world.event_risk >= 55:
            scalar *= 0.90
        if self.missing_required:
            # Read the regime from what loaded, but do not lever up on it.
            # Sizing above 1.0 is a statement that conditions are confirmed
            # risk-on, and a partial macro picture cannot confirm that.
            scalar = min(scalar, 1.00)
        return float(clamp(scalar, 0.40, 1.20))

    def panel_m_score(self) -> float:
        """Use the regime score directly as O'Neil's M (Market Direction)
        score, replacing the SPX-only proxy with a real macro composite."""
        if self.regime_score is None:
            raise PipelineError("Macro regime score unavailable for panel scoring.")
        return float(self.regime_score)

    def to_dict(self) -> Dict:
        out = {"regime_score": round(self.regime_score, 1) if self.regime_score is not None else None,
               "regime_label": self.regime_label,
               "notes": list(self.notes),
               "inputs_loaded": f"{len(self.snapshot)}/{len(self.SYMBOLS)}",
               "inputs_missing": list(self.missing_symbols),
               "inputs_missing_required": list(self.missing_required),
               "macro_degraded": bool(self.degraded),
               "world_context": self.world.to_dict(),
               "snapshots": {}}
        for name, snap in self.snapshot.items():
            out["snapshots"][name] = {
                "last": round(snap["last"], 4),
                "last_date": snap.get("last_date"),
                "ret_1d": round(snap.get("ret_1d", 0.0), 4),
                "ret_20d": round(snap.get("ret_20d", 0.0), 4),
                "ret_63d": round(snap.get("ret_63d", 0.0), 4),
            }
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 -- UNIVERSE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

class UniverseDiscovery:
    """Dynamically discovers the live U.S. equity universe from MASSIVE."""

    MASSIVE_BASE = "https://api.massive.com/v3"

    # Connect/read timeout for a single page. A 10-minute read timeout let a
    # slow-drip endpoint stall each page and burn the whole CI budget before
    # any equity work started.
    REQUEST_TIMEOUT = (10, 30)
    # Hard ceiling on pagination so a malformed `next_url` cannot loop forever.
    MAX_PAGES = 100

    # Common stock types to include
    VALID_TYPES = {"Common Stock", "EQS", ""}

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=30, pool_maxsize=30)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"X-Massive-Token": self.api_key})

    def discover(self) -> List[str]:
        """
        Fetch all active U.S. equity symbols from Masssive.
        Returns list of ticker strings.
        Raises PipelineError on failure.
        """
        log.info("STAGE 1: Universe Discovery -- querying Masssive /stock/symbol")

        data = []
        url = f"{self.MASSIVE_BASE}/reference/tickers"
        params = {
            "market": "stocks",
            "locale": "us",
            "active": "true",
            "limit": 1000,
            "sort": "ticker",
            "order": "asc",
            "apiKey": self.api_key,
        }

        pages = 0
        seen_urls = set()
        while url:
            pages += 1
            if pages > self.MAX_PAGES:
                log.warning(
                    f"Universe discovery stopped at the {self.MAX_PAGES}-page ceiling."
                )
                break
            if url in seen_urls:
                log.warning("Universe discovery pagination looped -- stopping.")
                break
            seen_urls.add(url)

            retries = 0
            backoff = 1
            while retries < 3:
                try:
                    resp = self.session.get(
                        url, params=params, timeout=self.REQUEST_TIMEOUT
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception as e:
                    retries += 1
                    if retries >= 3:
                        log.warning(
                            f"Massive live universe failed ({e}); "
                            "trying Finnhub US symbol list. Still no preset basket."
                        )
                        return self._discover_finnhub()
                    log.warning(f"Massive retry {retries}/3 after {backoff}s -- {e}")
                    time.sleep(backoff)
                    backoff *= 2

            results = payload.get("results", [])
            if not isinstance(results, list):
                raise PipelineError(
                    "Massive returned an invalid ticker payload. Pipeline STOPPED."
                )

            data.extend(results)
            url = payload.get("next_url")
            params = {"apiKey": self.api_key} if url else None

        if not isinstance(data, list) or len(data) == 0:
            log.warning(
                "Massive returned an empty symbol list; trying Finnhub US symbol list."
            )
            return self._discover_finnhub()

        tickers = self._clean_listed_equities(data)
        log.info(
            f"STAGE 1 COMPLETE: {len(data)} raw symbols -> {len(tickers)} "
            f"cleaned tickers (common stocks + ETFs)"
        )

        if len(tickers) < MIN_UNIVERSE_SIZE:
            log.warning(
                f"Massive universe too small ({len(tickers)}); trying Finnhub live list."
            )
            alt = self._discover_finnhub()
            if len(alt) >= MIN_UNIVERSE_SIZE:
                return alt
            raise PipelineError(
                f"Universe too small ({len(tickers)} < {MIN_UNIVERSE_SIZE}). "
                f"Pipeline STOPPED. Do NOT substitute a preset basket."
            )

        return tickers

    def _discover_finnhub(self) -> List[str]:
        """Live US listing fallback. Still a full exchange dump, never a watchlist."""
        if not FINNHUB_API_KEY:
            raise PipelineError(
                "Live universe discovery FAILED on Massive and Finnhub is not "
                "configured. Do NOT substitute a preset basket."
            )
        log.info("STAGE 1: Universe Discovery -- querying Finnhub /stock/symbol")
        try:
            resp = self.session.get(
                f"{FINNHUB_BASE_URL}/stock/symbol",
                params={"exchange": "US", "token": FINNHUB_API_KEY},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise PipelineError(
                f"Live universe discovery FAILED. Sources: Massive then Finnhub. "
                f"Error: {exc}. Do NOT substitute a preset basket."
            )
        if not isinstance(payload, list) or not payload:
            raise PipelineError(
                "Finnhub returned an empty symbol list. Do NOT substitute a preset basket."
            )
        tickers = self._clean_listed_equities(payload)
        log.info(
            f"STAGE 1 COMPLETE via Finnhub: {len(payload)} raw symbols -> "
            f"{len(tickers)} cleaned tickers"
        )
        if len(tickers) < MIN_UNIVERSE_SIZE:
            raise PipelineError(
                f"Universe too small ({len(tickers)} < {MIN_UNIVERSE_SIZE}). "
                f"Pipeline STOPPED. Do NOT substitute a preset basket."
            )
        return tickers

    @staticmethod
    def _clean_listed_equities(data: List[Dict]) -> List[str]:
        VALID_MIC = {"XNYS", "XNAS", "XASE", "ARCX", "BATS"}
        VALID_TYPES = {
            "CS", "Common Stock", "EQS",
            "ETF", "ETP", "REIT", "MLP", "Closed-End Fund",
        }
        tickers = []
        for item in data:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("ticker") or item.get("symbol") or item.get("displaySymbol") or "")
            mic = str(item.get("primary_exchange") or item.get("mic") or "")
            sec_type = str(item.get("type") or "")
            if mic and mic not in VALID_MIC:
                continue
            if sec_type and sec_type not in VALID_TYPES:
                continue
            if any(c in sym for c in [".", "-", "/", "+"]):
                continue
            if len(sym) > 5 or len(sym) == 0:
                continue
            if not sym.isalpha():
                continue
            tickers.append(sym.upper())
        return sorted(set(tickers))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING -- MBOUM Pro (primary) + live provider cascade
# ═══════════════════════════════════════════════════════════════════════════════

class MboumAPI:
    """
    MBOUM Pro API client -- primary data source when credits remain.
    Provides: OHLCV history (5yr), fundamentals, options.
    Credit/quota failures trip a process-wide circuit so the rest of the
    scan falls through to Massive / TwelveData / Finnhub / Yahoo.
    """

    def __init__(self, api_key: str = MBOUM_API_KEY):
        self.api_key = api_key
        self.base = MBOUM_BASE_URL
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _decode_payload(self, resp, circuit: ProviderCircuit, what: str) -> Optional[Dict]:
        """Parse an MBOUM response, tripping the circuit on credit/auth failure."""
        body_text = resp.text or ""
        kind = classify_http_error(resp.status_code, body_text)
        if kind == "rate":
            return None
        if kind in ("credit", "auth"):
            snippet = body_text.replace("\n", " ")[:180]
            circuit.trip(f"{what} HTTP {resp.status_code}: {snippet}")
            raise ProviderExhausted("MBOUM", f"{what} {kind}: HTTP {resp.status_code}")
        if resp.status_code != 200:
            circuit.record_failure(f"{what} HTTP {resp.status_code}")
            return None
        try:
            data = resp.json()
        except Exception:
            circuit.record_failure(f"{what} invalid JSON")
            return None
        if not isinstance(data, dict):
            circuit.record_failure(f"{what} unexpected payload")
            return None

        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        body = data.get("body")
        meta_msg = str(meta.get("message") or meta.get("error") or "")
        meta_status = meta.get("status")
        inspect_text = " ".join(
            part for part in (meta_msg, body if isinstance(body, str) else "") if part
        )
        nested_status = meta_status if isinstance(meta_status, int) else resp.status_code
        nested_kind = classify_http_error(nested_status, inspect_text)
        if nested_kind in ("credit", "auth") or (
            isinstance(body, str) and nested_kind == "credit"
        ):
            circuit.trip(f"{what}: {inspect_text[:180] or nested_kind}")
            raise ProviderExhausted("MBOUM", f"{what} {nested_kind}")
        if isinstance(body, str):
            circuit.record_failure(f"{what}: {body[:160]}")
            return None
        if isinstance(meta_status, int) and meta_status >= 400:
            circuit.record_failure(f"{what} meta.status={meta_status}")
            return None
        return data

    def get_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch full OHLCV history for a ticker.
        Returns up to ~1257 daily bars (5 years) with adjusted close.
        """
        circuit = ProviderCircuit.get("MBOUM-OHLCV", fail_limit=4)
        if not self.api_key or not circuit.available():
            return None

        url = f"{self.base}/v1/markets/stock/history"
        params = {"symbol": symbol, "interval": "1d", "diffandsplits": "true"}

        try:
            resp = self.session.get(url, params=params, timeout=20)
            if classify_http_error(resp.status_code, resp.text or "") == "rate":
                time.sleep(3)
                resp = self.session.get(url, params=params, timeout=20)
            data = self._decode_payload(resp, circuit, f"history {symbol}")
            if not data:
                return None

            body = data.get("body", {})
            if not isinstance(body, dict) or len(body) < 50:
                circuit.record_failure(f"history {symbol}: empty/short body")
                return None

            rows = []
            for key, bar in body.items():
                if key == "events" or not isinstance(bar, dict):
                    continue
                if "close" not in bar:
                    continue
                close = normalize_api_scalar(bar.get("close"))
                adj_close = normalize_api_scalar(bar.get("adjclose", close))
                factor = safe_div(adj_close, close, default=1.0)
                if is_missing_value(factor) or factor <= 0:
                    factor = 1.0
                rows.append({
                    "Date": bar.get("date"),
                    "Open": normalize_api_scalar(bar.get("open")) * factor,
                    "High": normalize_api_scalar(bar.get("high")) * factor,
                    "Low": normalize_api_scalar(bar.get("low")) * factor,
                    "Close": adj_close,
                    "Volume": bar.get("volume"),
                })

            if not rows:
                circuit.record_failure(f"history {symbol}: no bars")
                return None

            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df = _normalize_ohlcv_frame(df)
            if df is None:
                circuit.record_failure(f"history {symbol}: unusable bars")
                return None
            circuit.record_success()
            return df

        except ProviderExhausted:
            raise
        except Exception as e:
            circuit.record_failure(str(e))
            return None

    def get_modules(self, symbol: str, modules: List[str]) -> Dict:
        """Fetch fundamental modules for a ticker."""
        circuit = ProviderCircuit.get("MBOUM-fundamentals", fail_limit=3)
        if not self.api_key or not circuit.available():
            return {}
        result = {}
        for mod in modules:
            try:
                url = f"{self.base}/v1/markets/stock/modules"
                params = {"symbol": symbol, "module": mod}
                resp = self.session.get(url, params=params, timeout=15)
                if classify_http_error(resp.status_code, resp.text or "") == "rate":
                    time.sleep(2)
                    resp = self.session.get(url, params=params, timeout=15)
                data = self._decode_payload(resp, circuit, f"module {mod} {symbol}")
                if not data:
                    continue
                body = data.get("body", {})
                if isinstance(body, dict) and body:
                    result[mod] = body
            except ProviderExhausted:
                raise
            except Exception:
                pass
        if result:
            circuit.record_success()
        else:
            circuit.record_failure(f"modules {symbol}: empty")
        return result

    def get_options_meta(self, symbol: str) -> Optional[Dict]:
        """Fetch the options meta envelope for a symbol.
        Returns the first body element which contains:
          - expirationDates: list of unix-second epochs
          - strikes: list of all strike prices on the chain
          - quote: full live quote (regularMarketPrice, bid, ask, IV envelope)
          - options: list with the FIRST expiration's chain (calls+puts)
        Use the dedicated options-tier key.
        """
        circuit = ProviderCircuit.get("MBOUM-options", fail_limit=3)
        if not MBOUM_OPTIONS_KEY or not circuit.available():
            return None
        url = f"{self.base}/v1/markets/options"
        params = {"symbol": symbol}
        headers = {"Authorization": f"Bearer {MBOUM_OPTIONS_KEY}"}
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=15)
            if classify_http_error(resp.status_code, resp.text or "") == "rate":
                time.sleep(2)
                resp = self.session.get(url, params=params, headers=headers, timeout=15)
            data = self._decode_payload(resp, circuit, f"options meta {symbol}")
            if not data:
                return None
            body = data.get("body")
            if isinstance(body, list) and body:
                circuit.record_success()
                return body[0]
            if isinstance(body, dict) and body:
                circuit.record_success()
                return body
            circuit.record_failure(f"options meta {symbol}: empty")
        except ProviderExhausted:
            raise
        except Exception:
            circuit.record_failure(f"options meta {symbol}: error")
            return None
        return None

    def get_options_for_expiration(self, symbol: str, expiration_epoch: int) -> Optional[Dict]:
        """Fetch the calls+puts chain for ONE expiration date.
        Returns the {'expirationDate', 'calls': [...], 'puts': [...]} entry."""
        circuit = ProviderCircuit.get("MBOUM-options", fail_limit=3)
        if not MBOUM_OPTIONS_KEY or not circuit.available():
            return None
        url = f"{self.base}/v1/markets/options"
        params = {"symbol": symbol, "expiration": int(expiration_epoch)}
        headers = {"Authorization": f"Bearer {MBOUM_OPTIONS_KEY}"}
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=15)
            if classify_http_error(resp.status_code, resp.text or "") == "rate":
                time.sleep(2)
                resp = self.session.get(url, params=params, headers=headers, timeout=15)
            data = self._decode_payload(resp, circuit, f"options exp {symbol}")
            if not data:
                return None
            body = data.get("body")
            if isinstance(body, list) and body:
                opts = body[0].get("options", [])
                if opts:
                    circuit.record_success()
                    return opts[0]
        except ProviderExhausted:
            raise
        except Exception:
            return None
        return None

class YahooDirectAPI:
    """
    Direct Yahoo Finance v8 chart API -- fallback for quick screening
    and options chains. Bypasses yfinance library.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.HEADERS)

    def quick_quote(self, symbol: str) -> Optional[Dict]:
        """Get a quick 3-month snapshot for pre-screening."""
        url = f"{self.BASE_URL}/{symbol}"
        params = {"range": "3mo", "interval": "1d", "includePrePost": "false"}

        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            result = data.get("chart", {}).get("result")
            if not result:
                return None

            chart = result[0]
            meta = chart.get("meta", {})
            quote = chart.get("indicators", {}).get("quote", [{}])[0]
            closes = [c for c in (quote.get("close") or []) if c is not None]
            volumes = [v for v in (quote.get("volume") or []) if v is not None]

            if len(closes) < 20 or len(volumes) < 20:
                return None

            return {
                "price": meta.get("regularMarketPrice", closes[-1]),
                "first_close": closes[0],
                "last_close": closes[-1],
                "avg_volume_20d": np.mean(volumes[-20:]),
                "bars": len(closes),
            }

        except Exception:
            return None

    def get_history(self, symbol: str, range_: str = "5y") -> Optional[pd.DataFrame]:
        """Full daily OHLCV via Yahoo v8 chart -- last-resort public source."""
        circuit = ProviderCircuit.get("Yahoo-OHLCV", fail_limit=40)
        if not circuit.available():
            return None
        url = f"{self.BASE_URL}/{symbol}"
        params = {
            "range": range_,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        }
        try:
            resp = self.session.get(url, params=params, timeout=15)
            kind = classify_http_error(resp.status_code, resp.text or "")
            if kind == "rate":
                time.sleep(1.5)
                resp = self.session.get(url, params=params, timeout=15)
                kind = classify_http_error(resp.status_code, resp.text or "")
            if kind in ("credit", "auth") or resp.status_code in (401, 403):
                circuit.trip(f"Yahoo chart HTTP {resp.status_code}")
                raise ProviderExhausted("Yahoo", f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                return None
            payload = resp.json() or {}
            result = (payload.get("chart") or {}).get("result")
            if not result:
                return None
            chart = result[0]
            ts = chart.get("timestamp") or []
            quote = (chart.get("indicators") or {}).get("quote", [{}])[0]
            adj_list = ((chart.get("indicators") or {}).get("adjclose") or [{}])
            adjclose = (adj_list[0] or {}).get("adjclose") if adj_list else None
            rows = []
            for i, epoch in enumerate(ts):
                closes = quote.get("close") or []
                close = closes[i] if i < len(closes) else None
                adj = adjclose[i] if adjclose and i < len(adjclose) else None
                if close is None and adj is None:
                    continue
                close = normalize_api_scalar(close)
                adj = normalize_api_scalar(adj if adj is not None else close)
                factor = safe_div(adj, close, default=1.0)
                if is_missing_value(factor) or factor <= 0:
                    factor = 1.0
                opens = quote.get("open") or []
                highs = quote.get("high") or []
                lows = quote.get("low") or []
                vols = quote.get("volume") or []
                rows.append({
                    "Date": pd.to_datetime(epoch, unit="s", utc=True),
                    "Open": normalize_api_scalar(opens[i] if i < len(opens) else None) * factor,
                    "High": normalize_api_scalar(highs[i] if i < len(highs) else None) * factor,
                    "Low": normalize_api_scalar(lows[i] if i < len(lows) else None) * factor,
                    "Close": adj,
                    "Volume": vols[i] if i < len(vols) else None,
                })
            if not rows:
                return None
            df = pd.DataFrame(rows).set_index("Date")
            return _normalize_ohlcv_frame(df)
        except ProviderExhausted:
            raise
        except Exception:
            return None


class MarketDataRouter:
    """
    Ordered live-data cascade.

    MBOUM is always attempted first when a key is present and the circuit is
    closed. Restored credits on a later Actions run automatically put MBOUM
    back in front -- circuits are process-local.
    """

    OHLCV_CHAIN = (
        ("MBOUM", "MBOUM-OHLCV", 4),
        ("Massive", "Massive-OHLCV", 10),
        ("TwelveData", "TwelveData-OHLCV", 8),
        ("Finnhub", "Finnhub-OHLCV", 3),
        ("Yahoo", "Yahoo-OHLCV", 40),
    )

    def __init__(self):
        self.mboum = MboumAPI() if MBOUM_API_KEY else None
        self.yahoo = YahooDirectAPI()
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=30, pool_maxsize=30)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _provider_enabled(self, label: str) -> bool:
        if label == "MBOUM":
            return bool(MBOUM_API_KEY) and self.mboum is not None
        if label == "Massive":
            return bool(MASSIVE_API_KEY)
        if label == "TwelveData":
            return bool(TWELVEDATA_API_KEY)
        if label == "Finnhub":
            return bool(FINNHUB_API_KEY)
        return True

    def get_history(self, symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
        fetchers = {
            "MBOUM": self._history_mboum,
            "Massive": self._history_massive,
            "TwelveData": self._history_twelvedata,
            "Finnhub": self._history_finnhub,
            "Yahoo": self._history_yahoo,
        }
        for label, circuit_name, fail_limit in self.OHLCV_CHAIN:
            if not self._provider_enabled(label):
                continue
            circuit = ProviderCircuit.get(circuit_name, fail_limit=fail_limit)
            if not circuit.available():
                continue
            try:
                df = fetchers[label](symbol)
            except ProviderExhausted as exc:
                log.warning(
                    f"  {label} unavailable for OHLCV ({exc.reason}); "
                    "continuing down the fallback chain"
                )
                continue
            except Exception as exc:
                if label != "Yahoo":
                    circuit.record_failure(str(exc))
                log.debug(f"  {label} history failed for {symbol}: {exc}")
                continue
            if df is not None and len(df) >= MIN_TRADING_DAYS:
                circuit.record_success()
                DATA_SOURCE_USAGE["ohlcv"][label] += 1
                df.attrs["source"] = label
                return df, label
            if df is not None and len(df) > 0:
                circuit.record_success()
                continue
            if label != "Yahoo":
                circuit.record_failure(f"{symbol}: empty history")
        return None, ""

    def _history_mboum(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.mboum.get_history(symbol) if self.mboum else None

    def _history_massive(self, symbol: str) -> Optional[pd.DataFrame]:
        circuit = ProviderCircuit.get("Massive-OHLCV", fail_limit=10)
        start, end = _history_window()
        url = (
            f"{MASSIVE_BASE_URL}/aggs/ticker/{symbol}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": MASSIVE_API_KEY,
        }
        resp = None
        for attempt in range(3):
            resp = self.session.get(url, params=params, timeout=20)
            kind = classify_http_error(resp.status_code, resp.text or "")
            if kind == "rate":
                time.sleep(1.5 * (attempt + 1))
                continue
            if kind in ("credit", "auth"):
                circuit.trip(f"Massive aggs HTTP {resp.status_code}")
                raise ProviderExhausted("Massive", f"HTTP {resp.status_code}")
            break
        if resp is None or resp.status_code != 200:
            return None
        payload = resp.json() or {}
        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            return None
        rows = []
        for bar in results:
            ts = bar.get("t")
            if ts is None:
                continue
            rows.append({
                "Date": pd.to_datetime(ts, unit="ms", utc=True),
                "Open": bar.get("o"),
                "High": bar.get("h"),
                "Low": bar.get("l"),
                "Close": bar.get("c"),
                "Volume": bar.get("v"),
            })
        if not rows:
            return None
        return _normalize_ohlcv_frame(pd.DataFrame(rows).set_index("Date"))

    def _history_twelvedata(self, symbol: str) -> Optional[pd.DataFrame]:
        circuit = ProviderCircuit.get("TwelveData-OHLCV", fail_limit=8)
        start, end = _history_window()
        params = {
            "symbol": symbol,
            "interval": "1day",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "outputsize": 5000,
            "order": "asc",
            "adjust": "all",
            "apikey": TWELVEDATA_API_KEY,
        }
        resp = None
        for attempt in range(3):
            resp = self.session.get(
                f"{TWELVEDATA_BASE_URL}/time_series", params=params, timeout=20
            )
            body_text = resp.text or ""
            payload = {}
            try:
                payload = resp.json() if body_text else {}
            except Exception:
                payload = {}
            msg = str(payload.get("message") or "")
            code = payload.get("code") if isinstance(payload.get("code"), int) else resp.status_code
            kind = classify_http_error(code, msg or body_text)
            if payload.get("status") == "error" or kind != "other":
                if kind == "rate":
                    time.sleep(8 * (attempt + 1))
                    continue
                if kind in ("credit", "auth"):
                    circuit.trip(msg or f"TwelveData HTTP {code}")
                    raise ProviderExhausted("TwelveData", msg or str(code))
                if payload.get("status") == "error":
                    return None
            if resp.status_code == 200:
                break
        if resp is None or resp.status_code != 200:
            return None
        payload = resp.json() or {}
        values = payload.get("values") or []
        if not isinstance(values, list) or not values:
            return None
        rows = []
        for bar in values:
            if not isinstance(bar, dict):
                continue
            rows.append({
                "Date": bar.get("datetime"),
                "Open": bar.get("open"),
                "High": bar.get("high"),
                "Low": bar.get("low"),
                "Close": bar.get("close"),
                "Volume": bar.get("volume"),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        return _normalize_ohlcv_frame(df.set_index("Date"))

    def _history_finnhub(self, symbol: str) -> Optional[pd.DataFrame]:
        circuit = ProviderCircuit.get("Finnhub-OHLCV", fail_limit=3)
        start, end = _history_window()
        start_ts = int(datetime.combine(start, dtime.min, tzinfo=ET_TZ).timestamp())
        end_ts = int(datetime.combine(end, dtime(23, 59), tzinfo=ET_TZ).timestamp())
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": start_ts,
            "to": end_ts,
            "token": FINNHUB_API_KEY,
        }
        resp = self.session.get(f"{FINNHUB_BASE_URL}/stock/candle", params=params, timeout=20)
        kind = classify_http_error(resp.status_code, resp.text or "")
        if kind == "rate":
            time.sleep(2)
            resp = self.session.get(f"{FINNHUB_BASE_URL}/stock/candle", params=params, timeout=20)
            kind = classify_http_error(resp.status_code, resp.text or "")
        if kind in ("credit", "auth") or resp.status_code in (401, 403):
            circuit.trip(f"Finnhub candles HTTP {resp.status_code}")
            raise ProviderExhausted("Finnhub", f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        if payload.get("s") != "ok":
            return None
        closes = payload.get("c") or []
        times = payload.get("t") or []
        if len(closes) < 50 or not times:
            return None
        rows = []
        for i, epoch in enumerate(times):
            rows.append({
                "Date": pd.to_datetime(epoch, unit="s", utc=True),
                "Open": (payload.get("o") or [None])[i] if i < len(payload.get("o") or []) else None,
                "High": (payload.get("h") or [None])[i] if i < len(payload.get("h") or []) else None,
                "Low": (payload.get("l") or [None])[i] if i < len(payload.get("l") or []) else None,
                "Close": closes[i] if i < len(closes) else None,
                "Volume": (payload.get("v") or [None])[i] if i < len(payload.get("v") or []) else None,
            })
        if not rows:
            return None
        return _normalize_ohlcv_frame(pd.DataFrame(rows).set_index("Date"))

    def _history_yahoo(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.yahoo.get_history(symbol, range_="5y")

    def yfinance_batch(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Last-resort public batch download via yfinance."""
        loaded: Dict[str, pd.DataFrame] = {}
        if not tickers:
            return loaded
        log.info(f"  yfinance last-resort batch for {len(tickers)} remaining tickers")
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            raw = None
            try:
                raw = yf.download(
                    batch,
                    period="5y",
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as exc:
                log.debug(f"  yfinance batch failed: {exc}")
            for ticker in batch:
                df = _frame_from_yfinance(raw, ticker) if raw is not None else None
                if df is None:
                    df = self._history_yfinance_one(ticker)
                if df is not None and len(df) >= MIN_TRADING_DAYS:
                    DATA_SOURCE_USAGE["ohlcv"]["yfinance"] += 1
                    df.attrs["source"] = "yfinance"
                    loaded[ticker] = df
        return loaded

    def _history_yfinance_one(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(
                symbol,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return _frame_from_yfinance(raw, symbol)
        except Exception:
            return None


def _frame_from_yfinance(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if raw is None or getattr(raw, "empty", True):
        return None
    df = raw
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(df.columns.get_level_values(0))
        if ticker in level0:
            df = df[ticker]
        elif "Close" in level0:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        else:
            return None
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return _normalize_ohlcv_frame(df)


_MARKET_ROUTER: Optional[MarketDataRouter] = None
_MARKET_ROUTER_LOCK = threading.Lock()


def get_market_router() -> MarketDataRouter:
    global _MARKET_ROUTER
    with _MARKET_ROUTER_LOCK:
        if _MARKET_ROUTER is None:
            _MARKET_ROUTER = MarketDataRouter()
        return _MARKET_ROUTER


def reset_market_router() -> None:
    global _MARKET_ROUTER
    with _MARKET_ROUTER_LOCK:
        _MARKET_ROUTER = None
    ProviderCircuit.reset_all()
    reset_data_source_usage()


class DataFetcher:
    """
    Two-phase data fetcher:
      Phase 1: Yahoo Direct API quick 3-month screen (fast, free, parallel)
      Phase 2: MBOUM Pro full history, then Massive / TwelveData / Finnhub /
               Yahoo / yfinance if MBOUM is out of credits.
    """

    def __init__(self, lookback_days: int = LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        self.yahoo = YahooDirectAPI()
        self.router = get_market_router()

    def fetch_ohlcv(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Two-phase download:
          1. Yahoo Direct quick 3mo screen (parallel, fast)
          2. Full 5y OHLCV with MBOUM primary and live fallbacks
        """
        log.info(f"PHASE 1: Quick 3-month pre-screen for {len(tickers)} tickers")
        promising = self._quick_screen(tickers)
        log.info(
            f"PHASE 1 COMPLETE: {len(promising)} tickers passed quick screen "
            f"(positive 3mo return, price >= $5, avg dollar vol > $1M)"
        )

        if not promising:
            raise PipelineError("No tickers passed quick screen. Pipeline STOPPED.")

        log.info(
            f"PHASE 2: Full OHLCV download for {len(promising)} tickers "
            "(MBOUM primary; Massive/TwelveData/Finnhub/Yahoo fallbacks)"
        )
        all_data = self._full_download(promising)
        return all_data

    def _quick_screen(self, tickers: List[str]) -> List[str]:
        """
        Parallel quick screen using Yahoo v8 3mo range.
        Keeps tickers with: positive 3mo return, close >= $5, avg dollar vol >= $1M.
        """
        promising = []
        checked = 0

        def _check_one(ticker: str) -> Optional[str]:
            q = self.yahoo.quick_quote(ticker)
            if q is None:
                return None
            price = q["last_close"]
            first = q["first_close"]
            if price < 5.0:
                return None
            if first <= 0:
                return None
            ret = (price / first) - 1
            if ret < 0:
                return None
            avg_dv = price * q["avg_volume_20d"]
            if avg_dv < 1_000_000:
                return None
            return ticker

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(_check_one, t): t for t in tickers
            }

            for future in as_completed(futures):
                checked += 1
                if checked % 500 == 0 or checked == len(tickers):
                    log.info(
                        f"  Quick screen: {checked}/{len(tickers)} checked, "
                        f"{len(promising)} promising"
                    )
                # A single malformed payload must never abort the screen of a
                # 10,000-name universe.
                try:
                    result = future.result()
                except Exception as e:
                    log.debug(f"  Quick screen failed for {futures[future]}: {e}")
                    continue
                if result:
                    promising.append(result)

        return promising

    def _full_download(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download full history with MBOUM primary and live provider fallbacks."""
        all_data: Dict[str, pd.DataFrame] = {}
        failed = 0
        missing: List[str] = []

        def _download_one(ticker: str) -> Tuple[str, Optional[pd.DataFrame], str]:
            df, source = self.router.get_history(ticker)
            if df is not None and len(df) >= MIN_TRADING_DAYS:
                return ticker, df, source
            return ticker, None, source

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(_download_one, t): t for t in tickers
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                ticker = futures[future]
                try:
                    ticker, df, _source = future.result()
                except Exception as e:
                    log.debug(f"  OHLCV history failed for {ticker}: {e}")
                    failed += 1
                    missing.append(ticker)
                    continue
                if df is not None:
                    all_data[ticker] = df
                else:
                    failed += 1
                    missing.append(ticker)

                if completed % 100 == 0 or completed == len(tickers):
                    sources = ", ".join(
                        f"{name}={count}"
                        for name, count in DATA_SOURCE_USAGE["ohlcv"].most_common()
                    ) or "none yet"
                    log.info(
                        f"  OHLCV download: {completed}/{len(tickers)} done, "
                        f"{len(all_data)} loaded, {failed} pending/failed "
                        f"[{sources}]"
                    )

        if missing:
            recovered = self.router.yfinance_batch(missing)
            for ticker, df in recovered.items():
                all_data[ticker] = df
            failed = max(0, len(tickers) - len(all_data))

        sources = ", ".join(
            f"{name}={count}"
            for name, count in DATA_SOURCE_USAGE["ohlcv"].most_common()
        ) or "none"
        log.info(
            f"Full OHLCV fetch complete: {len(all_data)} tickers with "
            f">= {MIN_TRADING_DAYS} trading days. "
            f"{failed} excluded (insufficient history). Sources: {sources}"
        )

        # CRITICAL FIX: drop intraday partial-bars when market is currently
        # open. Today's in-progress bar has incomplete volume which
        # systematically breaks volume-surge / volume-ratio rules and yields
        # zero survivors during RTH. We always operate on closed bars.
        if is_market_open_now():
            trimmed = 0
            for t in list(all_data.keys()):
                df = all_data[t]
                if last_bar_is_partial(df):
                    if len(df) > 1:
                        all_data[t] = df.iloc[:-1].copy()
                        trimmed += 1
                    else:
                        del all_data[t]
            if trimmed:
                log.info(
                    f"  Trimmed {trimmed} intraday partial bars "
                    f"(market currently open ET) -- using last fully-closed session."
                )

        if len(all_data) == 0:
            raise PipelineError(
                "No tickers had sufficient OHLCV data. Pipeline STOPPED."
            )

        if len(tickers) > 0:
            pct = len(all_data) / len(tickers) * 100
            if len(all_data) < MIN_UNIVERSE_SIZE and pct < 80:
                raise PipelineError(
                    f"Only {len(all_data)}/{len(tickers)} ({pct:.0f}%) tickers "
                    f"have valid data (< 80% threshold). Pipeline STOPPED."
                )
            elif pct < 80:
                log.warning(
                    f"Proceeding with {len(all_data)} tickers ({pct:.0f}% of universe)"
                )

        return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TechnicalEngine:
    """Computes all technical indicators required by the pipeline."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder-smoothed RSI."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Wilder's convention: with no average loss in the window RSI is 100,
        # not undefined. Returning NaN here silently disqualified the strongest
        # uninterrupted uptrends from every RSI-based rule and panel score.
        no_loss = (avg_loss == 0) & avg_gain.notna()
        return rsi.mask(no_loss, 100.0)

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        """Returns (macd_line, signal_line, histogram)."""
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period=20, std_dev=2):
        """Returns (upper, middle, lower)."""
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def vwap_daily(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Rolling N-day VWAP using typical price (H+L+C)/3 weighted by volume.
        For daily bars, a rolling 20-day VWAP is the institutional reference
        used by professional desks (Markert/Almgren) for entry-quality. A
        rolling(1) implementation collapses to (H+L+C)/3 which is meaningless
        as a VWAP confirmation signal.
        """
        tp = (high + low + close) / 3
        tpv = tp * volume
        cumtp_vol = tpv.rolling(window=window, min_periods=max(2, window // 2)).sum()
        cum_vol = volume.rolling(window=window, min_periods=max(2, window // 2)).sum()
        return cumtp_vol / cum_vol.replace(0, np.nan)

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's Average Directional Index -- trend strength gauge.
        ADX > 25 indicates a real trend (Wilder); used by Minervini/Livermore
        equivalents to filter range-bound chop."""
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=low.index)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators on an OHLCV DataFrame.
        Adds columns in-place and returns the DataFrame.
        """
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]

        # SMAs
        df["SMA_10"] = TechnicalEngine.sma(c, 10)
        df["SMA_20"] = TechnicalEngine.sma(c, 20)
        df["SMA_30"] = TechnicalEngine.sma(c, 30)
        df["SMA_50"] = TechnicalEngine.sma(c, 50)
        df["SMA_200"] = TechnicalEngine.sma(c, 200)

        # EMAs
        df["EMA_10"] = TechnicalEngine.ema(c, 10)
        df["EMA_20"] = TechnicalEngine.ema(c, 20)
        df["EMA_50"] = TechnicalEngine.ema(c, 50)

        # RSI
        df["RSI_14"] = TechnicalEngine.rsi(c, 14)

        # MACD
        macd_line, signal_line, histogram = TechnicalEngine.macd(c)
        df["MACD_line"] = macd_line
        df["MACD_signal"] = signal_line
        df["MACD_histogram"] = histogram
        # MACD in price units is meaningless across a universe: the same
        # momentum reads ~100x larger on a $900 name than a $9 one. The rules
        # only test its sign, so raw is fine there, but a cross-sectional model
        # needs it per-dollar-of-price to compare names at all.
        df["MACD_hist_pct"] = histogram / c.replace(0, np.nan)

        # Bollinger Bands
        upper, middle, lower = TechnicalEngine.bollinger_bands(c)
        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower

        # VWAP (daily proxy)
        df["VWAP"] = TechnicalEngine.vwap_daily(h, l, c, v)

        # Volume SMA
        df["Vol_SMA_20"] = TechnicalEngine.sma(v, 20)

        # Average dollar volume (20d)
        df["Dollar_Volume"] = c * v
        df["Avg_Dollar_Vol_20"] = TechnicalEngine.sma(df["Dollar_Volume"], 20)
        # Liquidity spans $5M to $50B+ across the universe -- four orders of
        # magnitude, almost all of the mass at the bottom. Standardising that
        # raw leaves a feature that is ~0 for every name except a handful of
        # mega-caps. On a log scale the same variable separates a $6M name from
        # a $60M one as readily as $6B from $60B, which is how liquidity
        # actually differs.
        df["Log_Dollar_Vol_20"] = np.log10(
            df["Avg_Dollar_Vol_20"].clip(lower=1.0)
        )

        # Rolling 20-day high close
        df["High_Close_20"] = c.rolling(window=20, min_periods=20).max()

        # Returns
        df["Return_1d"] = c.pct_change(1)
        df["Return_5d"] = c.pct_change(5)
        df["Return_20d"] = c.pct_change(20)
        df["Return_63d"] = c.pct_change(63)

        # Volume ratio
        df["Volume_Ratio"] = v / df["Vol_SMA_20"].replace(0, np.nan)

        # Close vs SMAs (pct)
        df["Close_vs_SMA50"] = (c - df["SMA_50"]) / df["SMA_50"].replace(0, np.nan)
        df["Close_vs_SMA200"] = (c - df["SMA_200"]) / df["SMA_200"].replace(0, np.nan)

        # EMA spread
        df["EMA20_vs_EMA50"] = (
            (df["EMA_20"] - df["EMA_50"]) / df["EMA_50"].replace(0, np.nan)
        )

        # OBV (On Balance Volume) -- for panel scoring
        obv = pd.Series(0.0, index=df.index)
        close_diff = c.diff()
        obv = v.where(close_diff > 0, -v.where(close_diff < 0, 0)).cumsum()
        df["OBV"] = obv

        # ATR (14) for volatility contraction scoring
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(window=14, min_periods=14).mean()
        df["ATR_pct"] = df["ATR_14"] / c.replace(0, np.nan)

        # ADX (14) -- trend strength filter (Wilder)
        df["ADX_14"] = TechnicalEngine.adx(h, l, c, 14)

        # Realized volatility (20d, annualized) -- for IV/RV ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c / c.shift(1)
            ratio = ratio.where(ratio > 0)  # drop non-positive ratios
            log_ret = np.log(ratio)
        df["RVol_20"] = log_ret.rolling(20, min_periods=20).std() * np.sqrt(252)

        # Distance to 52-week high / low (fundamental break-out reference)
        roll_high_252 = c.rolling(window=252, min_periods=126).max()
        roll_low_252 = c.rolling(window=252, min_periods=126).min()
        df["Pct_From_52w_High"] = (c - roll_high_252) / roll_high_252.replace(0, np.nan)
        df["Pct_Above_52w_Low"] = (c - roll_low_252) / roll_low_252.replace(0, np.nan)

        # ── Exhaustion / "already peaked" diagnostics ────────────────────────
        # Objective is to ride short-to-mid-term momentum, not to buy the last
        # tick of a vertical move. These columns quantify how stretched a name
        # is versus its own trend so blow-off tops can be excluded downstream.
        df["Ext_Above_SMA20"] = (c - df["SMA_20"]) / df["SMA_20"].replace(0, np.nan)
        # Distance from the 20-day mean measured in ATRs: regime-independent
        # and directly comparable across a $9 name and a $900 name.
        df["Stretch_ATR"] = (c - df["SMA_20"]) / df["ATR_14"].replace(0, np.nan)
        # Upper-Bollinger penetration: >1 means trading beyond the 2-sigma band.
        bb_width = (df["BB_upper"] - df["BB_middle"]).replace(0, np.nan)
        df["BB_Penetration"] = (c - df["BB_middle"]) / bb_width

        # ── Hype / crowd-participation diagnostics ───────────────────────────
        # Relative volume over a full week captures sustained crowding rather
        # than a single headline print.
        df["RVOL_5"] = (
            v.rolling(window=5, min_periods=5).mean()
            / df["Vol_SMA_20"].replace(0, np.nan)
        )
        # Rising OBV confirms the crowd is accumulating, not distributing.
        obv_sma20 = df["OBV"].rolling(window=20, min_periods=20).mean()
        df["OBV_Trend"] = df["OBV"] - obv_sma20

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM QUALITY -- hype, exhaustion and insider conviction
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumQuality:
    """
    Quantifies the three things that decide whether a strong chart is worth
    real capital over a short-to-mid-term hold:

      * hype_score       -- is the crowd actually participating (volume,
                            relative strength, squeeze fuel)?
      * exhaustion_score -- has the move already gone too far, too fast?
      * insider_score    -- are the people who know the business buying it?

    All three are 0-100 and computed strictly from live data. Missing inputs
    return a neutral 50 rather than a flattering default, so an absent data
    source can never manufacture conviction.
    """

    NEUTRAL = 50.0

    @staticmethod
    def hype_score(last: pd.Series) -> float:
        """
        0-100 measure of genuine crowd participation behind the move.

        Blends sustained relative volume, single-day volume surge, 20-day
        return and on-balance-volume accumulation. High values mean money is
        actively rotating in -- the momentum/hype the strategy wants to ride.
        """
        components: List[float] = []

        rvol5 = last.get("RVOL_5", np.nan)
        if not pd.isna(rvol5):
            # 1.0x = market-neutral participation, 2.5x = genuinely crowded.
            components.append(100.0 * clamp((float(rvol5) - 0.8) / 1.7, 0.0, 1.0))

        vol_ratio = last.get("Volume_Ratio", np.nan)
        if not pd.isna(vol_ratio):
            components.append(100.0 * clamp((float(vol_ratio) - 0.9) / 2.1, 0.0, 1.0))

        ret_20d = last.get("Return_20d", np.nan)
        if not pd.isna(ret_20d):
            components.append(100.0 * clamp(float(ret_20d) / 0.20, 0.0, 1.0))

        obv_trend = last.get("OBV_Trend", np.nan)
        if not pd.isna(obv_trend):
            # Direction matters more than magnitude; magnitude is unbounded.
            components.append(70.0 if float(obv_trend) > 0 else 30.0)

        if not components:
            return MomentumQuality.NEUTRAL
        return float(clamp(np.mean(components), 0.0, 100.0))

    @staticmethod
    def exhaustion_score(last: pd.Series) -> float:
        """
        0-100 measure of how stretched the move already is.

        100 means "this has gone vertical and is statistically due to unwind";
        0 means "the trend has room". Used to penalise, and at the extreme to
        reject, names that are already peaking.
        """
        components: List[float] = []

        ext_sma50 = last.get("Close_vs_SMA50", np.nan)
        if not pd.isna(ext_sma50):
            components.append(
                100.0 * clamp(float(ext_sma50) / MAX_EXTENSION_ABOVE_SMA50, 0.0, 1.0)
            )

        stretch = last.get("Stretch_ATR", np.nan)
        if not pd.isna(stretch):
            # Beyond ~4 ATR above the 20-day mean is classic climax territory.
            components.append(100.0 * clamp(float(stretch) / 4.0, 0.0, 1.0))

        rsi = last.get("RSI_14", np.nan)
        if not pd.isna(rsi):
            # Below 65 is unremarkable; 65 -> MAX_EXHAUSTION_RSI ramps to 100.
            span = max(MAX_EXHAUSTION_RSI - 65.0, 1.0)
            components.append(100.0 * clamp((float(rsi) - 65.0) / span, 0.0, 1.0))

        ret_5d = last.get("Return_5d", np.nan)
        if not pd.isna(ret_5d):
            components.append(
                100.0 * clamp(float(ret_5d) / MAX_SPIKE_RETURN_5D, 0.0, 1.0)
            )

        bb_pen = last.get("BB_Penetration", np.nan)
        if not pd.isna(bb_pen):
            # >1 is outside the upper 2-sigma band; 2.0 is a true blow-off.
            components.append(100.0 * clamp((float(bb_pen) - 0.5) / 1.5, 0.0, 1.0))

        if not components:
            return MomentumQuality.NEUTRAL
        return float(clamp(np.mean(components), 0.0, 100.0))

    @staticmethod
    def insider_score(fund: Optional[Dict]) -> float:
        """
        0-100 conviction score from live insider transaction data.

        Insider *buying* is the highest-signal fundamental input available on a
        short horizon: officers and directors sell for many reasons but buy for
        exactly one. Returns a neutral 50 when the provider has no insider
        record so a data gap neither rewards nor punishes a name.
        """
        fund = fund or {}
        net_pct = normalize_api_scalar(fund.get("insider_net_purchase_pct"))
        buys = normalize_api_scalar(fund.get("insider_buy_transactions"))
        sells = normalize_api_scalar(fund.get("insider_sell_transactions"))

        components: List[float] = []

        if not is_missing_value(net_pct):
            # Net shares purchased as a fraction of insider holdings. +2% is a
            # strong accumulation signal, -2% is meaningful distribution.
            components.append(50.0 + 50.0 * clamp(float(net_pct) / 0.02, -1.0, 1.0))

        if not is_missing_value(buys) and not is_missing_value(sells):
            total = float(buys) + float(sells)
            if total > 0:
                buy_ratio = float(buys) / total
                components.append(100.0 * clamp(buy_ratio, 0.0, 1.0))

        if not components:
            return MomentumQuality.NEUTRAL
        return float(clamp(np.mean(components), 0.0, 100.0))

    @staticmethod
    def squeeze_score(fund: Optional[Dict]) -> float:
        """
        0-100 short-squeeze fuel from live short interest.

        Elevated short interest on a name that is already trending is the
        classic accelerant behind the sharp short-to-mid-term moves this
        strategy targets.
        """
        fund = fund or {}
        short_pct = normalize_api_scalar(fund.get("short_pct_float"))
        if is_missing_value(short_pct):
            return MomentumQuality.NEUTRAL
        # 0% float short -> 0, 20%+ short -> 100.
        return float(100.0 * clamp(float(short_pct) / 0.20, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 -- EXECUTION QUALITY GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionGuards:
    """Pre-admission filters: data integrity, liquidity, spread, performance."""

    @staticmethod
    def apply(
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Apply all execution guards. Returns:
          - filtered dict of surviving ticker DataFrames
          - dict of rejected tickers with reason
        """
        log.info(f"STAGE 2: Execution Guards -- screening {len(data)} tickers")
        survivors = {}
        rejected = {}

        for ticker, df in data.items():
            reason = ExecutionGuards._check(ticker, df)
            if reason:
                rejected[ticker] = reason
            else:
                survivors[ticker] = df

        log.info(
            f"STAGE 2 COMPLETE: {len(survivors)} passed, "
            f"{len(rejected)} rejected"
        )
        return survivors, rejected

    # Real-money execution thresholds. Setting to $5M avg dollar volume
    # ensures retail-size positions (typical $5-50K) move <5bp at the
    # institutional spread, per Almgren-Chriss execution-cost models.
    MIN_DOLLAR_VOLUME = 5_000_000

    @staticmethod
    def ml_training_pool(
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Universe for *fitting* the ML models -- deliberately not the guarded set.

        The guards admit a name partly on its latest 63-day return. Training on
        that set conditions every historical row on performance that had not
        happened yet when the row was printed: the model only ever sees a
        January bar because the name went on to rally through August. That is
        look-ahead selection. It inflates the positive class, flatters CV
        accuracy, and teaches the ranker patterns that are an artifact of the
        filter rather than of the market.

        This pool only drops series that are degenerate end to end (too short,
        halted/flat). The price and liquidity screen is applied per-row inside
        MLRanker._build_dataset, using each bar's own close and trailing
        20-day dollar volume, so eligibility is judged with what was knowable
        on that bar rather than with today's values. Scoring still happens
        only on fully guarded candidates -- this changes what the model learns
        from, not what it is allowed to buy.
        """
        pool = {}
        for ticker, df in data.items():
            if df is None or df.empty or len(df) < MIN_TRADING_DAYS:
                continue
            # Halted / delisted-style series carry flat prices and a trivially
            # negative label, which is noise rather than signal. These are the
            # same integrity conditions GUARD_A uses, and unlike the 63-day
            # return they say nothing about how the name went on to perform.
            recent = df.iloc[-30:] if len(df) >= 30 else df
            if (recent["Volume"] <= 0).all() or recent["Close"].nunique() <= 2:
                continue
            pool[ticker] = df
        return pool

    @staticmethod
    def _check(
        ticker: str, df: pd.DataFrame, now: Optional[datetime] = None
    ) -> Optional[str]:
        """Returns rejection reason or None if passes all guards."""
        if df.empty or len(df) < MIN_TRADING_DAYS:
            return "GUARD_A: Insufficient data"

        last = df.iloc[-1]

        # GUARD_A: Data Integrity -- check for holes in the recent history.
        # The window must be measured against the ticker's OWN span: comparing
        # a full multi-year MBOUM history against a fixed 252 made this ratio
        # permanently negative, so the check never fired.
        recent = df.tail(MIN_TRADING_DAYS)
        actual_bars = recent["Close"].dropna().shape[0]
        try:
            span_start = pd.Timestamp(recent.index[0]).date()
            span_end = pd.Timestamp(recent.index[-1]).date()
            expected_bars = trading_sessions_between(span_start, span_end) + 1
        except Exception:
            expected_bars = 0
        if expected_bars > 0:
            missing_pct = 1 - (actual_bars / expected_bars)
            if missing_pct > 0.05:
                return f"GUARD_A: {missing_pct:.1%} bars missing (> 5%)"

        # GUARD_A: Check for stale data (last bar should be recent in ET).
        # Uses ET timezone -- prevents false stale flags when run on a UTC
        # server before US market data has rolled over locally.
        try:
            last_date = pd.Timestamp(df.index[-1]).date()
        except Exception:
            return "GUARD_A: Last bar timestamp invalid"
        expected_day = expected_last_closed_trading_day(now)
        lag_sessions = trading_sessions_between(last_date, expected_day)
        if lag_sessions > 0:
            return (
                "GUARD_A: Stale data "
                f"({lag_sessions} trading sessions behind expected {expected_day})"
            )

        # GUARD_A: Reject zero-volume or constant-price bars (delisted/halted)
        last_30 = df.iloc[-30:] if len(df) >= 30 else df
        if (last_30["Volume"] <= 0).all():
            return "GUARD_A: All recent volume zero (suspended/halted)"
        if last_30["Close"].nunique() <= 2:
            return "GUARD_A: Stale price (< 3 unique closes in 30 days)"

        # GUARD_B: Liquidity -- avg daily dollar volume over 20 days.
        # $5M minimum gives institutional-grade execution; cheaper names get
        # routed away on principle (real-money policy).
        avg_dv = last.get("Avg_Dollar_Vol_20", 0)
        if pd.isna(avg_dv) or avg_dv < ExecutionGuards.MIN_DOLLAR_VOLUME:
            return f"GUARD_B: Low liquidity (avg $vol ${avg_dv:,.0f} < ${ExecutionGuards.MIN_DOLLAR_VOLUME:,.0f})"

        # GUARD_B: 3-month performance must be positive
        ret_63d = last.get("Return_63d", np.nan)
        if pd.isna(ret_63d) or ret_63d < 0:
            return f"GUARD_B: Negative 3mo return ({ret_63d:.2%})"

        # GUARD_B2: Avoid hyper-volatile names (ATR/price > 12%) -- options
        # premiums become unaffordable and equity risk uncontrollable.
        atr_pct = last.get("ATR_pct", np.nan)
        if not pd.isna(atr_pct) and atr_pct > 0.12:
            return f"GUARD_B: Excessive volatility (ATR {atr_pct:.1%} > 12%)"

        # GUARD_C: Penny stock exclusion
        close = last["Close"]
        if pd.isna(close) or close < 5.0:
            return f"GUARD_C: Penny stock (${close:.2f} < $5)"

        # GUARD_D: Exhaustion / "already peaked" filter.
        # The strategy rides short-to-mid-term momentum but explicitly refuses
        # to chase a move that has already gone vertical. A name trading far
        # above its own 50DMA, printing blow-off RSI, or that has spiked
        # violently in the last week/month has, statistically, spent most of
        # its upside and carries mean-reversion risk that a 20-day hold cannot
        # absorb. Rejecting these is what keeps the top-7 out of the names that
        # tank the day after they are discovered.
        exhaustion = ExecutionGuards._exhaustion_reason(last)
        if exhaustion:
            return f"GUARD_D: {exhaustion}"

        return None  # All guards passed

    @staticmethod
    def _exhaustion_reason(last: pd.Series) -> Optional[str]:
        """
        Return a human-readable reason when the latest bar shows a parabolic,
        already-peaking move, otherwise None.

        Kept as a separate helper so the same definition drives the guard, the
        candidate flags and the final confidence penalty -- one rule, one place.
        """
        ext_sma50 = last.get("Close_vs_SMA50", np.nan)
        if not pd.isna(ext_sma50) and ext_sma50 > MAX_EXTENSION_ABOVE_SMA50:
            return (
                f"Over-extended ({ext_sma50:.0%} above 50DMA "
                f"> {MAX_EXTENSION_ABOVE_SMA50:.0%})"
            )

        rsi = last.get("RSI_14", np.nan)
        if not pd.isna(rsi) and rsi >= MAX_EXHAUSTION_RSI:
            return f"Blow-off RSI ({rsi:.0f} >= {MAX_EXHAUSTION_RSI:.0f})"

        ret_5d = last.get("Return_5d", np.nan)
        if not pd.isna(ret_5d) and ret_5d > MAX_SPIKE_RETURN_5D:
            return (
                f"Vertical 5-day spike ({ret_5d:.0%} > "
                f"{MAX_SPIKE_RETURN_5D:.0%})"
            )

        ret_20d = last.get("Return_20d", np.nan)
        if not pd.isna(ret_20d) and ret_20d > MAX_SPIKE_RETURN_20D:
            return (
                f"Parabolic 20-day run ({ret_20d:.0%} > "
                f"{MAX_SPIKE_RETURN_20D:.0%})"
            )

        # Volatility-normalised extension. The percentage checks above are
        # blind to how much a given name normally moves; this one is not, so a
        # $200 low-beta name that has quietly gone vertical is caught by the
        # same rule that leaves a high-beta breakout alone.
        stretch = last.get("Stretch_ATR", np.nan)
        if not pd.isna(stretch) and stretch > MAX_STRETCH_ATR:
            return (
                f"Climax extension ({stretch:.1f} ATR above the 20-day mean "
                f"> {MAX_STRETCH_ATR:.1f})"
            )

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 -- HARD BUY RULES (ALL 10 MUST PASS)
# ═══════════════════════════════════════════════════════════════════════════════

class HardBuyRules:
    """
    10 hard buy rules. ALL must pass -- no exceptions, no ML override.
    Returns results with per-rule pass/fail and flags.
    """

    # Rules a near-miss backfill may never fail.
    #
    # Most of the 10-rule card is about entry *timing* -- crossover recency,
    # breakout, VWAP reclaim, volume surge -- and a genuinely strong name can
    # miss one of those and still be the better trade. These two are not
    # timing. BUY_01 is the trend itself, so failing it means buying a name
    # below its own 200DMA with a rolling 50DMA, and BUY_08 failing on the
    # high side is the definition of arriving after the move. Backfills are
    # ranked and sized alongside strict passers, so they are held to both.
    CRITICAL_NEAR_MISS_RULES = ("BUY_01", "BUY_08")

    @staticmethod
    def apply(
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Apply all 10 hard buy rules.
        Returns:
          - list of dicts for survivors (ticker + latest data + flags)
          - dict of rejected tickers with failing rule
        """
        log.info(f"STAGE 3: Hard Buy Rules -- testing {len(data)} tickers")
        survivors = []
        rejected = {}
        flags_summary = {}

        for ticker, df in data.items():
            passed, fail_rule, flags = HardBuyRules._evaluate(ticker, df)
            if passed:
                record = HardBuyRules._candidate_record(
                    ticker, df, flags=flags, hard_buy_pass=True
                )
                survivors.append(record)
                if flags:
                    flags_summary[ticker] = flags
            else:
                rejected[ticker] = fail_rule

        log.info(
            f"STAGE 3 COMPLETE: {len(survivors)} survivors passed ALL 10 rules. "
            f"{len(rejected)} rejected."
        )
        if flags_summary:
            log.info(f"  Flags on survivors: {len(flags_summary)} tickers flagged")

        return survivors, rejected

    @staticmethod
    def _candidate_record(
        ticker: str,
        df: pd.DataFrame,
        flags: Optional[List[str]] = None,
        rule_result: Optional[Dict] = None,
        hard_buy_pass: bool = False,
    ) -> Dict:
        """Build the live candidate record used by ML, panel, options, and output."""
        last = df.iloc[-1]
        flags = list(flags or [])
        if rule_result is None:
            rule_result = HardBuyRules._evaluate_all_rules(ticker, df) or {}
        rules_passed = int(rule_result.get("rules_passed", 10 if hard_buy_pass else 0))
        rules_failed = int(rule_result.get("rules_failed", 0 if hard_buy_pass else 10 - rules_passed))
        failed_rules = list(rule_result.get("failed_rules", []))
        if not hard_buy_pass:
            flags.append(f"NEAR_MISS_{rules_passed}_OF_10")

        # Momentum-quality overlays: how crowded the move is (hype) and how
        # stretched it already is (exhaustion). Both are needed to buy strength
        # without buying the top.
        hype = MomentumQuality.hype_score(last)
        exhaustion = MomentumQuality.exhaustion_score(last)
        if exhaustion >= 70:
            flags.append("EXTENDED_MOVE")
        if hype >= 70:
            flags.append("HIGH_CROWD_INTEREST")

        return {
            "ticker": ticker,
            "price": last["Close"],
            "rsi_14": last.get("RSI_14", np.nan),
            "macd_histogram": last.get("MACD_histogram", np.nan),
            "volume_ratio": last.get("Volume_Ratio", np.nan),
            "return_1d": last.get("Return_1d", np.nan),
            "return_5d": last.get("Return_5d", np.nan),
            "return_20d": last.get("Return_20d", np.nan),
            "return_63d": last.get("Return_63d", np.nan),
            "close_vs_sma50": last.get("Close_vs_SMA50", np.nan),
            "close_vs_sma200": last.get("Close_vs_SMA200", np.nan),
            "ema20_vs_ema50": last.get("EMA20_vs_EMA50", np.nan),
            "avg_dollar_volume": last.get("Avg_Dollar_Vol_20", np.nan),
            "rvol_5": last.get("RVOL_5", np.nan),
            "stretch_atr": last.get("Stretch_ATR", np.nan),
            "pct_from_52w_high": last.get("Pct_From_52w_High", np.nan),
            "hype_score": round(hype, 1),
            "exhaustion_score": round(exhaustion, 1),
            "rules_passed": rules_passed,
            "rules_failed": rules_failed,
            "passed_rules": list(rule_result.get("passed_rules", [])),
            "failed_rules": failed_rules,
            "hard_buy_pass": bool(hard_buy_pass),
            "flags": flags,
        }

    @staticmethod
    def build_rank_pool(
        data: Dict[str, pd.DataFrame],
        strict_survivors: List[Dict],
        target_size: int = TARGET_FINAL_CANDIDATES,
        max_pool_size: int = MAX_PANEL_CANDIDATES,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Build a live candidate pool large enough to rank a top-7 output.
        Strict passers remain first-class; best near-misses backfill scarcity.
        """
        pool = list(strict_survivors)
        near_misses: List[Dict] = []
        seen = {s["ticker"] for s in pool}

        if len(pool) >= max_pool_size:
            log.info(
                f"STAGE 3B: Ranked candidate pool -- {len(pool[:max_pool_size])} strict passers"
            )
            return pool[:max_pool_size], near_misses

        needed_pool = max(target_size * 3, max_pool_size - len(pool))
        near_misses = HardBuyRules.near_misses(data, top_n=needed_pool)
        blocked = 0
        for nm in near_misses:
            ticker = nm["ticker"]
            if ticker in seen or nm.get("rules_passed", 0) < MIN_NEAR_MISS_RULES:
                continue
            df = data.get(ticker)
            if df is None:
                continue
            record = HardBuyRules._candidate_record(
                ticker,
                df,
                rule_result=nm,
                hard_buy_pass=False,
            )
            block_reason = HardBuyRules._backfill_block_reason(nm, record)
            if block_reason:
                blocked += 1
                log.debug(f"  Backfill rejected {ticker}: {block_reason}")
                continue
            pool.append(record)
            seen.add(ticker)
            if len(pool) >= max_pool_size:
                break

        log.info(
            f"STAGE 3B: Ranked candidate pool -- {len(pool)} total "
            f"({len(strict_survivors)} strict passers, {len(pool) - len(strict_survivors)} near-miss backfills)"
        )
        if blocked:
            log.info(
                f"  {blocked} near-miss(es) held out of the buy pool: broken "
                "trend, late RSI, or an already-extended move."
            )
        return pool, near_misses

    @staticmethod
    def _backfill_block_reason(nm: Dict, record: Dict) -> Optional[str]:
        """
        Why this near-miss must not be promoted into the buy pool, or None.

        A backfill is traded, not just reported, so it has to clear the parts
        of the card that decide *whether* a name is buyable rather than merely
        *when*. Without this, an 8-of-10 name could reach the top-7 while
        trading below its 200DMA, or with RSI in the 70s -- the exact
        "already ran, we are late" entry the strategy is built to avoid.
        """
        failed = {rule.split(":", 1)[0] for rule in nm.get("failed_rules", [])}
        blocked_rules = [
            rule for rule in HardBuyRules.CRITICAL_NEAR_MISS_RULES if rule in failed
        ]
        if blocked_rules:
            return f"failed critical rule(s) {', '.join(blocked_rules)}"
        # Reuses the exhaustion threshold that already drives EXTENDED_MOVE,
        # so the anti-chase line is defined in exactly one place.
        if "EXTENDED_MOVE" in record.get("flags", []):
            return f"already extended (exhaustion {record.get('exhaustion_score')})"
        return None

    @staticmethod
    def near_misses(
        data: Dict[str, pd.DataFrame], top_n: int = 25
    ) -> List[Dict]:
        """
        Evaluate ALL 10 rules for every ticker without short-circuiting.
        Returns the top_n tickers sorted by most rules passed (descending),
        with full per-rule pass/fail detail.
        Called by build_rank_pool() to backfill the candidate pool when strict
        passers are fewer than the target size; may also run when zero survivors emerge.
        """
        log.info(f"Computing near-miss rankings for {len(data)} tickers...")
        scoreboard = []

        for ticker, df in data.items():
            result = HardBuyRules._evaluate_all_rules(ticker, df)
            if result:
                scoreboard.append(result)

        # Sort by broad rule strength, then momentum/volume quality.
        scoreboard.sort(
            key=lambda x: (
                x["rules_passed"],
                x.get("return_20d", 0) if not pd.isna(x.get("return_20d", np.nan)) else -999,
                x.get("volume_ratio", 0) if not pd.isna(x.get("volume_ratio", np.nan)) else 0,
                -abs((x.get("rsi_14") if not pd.isna(x.get("rsi_14", np.nan)) else 55) - 55),
            ),
            reverse=True,
        )

        near = scoreboard[:top_n]
        for rank, nm in enumerate(near, 1):
            nm["near_miss_rank"] = rank
            log.info(
                f"  #{rank} {nm['ticker']}: {nm['rules_passed']}/10 passed "
                f"-- failed: {', '.join(nm['failed_rules'])}"
            )

        return near

    @staticmethod
    def _evaluate_all_rules(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate ALL 10 hard buy rules without short-circuiting.
        Returns a dict with pass count, fail list, and latest data.
        """
        if df.empty or len(df) < 10:
            return None

        last = df.iloc[-1]
        close = last.get("Close", np.nan)
        if pd.isna(close):
            return None

        passed_rules = []
        failed_rules = []

        sma50 = last.get("SMA_50", np.nan)
        sma200 = last.get("SMA_200", np.nan)

        # BUY_01: Trend Filter
        if (not pd.isna(sma50) and not pd.isna(sma200)
                and close > sma200 and sma50 > sma200):
            passed_rules.append("BUY_01")
        else:
            failed_rules.append("BUY_01:Trend")

        # BUY_02: 50-Day MA Bullish Slope
        if len(df) >= 6 and not pd.isna(sma50):
            sma50_5ago = df["SMA_50"].iloc[-6]
            if not pd.isna(sma50_5ago) and sma50 > sma50_5ago:
                passed_rules.append("BUY_02")
            else:
                failed_rules.append("BUY_02:Slope")
        else:
            failed_rules.append("BUY_02:Slope")

        # BUY_03: BB Breakout OR 20-Day High Close
        bb_upper = last.get("BB_upper", np.nan)
        high_close_20 = last.get("High_Close_20", np.nan)
        bb_break = not pd.isna(bb_upper) and close > bb_upper
        high_20 = not pd.isna(high_close_20) and close >= high_close_20
        if bb_break or high_20:
            passed_rules.append("BUY_03")
        else:
            failed_rules.append("BUY_03:BB/High")

        # BUY_04: MACD Bullish
        macd_line = last.get("MACD_line", np.nan)
        macd_signal = last.get("MACD_signal", np.nan)
        macd_hist = last.get("MACD_histogram", np.nan)
        if (not pd.isna(macd_line) and not pd.isna(macd_signal)
                and not pd.isna(macd_hist)
                and macd_line > macd_signal and macd_hist > 0):
            passed_rules.append("BUY_04")
        else:
            failed_rules.append("BUY_04:MACD")

        # BUY_05: SMA(10)/SMA(30) Crossover within 5 sessions
        sma10 = last.get("SMA_10", np.nan)
        sma30 = last.get("SMA_30", np.nan)
        cross_ok = False
        if not pd.isna(sma10) and not pd.isna(sma30) and sma10 > sma30:
            for lb in [1, 2, 3, 4, 5]:
                if len(df) > lb:
                    p10 = df["SMA_10"].iloc[-(lb + 1)]
                    p30 = df["SMA_30"].iloc[-(lb + 1)]
                    if not pd.isna(p10) and not pd.isna(p30) and p10 <= p30:
                        cross_ok = True
                        break
        if cross_ok:
            passed_rules.append("BUY_05")
        else:
            failed_rules.append("BUY_05:Crossover")

        # BUY_06: VWAP Confirmation
        vwap = last.get("VWAP", np.nan)
        if pd.isna(vwap) and len(df) >= 2:
            prev = df.iloc[-2]
            vwap = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        if not pd.isna(vwap) and close > vwap:
            passed_rules.append("BUY_06")
        else:
            failed_rules.append("BUY_06:VWAP")

        # BUY_07: EMA Stack
        ema20 = last.get("EMA_20", np.nan)
        ema50 = last.get("EMA_50", np.nan)
        if not pd.isna(ema20) and not pd.isna(ema50) and ema20 > ema50:
            passed_rules.append("BUY_07")
        else:
            failed_rules.append("BUY_07:EMA")

        # BUY_08: RSI Sweet Spot
        rsi = last.get("RSI_14", np.nan)
        if not pd.isna(rsi) and 40 <= rsi <= 70:
            passed_rules.append("BUY_08")
        else:
            failed_rules.append("BUY_08:RSI")

        # BUY_09: Volume Surge
        vol = last.get("Volume", 0)
        vol_sma = last.get("Vol_SMA_20", np.nan)
        if not pd.isna(vol_sma) and vol_sma > 0 and vol >= 1.25 * vol_sma:
            passed_rules.append("BUY_09")
        else:
            failed_rules.append("BUY_09:Volume")

        # BUY_10: No Penny Stocks
        if close >= 5.0:
            passed_rules.append("BUY_10")
        else:
            failed_rules.append("BUY_10:Penny")

        return {
            "ticker": ticker,
            "price": close,
            "rules_passed": len(passed_rules),
            "rules_failed": len(failed_rules),
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "rsi_14": rsi if not pd.isna(rsi) else None,
            "macd_histogram": macd_hist if not pd.isna(macd_hist) else None,
            "volume_ratio": last.get("Volume_Ratio", np.nan),
            "return_20d": last.get("Return_20d", np.nan),
            "sma50": sma50,
            "sma200": sma200,
        }

    @staticmethod
    def _evaluate(
        ticker: str, df: pd.DataFrame
    ) -> Tuple[bool, str, List[str]]:
        """
        Evaluate all 10 hard buy rules on the most recent row.
        Returns (passed_all, failing_rule_name, flags_list).
        """
        flags = []
        last = df.iloc[-1]
        close = last["Close"]

        # ── BUY_01: Trend Filter ─────────────────────────────────────
        # Close > SMA(200) AND SMA(50) > SMA(200)
        sma50 = last.get("SMA_50", np.nan)
        sma200 = last.get("SMA_200", np.nan)
        if pd.isna(sma50) or pd.isna(sma200):
            return False, "BUY_01: SMA data unavailable", flags
        if not (close > sma200 and sma50 > sma200):
            return False, "BUY_01: Trend Filter failed", flags

        # ── BUY_02: 50-Day MA Bullish Slope ──────────────────────────
        # SMA(50) today > SMA(50) 5 sessions ago
        if len(df) < 6:
            return False, "BUY_02: Insufficient data", flags
        sma50_5ago = df["SMA_50"].iloc[-6]
        if pd.isna(sma50_5ago) or sma50 <= sma50_5ago:
            return False, "BUY_02: SMA(50) slope not positive", flags

        # ── BUY_03: Bollinger Breakout OR 20-Day High Close ──────────
        bb_upper = last.get("BB_upper", np.nan)
        high_close_20 = last.get("High_Close_20", np.nan)
        bb_break = not pd.isna(bb_upper) and close > bb_upper
        high_20_break = not pd.isna(high_close_20) and close >= high_close_20

        if not (bb_break or high_20_break):
            return False, "BUY_03: No BB breakout or 20d high", flags

        # Flag marginal BB break
        if bb_break and not pd.isna(bb_upper) and bb_upper > 0:
            margin = (close - bb_upper) / bb_upper
            if margin < 0.001:
                flags.append("MARGINAL_BB_BREAK")

        # ── BUY_04: MACD Bullish ─────────────────────────────────────
        macd_line = last.get("MACD_line", np.nan)
        macd_signal = last.get("MACD_signal", np.nan)
        macd_hist = last.get("MACD_histogram", np.nan)
        if pd.isna(macd_line) or pd.isna(macd_signal) or pd.isna(macd_hist):
            return False, "BUY_04: MACD data unavailable", flags
        if not (macd_line > macd_signal and macd_hist > 0):
            return False, "BUY_04: MACD not bullish", flags

        # Flag weak MACD
        if 0 < macd_hist < 0.01:
            flags.append("WEAK_MACD")

        # ── BUY_05: Short-Term MA Crossover (10/30 within 5 sessions) ─
        sma10_today = last.get("SMA_10", np.nan)
        sma30_today = last.get("SMA_30", np.nan)
        if pd.isna(sma10_today) or pd.isna(sma30_today):
            return False, "BUY_05: SMA(10)/SMA(30) data unavailable", flags

        if sma10_today <= sma30_today:
            return False, "BUY_05: SMA(10) not above SMA(30)", flags

        # Check if crossover happened within last 5 sessions
        crossover_recent = False
        for lookback in [1, 2, 3, 4, 5]:
            if len(df) > lookback:
                prev_sma10 = df["SMA_10"].iloc[-(lookback + 1)]
                prev_sma30 = df["SMA_30"].iloc[-(lookback + 1)]
                if not pd.isna(prev_sma10) and not pd.isna(prev_sma30):
                    if prev_sma10 <= prev_sma30:
                        crossover_recent = True
                        break

        if not crossover_recent:
            return False, "BUY_05: Crossover not within last 5 sessions", flags

        # Flag tight crossover
        if sma30_today > 0 and abs(sma10_today - sma30_today) / sma30_today < 0.005:
            flags.append("TIGHT_CROSSOVER")

        # ── BUY_06: VWAP Confirmation ────────────────────────────────
        vwap = last.get("VWAP", np.nan)
        if pd.isna(vwap):
            # Fallback: use previous session VWAP
            if len(df) >= 2:
                prev = df.iloc[-2]
                vwap = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        if pd.isna(vwap) or close <= vwap:
            return False, "BUY_06: Price below VWAP", flags

        # ── BUY_07: EMA Stack ────────────────────────────────────────
        ema20 = last.get("EMA_20", np.nan)
        ema50 = last.get("EMA_50", np.nan)
        if pd.isna(ema20) or pd.isna(ema50) or ema20 <= ema50:
            return False, "BUY_07: EMA(20) not above EMA(50)", flags

        # ── BUY_08: RSI Sweet Spot ───────────────────────────────────
        rsi = last.get("RSI_14", np.nan)
        if pd.isna(rsi) or not (40 <= rsi <= 70):
            return False, "BUY_08: RSI outside 40-70 range", flags

        # ── BUY_09: Volume Surge ─────────────────────────────────────
        vol = last["Volume"]
        vol_sma20 = last.get("Vol_SMA_20", np.nan)
        if pd.isna(vol_sma20) or vol_sma20 == 0:
            return False, "BUY_09: Volume SMA unavailable", flags
        if vol < 1.25 * vol_sma20:
            return False, "BUY_09: Volume below 1.25x average", flags

        # Flag volume spike with no price move
        ret_1d = last.get("Return_1d", np.nan)
        if not pd.isna(ret_1d) and abs(ret_1d) < 0.005:
            flags.append("VOLUME_SPIKE_NO_MOVE")

        # ── BUY_10: No Penny Stocks ──────────────────────────────────
        if close < 5.0:
            return False, "BUY_10: Penny stock", flags

        return True, "", flags


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 -- ML RANKING LAYER
# ═══════════════════════════════════════════════════════════════════════════════


class _ModelDegradedError(Exception):
    """
    Raised by a trainer when it cannot produce a real fit (e.g. single-class
    labels, missing library) and has chosen to fall back to neutral scores.

    Raising instead of returning allows _safe_scores to detect the skip and
    record the model in degraded_models, so the ensemble status is always
    reported correctly.
    """


class MLRanker:
    """
    XGBoost + Random Forest ensemble ranking.
    Trains on pooled historical data from all survivors.
    Optional LSTM sequence layer.
    """

    # Every feature here has to be comparable across a $9 name and a $900 one,
    # because the model is fitted on the whole cross-section and then asked to
    # rank names against each other. Returns, RSI and the vs-MA spreads are
    # already ratios; MACD and dollar volume are the two that were not, and are
    # taken price-normalised and log-scaled respectively.
    FEATURE_COLS = [
        "Return_1d", "Return_5d", "Return_20d", "RSI_14",
        "MACD_hist_pct", "Volume_Ratio", "Close_vs_SMA50",
        "Close_vs_SMA200", "EMA20_vs_EMA50", "Log_Dollar_Vol_20",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.rf_model = None
        self.lstm_model = None
        self.feature_importances = {}
        # Per-model importances, averaged once at the end. Folding them in
        # pairwise as they arrived made the blend depend on training order and
        # silently halved the first model's contribution.
        self._importances_by_model: Dict[str, Dict[str, float]] = {}
        # Models that fell back to neutral scores this run, so the report can
        # say the ensemble was degraded instead of presenting 0.5 as a signal.
        self.degraded_models: List[str] = []

    def _finalize_feature_importances(self) -> None:
        """Average the per-model importances into the reported blend."""
        if not self._importances_by_model:
            return
        blended: Dict[str, float] = {}
        for name in self.FEATURE_COLS:
            values = [
                model_imp[name]
                for model_imp in self._importances_by_model.values()
                if name in model_imp
            ]
            if values:
                blended[name] = float(np.mean(values))
        self.feature_importances = blended

    def rank(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
        training_universe: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Dict]:
        """
        Train ML models on historical data, then score each survivor.

        training_universe (optional): broader pool of tickers used to fit the
        models. Using the full guarded universe (~2500 names × 252 days)
        instead of just survivors (~50) provides 50x more samples and far
        better discriminative power -- a key request from the panel review.
        """
        log.info(f"STAGE 4: ML Ranking -- {len(survivors)} survivors")

        if len(survivors) < 1:
            return survivors

        # Build training dataset from broad universe; score the survivors.
        train_pool = training_universe if training_universe else all_data
        X_train, y_train, X_current, current_tickers = self._build_dataset(
            survivors, all_data, train_pool
        )

        if X_train is None or len(X_train) < 250:
            log.warning("Insufficient training samples -- assigning uniform scores")
            for s in survivors:
                s["ml_score_xgb"] = 0.5
                s["ml_score_rf"] = 0.5
                s["ml_ensemble_score"] = 0.5
                s["lstm_score"] = None
            return survivors

        X_train, y_train = self._cap_training_rows(X_train, y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_current_scaled = self.scaler.transform(X_current)

        # Train XGBoost and Random Forest. A model that fails degrades to
        # neutral scores instead of ending the run at Stage 4.
        xgb_scores = self._safe_scores(
            "XGBoost", self._train_xgboost, X_train_scaled, y_train, X_current_scaled
        )
        rf_scores = self._safe_scores(
            "RandomForest", self._train_rf, X_train_scaled, y_train, X_current_scaled
        )

        # Ensemble
        ensemble_scores = (xgb_scores + rf_scores) / 2

        # Check score spread
        spread = ensemble_scores.max() - ensemble_scores.min()
        if spread < 0.02:
            log.warning(
                f"ML score spread is only {spread:.4f} -- "
                f"model may lack discriminative power"
            )

        # Optional LSTM -- keyed by ticker, not by the XGB/RF row index.
        #
        # Guarded separately from _safe_scores because it returns a
        # {ticker: score} map rather than a score array. This is the one Stage
        # 4 model that is purely informational: lstm_score is reported but
        # never ranked on. Letting an optional extra abort a ~100 minute scan
        # minutes before the output is written is the worst trade available.
        try:
            lstm_scores = self._train_lstm(survivors, all_data)
        except Exception as exc:
            log.error(
                f"  LSTM layer failed ({exc}) -- continuing with XGBoost + "
                "Random Forest."
            )
            log.debug(traceback.format_exc())
            if "LSTM" not in self.degraded_models:
                self.degraded_models.append("LSTM")
            lstm_scores = None

        # Assign scores back to survivors
        ticker_to_idx = {t: i for i, t in enumerate(current_tickers)}
        for s in survivors:
            idx = ticker_to_idx.get(s["ticker"])
            if idx is not None:
                s["ml_score_xgb"] = float(xgb_scores[idx])
                s["ml_score_rf"] = float(rf_scores[idx])
                s["ml_ensemble_score"] = float(ensemble_scores[idx])
            else:
                s["ml_score_xgb"] = 0.5
                s["ml_score_rf"] = 0.5
                s["ml_ensemble_score"] = 0.5
            s["lstm_score"] = (
                lstm_scores.get(s["ticker"]) if lstm_scores else None
            )
            if self.degraded_models:
                # Flag rather than hide it: a 0.5 from a failed model is the
                # absence of a signal, not a neutral verdict on the name.
                flags = s.setdefault("flags", [])
                flag = "ML_DEGRADED:" + "+".join(self.degraded_models)
                if flag not in flags:
                    flags.append(flag)

        # Log feature importances
        self._finalize_feature_importances()
        log.info("  ML Feature Importances (top 5):")
        for name, imp in sorted(
            self.feature_importances.items(), key=lambda x: -x[1]
        )[:5]:
            log.info(f"    {name}: {imp:.4f}")

        log.info("STAGE 4 COMPLETE: ML scores assigned")
        return survivors

    @staticmethod
    def _cap_training_rows(
        X_train: np.ndarray, y_train: np.ndarray, max_rows: int = 80_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep the most recent rows, in date order, so live scans finish.

        Rows arrive sorted by date and the walk-forward CV in _train_xgboost /
        _train_rf depends on that order. Taking the most recent N rows of each
        class *separately* breaks it: the majority class is drawn from a much
        shorter recent window than the minority class, so once the kept rows
        are re-sorted the oldest block contains minority-class rows only. On
        the live universe (511,649 samples, 30.6% positive) that left the
        first ~22,300 rows all class 1, TimeSeriesSplit handed XGBoost a
        single-class first fold, and the scan died at Stage 4 with
        "Invalid classes inferred from unique values of `y`".

        It also trained the models on a class prior that drifted with the
        calendar -- 100% positive in the oldest rows -- which is a data defect
        no fold-skipping would repair.

        Truncating chronologically keeps every fold contiguous in time and the
        class prior stable. Imbalance is handled where it belongs: by the
        estimators' own class weighting.
        """
        if len(X_train) <= max_rows:
            return X_train, y_train
        X_train = X_train[-max_rows:]
        y_train = y_train[-max_rows:]
        log.info(
            f"  Training set capped to the {len(y_train)} most recent samples "
            f"(positive class {float(y_train.mean()):.1%}); imbalance handled "
            "by model class weighting."
        )
        return X_train, y_train

    @staticmethod
    def _class_counts(y: np.ndarray) -> Tuple[int, int]:
        """(negatives, positives) in a binary label vector."""
        positives = int(np.count_nonzero(y == 1))
        return int(np.asarray(y).size - positives), positives

    @classmethod
    def _scale_pos_weight(cls, y: np.ndarray) -> float:
        """
        XGBoost's imbalance lever: negatives / positives.

        Takes over from the balanced subsample the trainer used to rely on, so
        the model still weighs the classes equally while the rows stay in
        unbroken date order. Keeping the effective balance also keeps the
        predicted probabilities on the same scale downstream scoring already
        assumes (0.5 = neutral).
        """
        negatives, positives = cls._class_counts(y)
        if not positives or not negatives:
            return 1.0
        return float(negatives) / float(positives)

    @staticmethod
    def _causal_sequence_norm(feat_df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Z-score each row against its own trailing window, never the full series.

        Normalising the whole series at once computes the mean and std from
        every bar, including ones after the row being scored: a day-50 sequence
        was scaled using information from day 1200. That is look-ahead -- the
        scale itself encodes where the bar sits relative to the future the
        model is being asked to predict, and it inflates the reported score.

        A trailing window keeps the transform causal. Dropping the level along
        with it also makes sequences comparable across names, which raw prices
        in the feature list otherwise are not.
        """
        roll = feat_df.rolling(window=window, min_periods=window)
        normed = (feat_df - roll.mean()) / roll.std().replace(0, np.nan)
        return normed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _safe_scores(
        self,
        label: str,
        trainer,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_current: np.ndarray,
    ) -> np.ndarray:
        """
        Run one model, degrading to neutral scores instead of aborting the scan.

        Stage 4 lands a few minutes into a ~100 minute run and every later
        stage depends on it, so a single estimator blowing up must not throw
        away the whole scan. The degradation is recorded and reported rather
        than passed off as a real score.

        Trainers signal a deliberate graceful skip by raising
        _ModelDegradedError; unexpected exceptions are caught with the same
        result so that no single model can abort the whole scan.
        """
        try:
            return trainer(X_train, y_train, X_current)
        except _ModelDegradedError as exc:
            log.warning(
                f"  {label} skipped ({exc}) -- scoring every survivor "
                "0.5 for this model and continuing."
            )
            if label not in self.degraded_models:
                self.degraded_models.append(label)
            return np.full(X_current.shape[0], 0.5)
        except Exception as exc:
            log.error(
                f"  {label} training failed ({exc}) -- scoring every survivor "
                "0.5 for this model and continuing."
            )
            log.debug(traceback.format_exc())
            if label not in self.degraded_models:
                self.degraded_models.append(label)
            return np.full(X_current.shape[0], 0.5)

    def _build_dataset(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
        training_pool: Dict[str, pd.DataFrame],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, List[str]]:
        """
        Build training and current-day feature matrices.

        training_pool: broader universe used to fit the model. The forward
        HOLDING_HORIZON_DAYS return label is generated identically across the
        pool. This prevents survivor-only training (heavy positive class bias)
        that previously made the classifier near-uniform.

        Current: most recent day's features for scoring (survivors only).
        """
        train_rows = []
        train_labels = []
        train_dates = []
        current_rows = []
        current_tickers = []

        # ---- TRAINING DATA: full guarded universe -----------------------
        # Cap per-ticker rows so a few long-history names don't dominate.
        MAX_TRAIN_ROWS_PER_TICKER = 252  # ~1 calendar year per name
        for ticker, df in training_pool.items():
            if df is None or len(df) < 60:
                continue

            try:
                feat_df = df[self.FEATURE_COLS].copy()
            except KeyError:
                continue
            close = df["Close"]
            avg_dv = df.get("Avg_Dollar_Vol_20")
            if avg_dv is None:
                continue

            # Forward-return label over the strategy's actual holding horizon.
            # The target is a *meaningful* move, not merely "up": training on
            # `fwd_ret > 0` rewards drifters, which is the opposite of a
            # max-profit objective. Requiring ML_TARGET_RETURN teaches the
            # model to separate real movers from noise.
            horizon = HOLDING_HORIZON_DAYS
            fwd_ret = close.shift(-horizon) / close - 1
            label = (fwd_ret > ML_TARGET_RETURN).astype(int)

            train_section = feat_df.iloc[:-horizon]
            label_section = label.iloc[:-horizon]
            eligible_section = (
                (close >= 5.0) &
                (avg_dv >= ExecutionGuards.MIN_DOLLAR_VOLUME)
            ).iloc[:-horizon].fillna(False)

            valid = train_section.loc[eligible_section].dropna()
            valid_labels = label_section.loc[valid.index].dropna()
            common_idx = valid.index.intersection(valid_labels.index)

            if len(common_idx) > MAX_TRAIN_ROWS_PER_TICKER:
                # Take the most recent slice (most relevant regime)
                common_idx = common_idx[-MAX_TRAIN_ROWS_PER_TICKER:]

            if len(common_idx) > 0:
                train_rows.append(valid.loc[common_idx].values)
                train_labels.append(valid_labels.loc[common_idx].values)
                train_dates.append(np.array(common_idx, dtype="datetime64[ns]"))

        # ---- CURRENT-DAY FEATURES: only the survivors we will score -----
        for s in survivors:
            ticker = s["ticker"]
            df = all_data.get(ticker)
            if df is None:
                continue
            try:
                feat_df = df[self.FEATURE_COLS].copy()
            except KeyError:
                continue
            current_feat = feat_df.iloc[-1].values
            if not np.any(np.isnan(current_feat)):
                current_rows.append(current_feat)
                current_tickers.append(ticker)

        if not train_rows or not current_rows:
            return None, None, np.array([]), []

        X_train = np.vstack(train_rows)
        y_train = np.concatenate(train_labels)
        train_dates_arr = np.concatenate(train_dates)
        X_current = np.array(current_rows)

        # Remove any remaining NaN/inf
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]
        train_dates_arr = train_dates_arr[mask]
        sort_idx = np.argsort(train_dates_arr)
        X_train = X_train[sort_idx]
        y_train = y_train[sort_idx]
        train_dates_arr = train_dates_arr[sort_idx]

        log.info(
            f"  Training set: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features. "
            f"Current set: {X_current.shape[0]} tickers."
        )
        if y_train.size:
            log.info(
                f"  Label: forward {HOLDING_HORIZON_DAYS}-session return > "
                f"{ML_TARGET_RETURN:.0%} -- positive class "
                f"{float(y_train.mean()):.1%} of samples."
            )

        return X_train, y_train, X_current, current_tickers

    def _train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray, X_current: np.ndarray
    ) -> np.ndarray:
        """Train XGBoost and return predicted probabilities for current data."""
        if not XGB_AVAILABLE:
            raise _ModelDegradedError("XGBoost not installed")

        negatives, positives = self._class_counts(y_train)
        if not negatives or not positives:
            # XGBoost treats a single-class target as a fatal ValueError.
            raise _ModelDegradedError(
                "training labels are single-class "
                f"(all {'positive' if positives else 'negative'})"
            )

        scale_pos_weight = self._scale_pos_weight(y_train)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
        )

        # Time-series cross-validation on date-sorted rows approximates
        # walk-forward validation across the whole market, not ticker blocks.
        gap = 20
        tscv = TimeSeriesSplit(n_splits=5)
        val_accs = []
        train_accs = []
        skipped_folds = 0
        for train_idx, val_idx in tscv.split(X_train):
            if gap > 0:
                val_start = val_idx[0]
                train_idx = train_idx[train_idx < (val_start - gap)]
                if train_idx.size == 0:
                    continue
            if np.unique(y_train[train_idx]).size < 2:
                # Still reachable on a narrow or quiet universe, where an early
                # window can hold one class only. Skip rather than let XGBoost
                # raise and take the whole scan down with it.
                skipped_folds += 1
                continue
            model.fit(X_train[train_idx], y_train[train_idx])
            train_pred = model.predict(X_train[train_idx])
            val_pred = model.predict(X_train[val_idx])
            train_accs.append(accuracy_score(y_train[train_idx], train_pred))
            val_accs.append(accuracy_score(y_train[val_idx], val_pred))

        if skipped_folds:
            log.warning(
                f"  XGBoost CV skipped {skipped_folds} single-class fold(s)."
            )

        if val_accs:
            avg_train = float(np.mean(train_accs))
            avg_val = float(np.mean(val_accs))
            log.info(
                f"  XGBoost CV -- Train acc: {avg_train:.3f}, Val acc: {avg_val:.3f}"
            )
        else:
            avg_train = avg_val = float("nan")
            log.warning(
                "  XGBoost CV produced no usable folds -- fitting on the full "
                "training set without validation accuracy."
            )

        if avg_train > 0.90 and avg_val < 0.60:
            log.warning(
                "  XGBoost overfitting detected. Increasing regularization."
            )
            model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=2.0,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbosity=0,
            )

        # Final fit on all training data
        model.fit(X_train, y_train)
        self.xgb_model = model

        # Feature importances
        self._importances_by_model["xgboost"] = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_COLS, model.feature_importances_)
        }

        probs = model.predict_proba(X_current)[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def _train_rf(
        self, X_train: np.ndarray, y_train: np.ndarray, X_current: np.ndarray
    ) -> np.ndarray:
        """Train Random Forest and return predicted probabilities."""
        negatives, positives = self._class_counts(y_train)
        if not negatives or not positives:
            # A single-class fit leaves predict_proba with one column, so the
            # [:, 1] lookup below would raise IndexError.
            raise _ModelDegradedError(
                "training labels are single-class "
                f"(all {'positive' if positives else 'negative'})"
            )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            # Matches XGBoost's scale_pos_weight now that the training rows are
            # capped chronologically instead of class-balanced by subsampling.
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Match XGBoost's walk-forward validation with a 20-session label gap.
        gap = 20
        try:
            tscv = TimeSeriesSplit(n_splits=5, gap=gap)
            use_manual_gap = False
        except TypeError:
            # Older scikit-learn versions do not support the `gap` argument.
            tscv = TimeSeriesSplit(n_splits=5)
            use_manual_gap = True

        val_accs = []
        skipped_folds = 0
        for train_idx, val_idx in tscv.split(X_train):
            if use_manual_gap:
                if len(train_idx) <= gap:
                    continue
                train_idx = train_idx[:-gap]
            if np.unique(y_train[train_idx]).size < 2:
                # Single-class fold: the fit would produce a one-column
                # predict_proba and a meaningless accuracy. Skip it.
                skipped_folds += 1
                continue
            model.fit(X_train[train_idx], y_train[train_idx])
            val_pred = model.predict(X_train[val_idx])
            val_accs.append(accuracy_score(y_train[val_idx], val_pred))

        if skipped_folds:
            log.warning(
                f"  Random Forest CV skipped {skipped_folds} single-class fold(s)."
            )
        if val_accs:
            log.info(f"  Random Forest CV -- Val acc: {np.mean(val_accs):.3f}")
        else:
            log.warning(
                "  Random Forest CV produced no usable folds -- fitting on the "
                "full training set without validation accuracy."
            )

        # Final fit
        model.fit(X_train, y_train)
        self.rf_model = model

        # Merge feature importances
        self._importances_by_model["random_forest"] = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_COLS, model.feature_importances_)
        }

        probs = model.predict_proba(X_current)[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def _train_lstm(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
    ) -> Optional[Dict[str, float]]:
        """
        Optional LSTM sequence scoring layer (PyTorch implementation).
        Architecture: LSTM(64) -> Dropout(0.3) -> Dense(32,ReLU) -> Dropout(0.2) -> Dense(1,Sigmoid)
        Returns a {ticker: score} mapping or None if unavailable.

        The mapping is keyed by ticker rather than positional index: the LSTM
        admits a different subset of tickers than the XGB/RF matrix (it needs a
        full clean 20-day sequence), so indexing LSTM output by the XGB index
        attributed one ticker's sequence score to a different ticker.
        """
        if not LSTM_AVAILABLE:
            log.info("  LSTM dependencies unavailable -- continuing with XGBoost + RF only.")
            return None

        log.info("  Training optional LSTM sequence layer (PyTorch)...")

        SEQ_LEN = 20
        SEQ_FEATURES = [
            "Close", "Volume", "RSI_14", "MACD_histogram",
            "EMA_20", "EMA_50", "SMA_50", "SMA_200",
            "Avg_Dollar_Vol_20", "Vol_SMA_20",
        ]

        train_X, train_y = [], []
        current_X = []
        current_tickers = []

        for s in survivors:
            ticker = s["ticker"]
            df = all_data.get(ticker)
            if df is None or len(df) < MIN_TRADING_DAYS:
                continue

            feat_df = df[SEQ_FEATURES].copy()
            close = df["Close"]
            values = MLRanker._causal_sequence_norm(feat_df, SEQ_LEN).values

            # Forward return labels on the same horizon/threshold as the
            # tree ensemble so all three models optimise the same objective.
            fwd_ret = close.shift(-HOLDING_HORIZON_DAYS) / close - 1
            labels = (fwd_ret > ML_TARGET_RETURN).astype(int)

            # Build sequences for training
            for i in range(SEQ_LEN, len(values) - HOLDING_HORIZON_DAYS):
                seq = values[i - SEQ_LEN : i]
                if not np.any(np.isnan(seq)) and not pd.isna(labels.iloc[i]):
                    train_X.append(seq)
                    train_y.append(labels.iloc[i])

            # Current sequence (last SEQ_LEN days)
            current_seq = values[-SEQ_LEN:]
            if not np.any(np.isnan(current_seq)):
                current_X.append(current_seq)
                current_tickers.append(ticker)

        if len(train_X) < 200 or not current_X:
            log.info("  Insufficient LSTM training data -- skipping")
            return None

        train_X = np.array(train_X, dtype=np.float32)
        train_y = np.array(train_y, dtype=np.float32)
        current_X = np.array(current_X, dtype=np.float32)

        try:
            # ------- PyTorch LSTM model (mirrors original Keras architecture) -------
            class _LSTMScorer(nn.Module):
                def __init__(self, input_dim: int, hidden_dim: int = 64):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim, hidden_size=hidden_dim,
                        batch_first=True,
                    )
                    self.dropout1 = nn.Dropout(0.3)
                    self.fc1 = nn.Linear(hidden_dim, 32)
                    self.relu = nn.ReLU()
                    self.dropout2 = nn.Dropout(0.2)
                    self.fc2 = nn.Linear(32, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    lstm_out, _ = self.lstm(x)          # (batch, seq, hidden)
                    last_hidden = lstm_out[:, -1, :]    # (batch, hidden)
                    x = self.dropout1(last_hidden)
                    x = self.relu(self.fc1(x))
                    x = self.dropout2(x)
                    return self.sigmoid(self.fc2(x)).squeeze(-1)

            device = torch.device("cpu")
            model = _LSTMScorer(input_dim=len(SEQ_FEATURES)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            # DataLoader
            t_X = torch.from_numpy(train_X).to(device)
            t_y = torch.from_numpy(train_y).to(device)
            dataset = TensorDataset(t_X, t_y)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            # ------- Training with early stopping (patience=5) -------
            best_loss = float("inf")
            best_state = None
            patience_counter = 0
            PATIENCE = 5
            EPOCHS = 50

            model.train()
            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                n_batches = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break

            # Restore best weights
            if best_state is not None:
                model.load_state_dict(best_state)

            # Check convergence
            if best_loss > 0.69:  # Worse than random
                log.warning(f"  LSTM did not converge (loss={best_loss:.4f}) -- discarding")
                return None

            # ------- Inference on current sequences -------
            model.eval()
            with torch.no_grad():
                c_X = torch.from_numpy(current_X).to(device)
                scores = model(c_X).cpu().numpy()

            log.info(f"  LSTM scores computed for {len(scores)} tickers (PyTorch, loss={best_loss:.4f})")
            self.lstm_model = model
            scores = np.clip(scores, 0.0, 1.0)
            return {t: float(v) for t, v in zip(current_tickers, scores)}

        except Exception as e:
            log.warning(f"  LSTM training failed: {e}")
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS FETCHER -- for panel scoring (survivors only)
# ═══════════════════════════════════════════════════════════════════════════════

class FundamentalsFetcher:
    """
    Fetch fundamental data via MBOUM Pro modules.
    Uses: financial-data, default-key-statistics, asset-profile,
    calendar-events, net-share-purchase-activity, insider-transactions.
    """

    # Modules whose absence degrades a score but must not fail the run.
    CORE_MODULES = (
        "financial-data",
        "default-key-statistics",
        "asset-profile",
        "calendar-events",
    )
    # Insider conviction modules -- "follow the insiders" needs live filings.
    INSIDER_MODULES = (
        "net-share-purchase-activity",
        "insider-transactions",
    )

    def __init__(self, massive_key: str):
        self.massive_key = massive_key or MASSIVE_API_KEY
        self.mboum = MboumAPI()
        self.massive_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.massive_session.mount("https://", adapter)
        self.massive_session.mount("http://", adapter)
        self.massive_session.headers.update({"X-massive-Token": self.massive_key})

    def fetch_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch fundamentals for a list of tickers. Returns dict of info dicts."""
        log.info(
            f"Fetching fundamentals for {len(tickers)} survivors "
            "(MBOUM primary; Massive/TwelveData/Finnhub/Yahoo fallbacks)"
        )
        results = {}

        def _fetch_one(ticker: str) -> Tuple[str, Dict]:
            info = {"name": ticker, "sector": "Unknown"}
            try:
                modules = {}
                try:
                    if MBOUM_API_KEY and ProviderCircuit.get("MBOUM-fundamentals").available():
                        modules = self.mboum.get_modules(
                            ticker,
                            list(self.CORE_MODULES) + list(self.INSIDER_MODULES),
                        )
                except ProviderExhausted:
                    modules = {}

                fin = modules.get("financial-data", {})
                stats = modules.get("default-key-statistics", {})
                profile = modules.get("asset-profile", {})
                cal = modules.get("calendar-events", {})
                missing_modules = [
                    m for m in self.CORE_MODULES
                    if not isinstance(modules.get(m), dict) or not modules.get(m)
                ]

                def _raw(d, key):
                    return normalize_api_scalar(d.get(key))

                numeric_fields = (
                    "market_cap", "pe_ratio", "forward_pe", "peg_ratio", "beta",
                    "profit_margin", "revenue_growth", "earnings_growth",
                    "debt_to_equity", "free_cash_flow", "return_on_equity",
                    "52w_high", "52w_low", "avg_volume", "shares_float",
                    "shares_outstanding", "inst_ownership_pct", "analyst_count",
                    "target_price", "short_pct_float", "short_ratio",
                    "insider_net_purchase_pct", "insider_buy_transactions",
                    "insider_sell_transactions", "insider_net_shares",
                )

                info = {
                    "name": normalize_api_scalar(
                        profile.get("longName", profile.get("shortName", ticker))
                    ),
                    "sector": normalize_api_scalar(profile.get("sector", "Unknown")),
                    "industry": normalize_api_scalar(profile.get("industry", "Unknown")),
                    "quote_type": normalize_api_scalar(profile.get("quoteType")),
                    "market_cap": _raw(stats, "marketCap"),
                    "pe_ratio": _raw(stats, "trailingPE"),
                    "forward_pe": _raw(stats, "forwardPE"),
                    "peg_ratio": _raw(stats, "pegRatio"),
                    "beta": _raw(stats, "beta"),
                    "profit_margin": _raw(stats, "profitMargins"),
                    "revenue_growth": _raw(fin, "revenueGrowth"),
                    "earnings_growth": _raw(fin, "earningsGrowth"),
                    "debt_to_equity": _raw(fin, "debtToEquity"),
                    "free_cash_flow": _raw(fin, "freeCashflow"),
                    "return_on_equity": _raw(fin, "returnOnEquity"),
                    "52w_high": _raw(stats, "fiftyTwoWeekHigh"),
                    "52w_low": _raw(stats, "fiftyTwoWeekLow"),
                    "avg_volume": _raw(stats, "averageVolume"),
                    "shares_float": _raw(stats, "floatShares"),
                    "shares_outstanding": _raw(stats, "sharesOutstanding"),
                    "inst_ownership_pct": _raw(stats, "heldPercentInstitutions"),
                    "analyst_count": _raw(fin, "numberOfAnalystOpinions"),
                    "target_price": _raw(fin, "targetMeanPrice"),
                    # Squeeze fuel -- elevated short interest accelerates the
                    # short-to-mid-term moves this strategy targets.
                    "short_pct_float": _raw(stats, "shortPercentOfFloat"),
                    "short_ratio": _raw(stats, "shortRatio"),
                    "earnings_date": None,
                    "fundamentals_quality": "complete" if not missing_modules else "partial",
                    "missing_fundamental_modules": missing_modules,
                }

                # Live insider activity. Officers and directors sell for many
                # reasons but buy for exactly one, so net insider purchasing is
                # the highest-signal fundamental input on a short horizon.
                info.update(
                    FundamentalsFetcher._extract_insider_activity(modules)
                )

                # Extract earnings date from calendar
                earnings = cal.get("earnings", {})
                ed_list = earnings.get("earningsDate", [])
                if ed_list:
                    ed = ed_list[0] if isinstance(ed_list, list) else ed_list
                    if isinstance(ed, dict):
                        info["earnings_date"] = ed.get("fmt")
                    elif isinstance(ed, str):
                        info["earnings_date"] = ed

                # Supplement with Massive metrics (PE, beta, 52w range).
                # NOTE: previous code passed a literal "{symbol}" template
                # in the URL which never substituted -- fixed to use the
                # ticker-formatted endpoint with the canonical query string.
                try:
                    snap_url = f"https://api.massive.com/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
                    r = self.massive_session.get(
                        snap_url,
                        params={"apiKey": MASSIVE_API_KEY},
                        timeout=15,
                    )
                    if r.status_code == 200:
                        # Massive single-ticker snapshot returns a {"ticker":{...}} payload
                        snap_payload = r.json() or {}
                        ticker_data = (
                            snap_payload.get("ticker")
                            or snap_payload.get("results")
                            or {}
                        )
                        if isinstance(ticker_data, list) and ticker_data:
                            ticker_data = ticker_data[0]
                        last_trade = ticker_data.get("lastTrade") or ticker_data.get("last_trade") or {}
                        last_quote = ticker_data.get("lastQuote") or ticker_data.get("last_quote") or {}
                        info["live_last"] = normalize_api_scalar(last_trade.get("p"))
                        info["live_bid"] = normalize_api_scalar(last_quote.get("p"))
                        info["live_ask"] = normalize_api_scalar(last_quote.get("P"))
                        info["live_ts"] = normalize_api_scalar(last_trade.get("t"))
                except Exception:
                    pass

                for field in numeric_fields:
                    info[field] = normalize_api_scalar(info.get(field, np.nan))

                if not isinstance(info.get("name"), str) or is_missing_value(info["name"]):
                    info["name"] = ticker
                if not isinstance(info.get("sector"), str) or is_missing_value(info["sector"]):
                    info["sector"] = "Unknown"
                if not isinstance(info.get("industry"), str) or is_missing_value(info["industry"]):
                    info["industry"] = "Unknown"

                info["earnings_date"] = normalize_api_scalar(info.get("earnings_date"))
                if is_missing_value(info["earnings_date"]):
                    info["earnings_date"] = None
                else:
                    info["earnings_date"] = str(info["earnings_date"])

                info = self._enrich_fundamentals(ticker, info)

            except Exception as e:
                info = {
                    "name": ticker,
                    "sector": "Unknown",
                    "fundamentals_quality": "failed",
                    "missing_fundamental_modules": [
                        "financial-data", "default-key-statistics",
                        "asset-profile", "calendar-events"
                    ],
                    "error": str(e),
                }
                try:
                    info = self._enrich_fundamentals(ticker, info)
                except Exception:
                    pass

            return ticker, info

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    ticker, info = future.result()
                except Exception as e:
                    log.debug(f"  Fundamentals failed for {ticker}: {e}")
                    info = {
                        "name": ticker,
                        "sector": "Unknown",
                        "fundamentals_quality": "failed",
                        "error": str(e),
                    }
                results[ticker] = info

        log.info(f"Fundamentals fetched for {len(results)} tickers")
        with_insider = sum(
            1 for info in results.values()
            if not is_missing_value(info.get("insider_net_purchase_pct"))
            or not is_missing_value(info.get("insider_buy_transactions"))
        )
        sources = ", ".join(
            f"{name}={count}"
            for name, count in DATA_SOURCE_USAGE["fundamentals"].most_common()
        ) or "none"
        log.info(f"  Live insider activity available for {with_insider} tickers")
        log.info(f"  Fundamentals sources: {sources}")
        return results

    def _needs_fundamental_enrichment(self, info: Dict) -> bool:
        if info.get("fundamentals_quality") in {"failed", "partial"}:
            return True
        if not isinstance(info.get("sector"), str) or info.get("sector") in {"", "Unknown"}:
            return True
        if is_missing_value(info.get("market_cap")):
            return True
        return False

    def _fill_missing_fundamentals(self, dst: Dict, src: Dict) -> Dict:
        for key, value in src.items():
            if key in {
                "fundamentals_quality", "missing_fundamental_modules",
                "fundamentals_sources", "error",
            }:
                continue
            current = dst.get(key)
            if key in {"name", "sector", "industry"}:
                if not isinstance(current, str) or current in {"", "Unknown"} or is_missing_value(current):
                    if isinstance(value, str) and value and value != "Unknown":
                        dst[key] = value
                continue
            if is_missing_value(current) and not is_missing_value(value):
                dst[key] = value
        return dst

    def _enrich_fundamentals(self, ticker: str, info: Dict) -> Dict:
        """Fill missing MBOUM fields from Massive / TwelveData / Finnhub / yfinance."""
        sources = list(info.get("fundamentals_sources") or [])
        mboum_complete = (
            info.get("fundamentals_quality") == "complete"
            and not self._needs_fundamental_enrichment(info)
        )
        if mboum_complete:
            DATA_SOURCE_USAGE["fundamentals"]["MBOUM"] += 1
            info["fundamentals_sources"] = ["MBOUM"]
            return info
        if MBOUM_API_KEY and info.get("fundamentals_quality") in {"complete", "partial"}:
            sources.append("MBOUM")
            DATA_SOURCE_USAGE["fundamentals"]["MBOUM"] += 1

        enrichers = (
            ("Massive", self._fundamentals_massive),
            ("TwelveData", self._fundamentals_twelvedata),
            ("Finnhub", self._fundamentals_finnhub),
            ("yfinance", self._fundamentals_yfinance),
        )
        for label, fn in enrichers:
            if not self._needs_fundamental_enrichment(info) and not is_missing_value(
                info.get("insider_buy_transactions")
            ):
                break
            try:
                extra = fn(ticker)
            except ProviderExhausted:
                continue
            except Exception as exc:
                log.debug(f"  {label} fundamentals failed for {ticker}: {exc}")
                continue
            if not extra:
                continue
            filled = False
            before_keys = {k: dst for k, dst in info.items() if k not in {"missing_fundamental_modules"}}
            info = self._fill_missing_fundamentals(info, extra)
            for key, value in extra.items():
                if key in info and not is_missing_value(info.get(key)) and (
                    key not in before_keys or is_missing_value(before_keys.get(key))
                ):
                    filled = True
                    break
            if filled or extra.get("insider_buy_transactions") is not None:
                sources.append(label)
                DATA_SOURCE_USAGE["fundamentals"][label] += 1

        if sources:
            info["fundamentals_sources"] = sources
        if not self._needs_fundamental_enrichment(info):
            info["fundamentals_quality"] = "complete"
            info["missing_fundamental_modules"] = []
        elif info.get("fundamentals_quality") == "failed" and sources:
            info["fundamentals_quality"] = "partial"
        return info

    def _fundamentals_massive(self, ticker: str) -> Dict:
        if not MASSIVE_API_KEY:
            return {}
        url = f"https://api.massive.com/v3/reference/tickers/{ticker}"
        r = self.massive_session.get(url, params={"apiKey": MASSIVE_API_KEY}, timeout=15)
        kind = classify_http_error(r.status_code, r.text or "")
        if kind in ("credit", "auth"):
            ProviderCircuit.get("Massive-fundamentals").trip(f"HTTP {r.status_code}")
            raise ProviderExhausted("Massive", f"HTTP {r.status_code}")
        if r.status_code != 200:
            return {}
        results = (r.json() or {}).get("results") or {}
        if not isinstance(results, dict):
            return {}
        return {
            "name": results.get("name") or ticker,
            "industry": results.get("sic_description") or "Unknown",
            "market_cap": normalize_api_scalar(results.get("market_cap")),
            "shares_outstanding": normalize_api_scalar(
                results.get("share_class_shares_outstanding")
                or results.get("weighted_shares_outstanding")
            ),
        }

    def _fundamentals_twelvedata(self, ticker: str) -> Dict:
        circuit = ProviderCircuit.get("TwelveData-fundamentals", fail_limit=4)
        if not TWELVEDATA_API_KEY or not circuit.available():
            return {}
        session = get_market_router().session
        out: Dict[str, Any] = {}
        try:
            prof = session.get(
                f"{TWELVEDATA_BASE_URL}/profile",
                params={"symbol": ticker, "apikey": TWELVEDATA_API_KEY},
                timeout=15,
            )
            payload = prof.json() if prof.status_code == 200 else {}
            if payload.get("status") == "error":
                kind = classify_http_error(
                    payload.get("code") or prof.status_code,
                    str(payload.get("message") or ""),
                )
                if kind in ("credit", "auth"):
                    circuit.trip(str(payload.get("message") or ""))
                    raise ProviderExhausted("TwelveData", str(payload.get("message") or ""))
            else:
                out["name"] = payload.get("name") or ticker
                out["sector"] = payload.get("sector") or "Unknown"
                out["industry"] = payload.get("industry") or "Unknown"

            stats_resp = session.get(
                f"{TWELVEDATA_BASE_URL}/statistics",
                params={"symbol": ticker, "apikey": TWELVEDATA_API_KEY},
                timeout=15,
            )
            stats_payload = stats_resp.json() if stats_resp.status_code == 200 else {}
            if stats_payload.get("status") == "error":
                kind = classify_http_error(
                    stats_payload.get("code") or stats_resp.status_code,
                    str(stats_payload.get("message") or ""),
                )
                if kind in ("credit", "auth"):
                    circuit.trip(str(stats_payload.get("message") or ""))
                    raise ProviderExhausted("TwelveData", str(stats_payload.get("message") or ""))
            statistics = stats_payload.get("statistics") or {}
            val = statistics.get("valuations_metrics") or {}
            fin = statistics.get("financials") or {}
            stock = statistics.get("stock_statistics") or {}
            px = statistics.get("stock_price_summary") or {}
            inc = fin.get("income_statement") if isinstance(fin.get("income_statement"), dict) else {}
            cf = fin.get("cash_flow") if isinstance(fin.get("cash_flow"), dict) else {}
            bs = fin.get("balance_sheet") if isinstance(fin.get("balance_sheet"), dict) else {}
            out.update({
                "market_cap": normalize_api_scalar(val.get("market_capitalization")),
                "pe_ratio": normalize_api_scalar(val.get("trailing_pe")),
                "forward_pe": normalize_api_scalar(val.get("forward_pe")),
                "peg_ratio": normalize_api_scalar(val.get("peg_ratio")),
                "profit_margin": normalize_api_scalar(fin.get("profit_margin")),
                "return_on_equity": normalize_api_scalar(fin.get("return_on_equity_ttm")),
                "revenue_growth": normalize_api_scalar(
                    inc.get("quarterly_revenue_growth") or inc.get("revenue_growth")
                ),
                "earnings_growth": normalize_api_scalar(
                    inc.get("quarterly_earnings_growth") or inc.get("earnings_growth")
                ),
                "free_cash_flow": normalize_api_scalar(
                    cf.get("free_cash_flow") or cf.get("levered_free_cash_flow")
                ),
                "debt_to_equity": normalize_api_scalar(bs.get("debt_to_equity")),
                "shares_outstanding": normalize_api_scalar(stock.get("shares_outstanding")),
                "shares_float": normalize_api_scalar(stock.get("float_shares")),
                "avg_volume": normalize_api_scalar(
                    stock.get("avg_90_volume") or stock.get("avg_10_volume")
                ),
                "short_ratio": normalize_api_scalar(stock.get("short_ratio")),
                "short_pct_float": normalize_api_scalar(
                    stock.get("short_percent_of_shares_outstanding")
                ),
                "inst_ownership_pct": normalize_api_scalar(
                    stock.get("percent_held_by_institutions")
                ),
                "52w_high": normalize_api_scalar(px.get("fifty_two_week_high")),
                "52w_low": normalize_api_scalar(px.get("fifty_two_week_low")),
                "beta": normalize_api_scalar(px.get("beta")),
            })
            if out:
                circuit.record_success()
        except ProviderExhausted:
            raise
        except Exception:
            circuit.record_failure(f"statistics {ticker}")
        return out

    def _fundamentals_finnhub(self, ticker: str) -> Dict:
        circuit = ProviderCircuit.get("Finnhub-fundamentals", fail_limit=4)
        if not FINNHUB_API_KEY or not circuit.available():
            return {}
        session = get_market_router().session
        out: Dict[str, Any] = {}

        def _get(path: str, extra: Optional[Dict] = None) -> Dict:
            params = {"symbol": ticker, "token": FINNHUB_API_KEY}
            if extra:
                params.update(extra)
            resp = session.get(f"{FINNHUB_BASE_URL}{path}", params=params, timeout=15)
            kind = classify_http_error(resp.status_code, resp.text or "")
            if kind in ("credit", "auth"):
                circuit.trip(f"{path} HTTP {resp.status_code}")
                raise ProviderExhausted("Finnhub", f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                return {}
            payload = resp.json()
            return payload if isinstance(payload, dict) else {}

        def _pct(val):
            num = normalize_api_scalar(val)
            if is_missing_value(num):
                return num
            try:
                num = float(num)
            except (TypeError, ValueError):
                return num
            if abs(num) > 1.5:
                return num / 100.0
            return num

        profile = _get("/stock/profile2")
        if profile:
            mc = normalize_api_scalar(profile.get("marketCapitalization"))
            if not is_missing_value(mc):
                mc = float(mc) * 1_000_000.0
            shares = normalize_api_scalar(profile.get("shareOutstanding"))
            if not is_missing_value(shares):
                shares = float(shares) * 1_000_000.0
            out.update({
                "name": profile.get("name") or ticker,
                "sector": profile.get("finnhubIndustry") or "Unknown",
                "market_cap": mc,
                "shares_outstanding": shares,
            })

        metric_payload = _get("/stock/metric", {"metric": "all"})
        metric = metric_payload.get("metric") or {}
        if metric:
            out.update({
                "pe_ratio": normalize_api_scalar(metric.get("peTTM") or metric.get("peNormalizedAnnual")),
                "forward_pe": normalize_api_scalar(metric.get("forwardPE")),
                "peg_ratio": normalize_api_scalar(metric.get("pegRatio")),
                "beta": normalize_api_scalar(metric.get("beta")),
                "52w_high": normalize_api_scalar(metric.get("52WeekHigh")),
                "52w_low": normalize_api_scalar(metric.get("52WeekLow")),
                "profit_margin": _pct(metric.get("netProfitMarginTTM")),
                "revenue_growth": _pct(metric.get("revenueGrowthTTMYoy")),
                "earnings_growth": _pct(metric.get("epsGrowthTTMYoy")),
                "return_on_equity": _pct(metric.get("roeTTM")),
                "free_cash_flow": normalize_api_scalar(
                    metric.get("freeCashFlowTTM") or metric.get("freeCashFlowAnnual")
                ),
                "short_pct_float": _pct(
                    metric.get("shortInterestPercentFloat") or metric.get("shortPercentOutstanding")
                ),
                "short_ratio": normalize_api_scalar(metric.get("shortRatio")),
            })

        insider = _get("/stock/insider-transactions")
        rows = insider.get("data") if isinstance(insider, dict) else None
        if isinstance(rows, list) and rows:
            buys = sells = 0
            net_shares = 0.0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("transactionCode") or "").upper()
                change = normalize_api_scalar(row.get("change"))
                if code == "P":
                    buys += 1
                    if not is_missing_value(change):
                        net_shares += float(change)
                elif code == "S":
                    sells += 1
                    if not is_missing_value(change):
                        net_shares += float(change)
            if buys or sells:
                out["insider_buy_transactions"] = float(buys)
                out["insider_sell_transactions"] = float(sells)
                out["insider_net_shares"] = net_shares
        if out:
            circuit.record_success()
        return out

    def _fundamentals_yfinance(self, ticker: str) -> Dict:
        try:
            tk = yf.Ticker(ticker)
            raw = tk.info or {}
        except Exception:
            return {}
        if not isinstance(raw, dict) or not raw:
            return {}
        out = {
            "name": raw.get("longName") or raw.get("shortName") or ticker,
            "sector": raw.get("sector") or "Unknown",
            "industry": raw.get("industry") or "Unknown",
            "market_cap": normalize_api_scalar(raw.get("marketCap")),
            "pe_ratio": normalize_api_scalar(raw.get("trailingPE")),
            "forward_pe": normalize_api_scalar(raw.get("forwardPE")),
            "peg_ratio": normalize_api_scalar(raw.get("pegRatio")),
            "beta": normalize_api_scalar(raw.get("beta")),
            "profit_margin": normalize_api_scalar(raw.get("profitMargins")),
            "revenue_growth": normalize_api_scalar(raw.get("revenueGrowth")),
            "earnings_growth": normalize_api_scalar(raw.get("earningsGrowth")),
            "debt_to_equity": normalize_api_scalar(raw.get("debtToEquity")),
            "free_cash_flow": normalize_api_scalar(raw.get("freeCashflow")),
            "return_on_equity": normalize_api_scalar(raw.get("returnOnEquity")),
            "52w_high": normalize_api_scalar(raw.get("fiftyTwoWeekHigh")),
            "52w_low": normalize_api_scalar(raw.get("fiftyTwoWeekLow")),
            "avg_volume": normalize_api_scalar(raw.get("averageVolume")),
            "shares_float": normalize_api_scalar(raw.get("floatShares")),
            "shares_outstanding": normalize_api_scalar(raw.get("sharesOutstanding")),
            "inst_ownership_pct": normalize_api_scalar(raw.get("heldPercentInstitutions")),
            "analyst_count": normalize_api_scalar(raw.get("numberOfAnalystOpinions")),
            "target_price": normalize_api_scalar(raw.get("targetMeanPrice")),
            "short_pct_float": normalize_api_scalar(raw.get("shortPercentOfFloat")),
            "short_ratio": normalize_api_scalar(raw.get("shortRatio")),
        }
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earningsDate")
                if ed is not None:
                    out["earnings_date"] = str(ed)
        except Exception:
            pass
        try:
            txns = tk.insider_transactions
            if txns is not None and len(txns) > 0:
                text_col = "Text" if "Text" in txns.columns else None
                buys = sells = 0
                if text_col:
                    for text in txns[text_col].astype(str).str.lower():
                        if "purchase" in text or "buy" in text:
                            buys += 1
                        elif "sale" in text or "sold" in text:
                            sells += 1
                if buys or sells:
                    out["insider_buy_transactions"] = float(buys)
                    out["insider_sell_transactions"] = float(sells)
        except Exception:
            pass
        return out

    @staticmethod
    def _extract_insider_activity(modules: Dict) -> Dict:
        """
        Normalise MBOUM insider modules into flat, comparable fields.

        `net-share-purchase-activity` gives an aggregated six-month view;
        `insider-transactions` gives the individual filings. Both are optional:
        when neither is present every field stays missing so downstream
        scoring falls back to neutral instead of inventing conviction.
        """
        out: Dict[str, Any] = {
            "insider_net_purchase_pct": np.nan,
            "insider_net_shares": np.nan,
            "insider_buy_transactions": np.nan,
            "insider_sell_transactions": np.nan,
            "insider_period": None,
        }

        net = modules.get("net-share-purchase-activity")
        if isinstance(net, dict) and net:
            out["insider_net_purchase_pct"] = normalize_api_scalar(
                net.get("netPercentInsiderShares")
            )
            out["insider_net_shares"] = normalize_api_scalar(
                net.get("netInfoShares")
            )
            buy_count = normalize_api_scalar(net.get("buyInfoCount"))
            sell_count = normalize_api_scalar(net.get("sellInfoCount"))
            if not is_missing_value(buy_count):
                out["insider_buy_transactions"] = buy_count
            if not is_missing_value(sell_count):
                out["insider_sell_transactions"] = sell_count
            period = net.get("period")
            if isinstance(period, str) and period:
                out["insider_period"] = period

        # Fall back to counting the raw filings when the aggregate is absent.
        if is_missing_value(out["insider_buy_transactions"]):
            txns = modules.get("insider-transactions")
            rows = txns.get("transactions") if isinstance(txns, dict) else None
            if isinstance(rows, list) and rows:
                buys = sells = 0
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    text = str(row.get("transactionText", "")).lower()
                    if "purchase" in text or "buy" in text:
                        buys += 1
                    elif "sale" in text or "sold" in text or "sell" in text:
                        sells += 1
                if buys or sells:
                    out["insider_buy_transactions"] = float(buys)
                    out["insider_sell_transactions"] = float(sells)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 -- FIVE-INVESTOR PANEL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class InvestorPanel:
    """
    Scores each survivor through 5 investor lenses:
      Livermore (20%), Druckenmiller (20%), Lynch (20%),
      Minervini (20%), O'Neil (20%).
    Composite minimum: 60. Consensus: >= 3 panelists score >= 55.
    """

    def __init__(self, macro: Optional["MacroRegime"] = None):
        self.benchmark_data = None
        self.macro = macro
        # Universe-wide percentile lookups (computed lazily)
        self._rs_universe_returns: Optional[np.ndarray] = None

    def set_universe_returns(self, all_data: Dict[str, pd.DataFrame]) -> None:
        """Cache 63-day returns of the broad guarded universe for true
        IBD-style RS-rank percentile (vs prior heuristic absolute thresholds)."""
        rets = []
        for df in all_data.values():
            try:
                if len(df) >= 64:
                    last = df["Close"].iloc[-1]
                    base = df["Close"].iloc[-64]
                    if base and base > 0 and pd.notna(last) and pd.notna(base):
                        rets.append(float(last / base - 1.0))
            except Exception:
                continue
        if rets:
            self._rs_universe_returns = np.array(rets)

    def rs_percentile(self, ret_63d: float) -> float:
        """Return the IBD-style RS rank percentile (0-100) of a 3-month
        return vs the live universe. Falls back to 50 if uncalibrated."""
        if self._rs_universe_returns is None or len(self._rs_universe_returns) < 50:
            return 50.0
        if ret_63d is None or pd.isna(ret_63d):
            return 50.0
        return float((self._rs_universe_returns < ret_63d).mean() * 100.0)

    def load_benchmark(self):
        """Load a live market INDEX for relative strength.

        Uses the S&P 500 series already fetched for macro context. If that
        snapshot is missing, refetches the live ^GSPC index -- never an
        equity ETF such as SPY, and never a stock pick.
        """
        try:
            if self.macro is not None:
                snap = self.macro.snapshot.get("spx") or {}
                bench = snap.get("df")
                if bench is not None and len(bench) > 100:
                    self.benchmark_data = bench
                    log.info(
                        f"  Market benchmark loaded ({len(bench)} bars via live SPX) "
                        "for relative strength"
                    )
                    return
                bench = self.macro._series("^GSPC", range_="2y")
                if bench is not None and len(bench) > 100:
                    self.benchmark_data = bench
                    log.info(
                        f"  Market benchmark loaded ({len(bench)} bars via live ^GSPC) "
                        "for relative strength"
                    )
                    return
            log.warning("  Could not load live market index benchmark")
        except Exception as e:
            log.warning(f"  Could not load live market index benchmark: {e}")

    def score_all(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
        fundamentals: Dict[str, Dict],
        apply_filter: bool = True,
    ) -> List[Dict]:
        """Score all survivors through the 5-investor panel."""
        log.info(f"STAGE 5: 5-Investor Panel -- scoring {len(survivors)} survivors")
        self.load_benchmark()

        scored = []
        for s in survivors:
            ticker = s["ticker"]
            df = all_data.get(ticker)
            fund = fundamentals.get(ticker, {})

            if df is None:
                continue

            # Compute individual panel scores
            liv = self._score_livermore(ticker, df, fund)
            druck = self._score_druckenmiller(ticker, df, fund)
            lynch = self._score_lynch(ticker, df, fund)
            minerv = self._score_minervini(ticker, df, fund)
            oneil = self._score_oneil(ticker, df, fund)

            composite = (
                liv * 0.20 + druck * 0.20 + lynch * 0.20 +
                minerv * 0.20 + oneil * 0.20
            )

            # Count panelists scoring >= 55
            panel_scores = [liv, druck, lynch, minerv, oneil]
            consensus_count = sum(1 for ps in panel_scores if ps >= 55)

            s["panel_livermore"] = round(liv, 1)
            s["panel_druckenmiller"] = round(druck, 1)
            s["panel_lynch"] = round(lynch, 1)
            s["panel_minervini"] = round(minerv, 1)
            s["panel_oneil"] = round(oneil, 1)
            s["panel_composite_score"] = round(composite, 1)
            s["panel_consensus"] = consensus_count

            # Live conviction overlays -- who is buying, and is there fuel.
            insider = MomentumQuality.insider_score(fund)
            squeeze = MomentumQuality.squeeze_score(fund)
            s["insider_score"] = round(insider, 1)
            s["squeeze_score"] = round(squeeze, 1)
            s["insider_net_purchase_pct"] = fund.get("insider_net_purchase_pct", np.nan)
            s["insider_buy_transactions"] = fund.get("insider_buy_transactions", np.nan)
            s["insider_sell_transactions"] = fund.get("insider_sell_transactions", np.nan)
            s["short_pct_float"] = fund.get("short_pct_float", np.nan)

            flags = s.get("flags")
            if isinstance(flags, list):
                if insider >= 70:
                    flags.append("INSIDER_BUYING")
                elif insider <= 30:
                    flags.append("INSIDER_SELLING")
                if squeeze >= 70:
                    flags.append("HIGH_SHORT_INTEREST")

            # Merge fundamentals into record
            s["name"] = fund.get("name", ticker)
            s["sector"] = fund.get("sector", "Unknown")
            s["market_cap"] = fund.get("market_cap", np.nan)

            scored.append(s)

        if not apply_filter:
            qualified = sum(
                1 for s in scored
                if s["panel_composite_score"] >= 60 and s["panel_consensus"] >= 3
            )
            log.info(
                f"STAGE 5 COMPLETE: {len(scored)} scored; {qualified} meet "
                f"panel quality (composite >= 60, consensus >= 3)."
            )
            return scored

        # Filter: composite >= 60 AND consensus >= 3
        final = [
            s for s in scored
            if s["panel_composite_score"] >= 60 and s["panel_consensus"] >= 3
        ]

        log.info(
            f"STAGE 5 COMPLETE: {len(final)} survivors passed panel "
            f"(composite >= 60, consensus >= 3). "
            f"{len(scored) - len(final)} eliminated."
        )
        return final

    # ── LIVERMORE: Pure Momentum & Tape Reading ─────────────────────────

    def _score_livermore(self, ticker: str, df: pd.DataFrame, fund: Dict) -> float:
        last = df.iloc[-1]
        close = last["Close"]

        # 1. Price Trend Strength (30%)
        sma50 = last.get("SMA_50", np.nan)
        sma200 = last.get("SMA_200", np.nan)
        ema20 = last.get("EMA_20", np.nan)
        ema50 = last.get("EMA_50", np.nan)

        trend_score = 50
        if not pd.isna(sma50) and not pd.isna(sma200):
            if close > ema20 > ema50 > sma50 > sma200:
                trend_score = 95  # Perfect MA stack
            elif close > sma50 > sma200:
                trend_score = 80
            elif close > sma200:
                trend_score = 60

            # Check trend consistency: count days above SMA50 in last 20
            above_count = (df["Close"].iloc[-20:] > df["SMA_50"].iloc[-20:]).sum()
            if above_count >= 18:
                trend_score = min(100, trend_score + 10)
            elif above_count < 10:
                trend_score = max(0, trend_score - 15)

        # 2. Breakout Quality (25%)
        bb_upper = last.get("BB_upper", np.nan)
        volume_ratio = last.get("Volume_Ratio", np.nan)

        breakout_score = 50
        high_20 = last.get("High_Close_20", np.nan)
        if not pd.isna(high_20) and close >= high_20:
            breakout_score = 75
            if not pd.isna(volume_ratio) and volume_ratio > 2.0:
                breakout_score = 95
            elif not pd.isna(volume_ratio) and volume_ratio > 1.5:
                breakout_score = 85
        if not pd.isna(bb_upper) and close > bb_upper:
            breakout_score = max(breakout_score, 80)

        # 3. Volume Confirmation (20%)
        vol_score = 50
        if len(df) >= 20:
            recent = df.iloc[-20:]
            up_days = recent[recent["Return_1d"] > 0]
            down_days = recent[recent["Return_1d"] < 0]
            avg_up_vol = up_days["Volume"].mean() if len(up_days) > 0 else 0
            avg_down_vol = down_days["Volume"].mean() if len(down_days) > 0 else 1
            vol_ratio = avg_up_vol / max(avg_down_vol, 1)
            if vol_ratio > 1.5:
                vol_score = 90
            elif vol_ratio > 1.2:
                vol_score = 75
            elif vol_ratio > 1.0:
                vol_score = 60
            else:
                vol_score = 35

        # 4. Pullback Behavior (15%)
        pullback_score = 60
        if len(df) >= 50:
            rolling_high = df["Close"].rolling(50).max().iloc[-1]
            drawdown = (close - rolling_high) / rolling_high if rolling_high > 0 else 0
            if drawdown > -0.03:
                pullback_score = 90
            elif drawdown > -0.07:
                pullback_score = 75
            elif drawdown > -0.12:
                pullback_score = 55
            else:
                pullback_score = 35

        # 5. Relative Strength vs live SPX (10%)
        rs_score = 60
        if self.benchmark_data is not None and len(self.benchmark_data) >= 126 and len(df) >= 126:
            bench_close = self.benchmark_data["Close"]
            stock_ret_1m = (close / df["Close"].iloc[-21]) - 1 if len(df) >= 21 else 0
            stock_ret_3m = (close / df["Close"].iloc[-63]) - 1 if len(df) >= 63 else 0
            stock_ret_6m = (close / df["Close"].iloc[-126]) - 1 if len(df) >= 126 else 0

            spy_ret_1m = (bench_close.iloc[-1] / bench_close.iloc[-21]) - 1 if len(bench_close) >= 21 else 0
            spy_ret_3m = (bench_close.iloc[-1] / bench_close.iloc[-63]) - 1 if len(bench_close) >= 63 else 0
            spy_ret_6m = (bench_close.iloc[-1] / bench_close.iloc[-126]) - 1 if len(bench_close) >= 126 else 0

            outperform_count = sum([
                stock_ret_1m > spy_ret_1m,
                stock_ret_3m > spy_ret_3m,
                stock_ret_6m > spy_ret_6m,
            ])
            excess = (stock_ret_3m - spy_ret_3m)
            if outperform_count == 3 and excess > 0.10:
                rs_score = 95
            elif outperform_count >= 2:
                rs_score = 80
            elif outperform_count == 1:
                rs_score = 55
            else:
                rs_score = 35

        return (
            trend_score * 0.30 + breakout_score * 0.25 +
            vol_score * 0.20 + pullback_score * 0.15 + rs_score * 0.10
        )

    # ── DRUCKENMILLER: Macro-Catalyst + Asymmetric Risk/Reward ─────────

    def _score_druckenmiller(self, ticker: str, df: pd.DataFrame, fund: Dict) -> float:
        last = df.iloc[-1]
        close = last["Close"]

        # 1. Macro Alignment (25%) -- sector momentum as proxy
        macro_score = 60
        sector = fund.get("sector", "Unknown")
        # Use relative strength vs the live SPX index as macro proxy
        if self.benchmark_data is not None and len(df) >= 63:
            stock_ret = (close / df["Close"].iloc[-63]) - 1
            spy_ret = (
                self.benchmark_data["Close"].iloc[-1] / self.benchmark_data["Close"].iloc[-63] - 1
            ) if len(self.benchmark_data) >= 63 else 0
            excess = stock_ret - spy_ret
            if excess > 0.15:
                macro_score = 90
            elif excess > 0.05:
                macro_score = 75
            elif excess > 0:
                macro_score = 60
            else:
                macro_score = 40

        # 2. Catalyst Proximity (25%) -- earnings date proximity.
        # Druckenmiller-style: catalyst SOON (sweet spot 7-30 days out). Too
        # close (<= 5 days) = binary event risk; flag separately so options
        # stage can avoid IV-crush trades. Pure post-earnings drift (5-21
        # days post-print) = highest expected drift edge per academic research.
        catalyst_score = 55
        earnings_date = fund.get("earnings_date")
        if earnings_date:
            try:
                ed = pd.Timestamp(earnings_date).tz_localize(None)
                today_naive = pd.Timestamp(now_et_dt().date())
                days_to = (ed - today_naive).days
                if 0 < days_to <= 5:
                    catalyst_score = 70  # Imminent -- binary risk, slight bonus
                elif 5 < days_to <= 14:
                    catalyst_score = 92  # Sweet spot
                elif 14 < days_to <= 30:
                    catalyst_score = 78
                elif 30 < days_to <= 60:
                    catalyst_score = 60
                elif -21 <= days_to <= 0:
                    catalyst_score = 80  # Just-printed -- post-earnings drift
            except Exception:
                pass

        # 3. Risk/Reward Asymmetry (25%)
        rr_score = 55
        sma50 = last.get("SMA_50", np.nan)
        if not pd.isna(sma50) and sma50 > 0:
            # Support at SMA50; resistance at recent high
            support = sma50
            if len(df) >= 50:
                resistance = df["Close"].iloc[-50:].max()
            else:
                resistance = close * 1.1
            risk = (close - support) / close if close > support else 0.05
            reward = (resistance - close) / close if resistance > close else 0.02
            rr_ratio = reward / max(risk, 0.01)
            if rr_ratio >= 3:
                rr_score = 95
            elif rr_ratio >= 2:
                rr_score = 80
            elif rr_ratio >= 1.5:
                rr_score = 65
            else:
                rr_score = 40

        # 4. Institutional Flow (15%) -- OBV trend
        flow_score = 55
        if len(df) >= 20:
            obv = df["OBV"]
            obv_sma = obv.rolling(20).mean()
            if obv.iloc[-1] > obv_sma.iloc[-1]:
                flow_score = 80
                if obv.iloc[-1] > obv.iloc[-20]:
                    flow_score = 90
            else:
                flow_score = 40

        # 5. Position Sizing Confidence (10%) -- empirical Kelly proxy
        # built from the stock's own rolling win-rate and payoff ratio over
        # the last 60 sessions. Replaces the prior circular self-average
        # which contained zero new information.
        sizing_score = 50.0
        if len(df) >= 63:
            recent = df["Return_1d"].iloc[-60:].dropna()
            if len(recent) >= 30:
                wins = recent[recent > 0]
                losses = recent[recent < 0]
                w = len(wins) / max(len(recent), 1)
                avg_win = wins.mean() if len(wins) else 0.0
                avg_loss = abs(losses.mean()) if len(losses) else 1.0
                payoff = avg_win / max(avg_loss, 1e-6)
                # Kelly fraction: f* = w - (1-w)/payoff. Cap, then map to 0-100.
                kelly = w - (1 - w) / max(payoff, 1e-6)
                kelly_capped = clamp(kelly, -0.5, 0.5)
                # Map [-0.5, 0.5] -> [0, 100]; positive Kelly = positive edge.
                sizing_score = clamp(50 + kelly_capped * 100, 0, 100)
        # Pull macro regime in as additional weight (Druckenmiller is the
        # macro guy -- if regime is risk-off, even great names get marked down)
        if self.macro is not None:
            macro_overlay = self.macro.regime_score
            macro_score = 0.7 * macro_score + 0.3 * macro_overlay

        return (
            macro_score * 0.25 + catalyst_score * 0.25 +
            rr_score * 0.25 + flow_score * 0.15 + sizing_score * 0.10
        )

    # ── LYNCH: Growth At Reasonable Price ─────────────────────────────

    def _score_lynch(self, ticker: str, df: pd.DataFrame, fund: Dict) -> float:
        if fund.get("fundamentals_missing") or fund.get("fundamentals_quality") == "failed":
            return 35.0

        # 1. Earnings Growth (30%)
        eg_score = 50
        eg = fund.get("earnings_growth", np.nan)
        if not is_missing_value(eg):
            if eg > 0.25:
                eg_score = 95
            elif eg > 0.15:
                eg_score = 80
            elif eg > 0.05:
                eg_score = 60
            elif eg > 0:
                eg_score = 45
            else:
                eg_score = 30

        # 2. PEG Ratio (25%)
        peg_score = 50
        peg = fund.get("peg_ratio", np.nan)
        if not is_missing_value(peg) and peg > 0:
            if peg < 0.75:
                peg_score = 95
            elif peg < 1.0:
                peg_score = 80
            elif peg < 1.5:
                peg_score = 60
            else:
                peg_score = 35

        # 3. Revenue Momentum (20%)
        rev_score = 50
        rg = fund.get("revenue_growth", np.nan)
        if not is_missing_value(rg):
            if rg > 0.20:
                rev_score = 90
            elif rg > 0.10:
                rev_score = 75
            elif rg > 0:
                rev_score = 55
            else:
                rev_score = 30

        # 4. Balance Sheet Health (15%)
        bs_score = 60
        dte = fund.get("debt_to_equity", np.nan)
        fcf = fund.get("free_cash_flow", np.nan)
        if not is_missing_value(dte):
            if dte < 30:
                bs_score = 90
            elif dte < 80:
                bs_score = 75
            elif dte < 150:
                bs_score = 55
            else:
                bs_score = 35
        if not is_missing_value(fcf) and fcf > 0:
            bs_score = min(100, bs_score + 10)

        # 5. Story Clarity (10%) -- analyst coverage as proxy
        story_score = 55
        analysts = fund.get("analyst_count", np.nan)
        if not is_missing_value(analysts):
            if analysts >= 20:
                story_score = 90
            elif analysts >= 10:
                story_score = 75
            elif analysts >= 5:
                story_score = 60
            else:
                story_score = 45

        # ETF adjustment only when explicitly identified; unknown fundamentals
        # must not receive neutral GARP credit in a real-money scan.
        sector = fund.get("sector", "Unknown")
        quote_type = str(fund.get("quote_type", "")).upper()
        if quote_type in {"ETF", "ETP", "MUTUALFUND"}:
            # Assign neutral fundamental scores for ETFs
            eg_score = 60
            peg_score = 60
            rev_score = 60
        elif sector in {"Unknown", ""} or fund.get("fundamentals_quality") == "failed":
            eg_score = min(eg_score, 35)
            peg_score = min(peg_score, 35)
            rev_score = min(rev_score, 35)
            bs_score = min(bs_score, 45)
            story_score = min(story_score, 40)

        return (
            eg_score * 0.30 + peg_score * 0.25 +
            rev_score * 0.20 + bs_score * 0.15 + story_score * 0.10
        )

    # ── MINERVINI: VCP + Stage Analysis ──────────────────────────────

    def _score_minervini(self, ticker: str, df: pd.DataFrame, fund: Dict) -> float:
        last = df.iloc[-1]
        close = last["Close"]

        # 1. Stage Analysis (30%)
        stage_score = 50
        sma50 = last.get("SMA_50", np.nan)
        sma200 = last.get("SMA_200", np.nan)
        ema20 = last.get("EMA_20", np.nan)
        ema50 = last.get("EMA_50", np.nan)

        if not pd.isna(sma50) and not pd.isna(sma200):
            if close > ema20 > ema50 > sma50 > sma200:
                stage_score = 95  # Perfect Stage 2
            elif close > sma50 > sma200:
                stage_score = 80  # Stage 2 confirmed
                # Check if early or late stage 2
                pct_above_200 = (close - sma200) / sma200
                if pct_above_200 < 0.30:
                    stage_score = 85  # Early, more room
                elif pct_above_200 > 0.80:
                    stage_score = 65  # Extended
            elif close > sma200:
                stage_score = 55
            else:
                stage_score = 30

        # 2. Volatility Contraction Pattern (25%)
        vcp_score = 50
        if len(df) >= 60:
            atr = df["ATR_14"]
            # Compare ATR at 3 points: 60d ago, 30d ago, today
            atr_60 = atr.iloc[-60] if not pd.isna(atr.iloc[-60]) else atr.iloc[-50]
            atr_30 = atr.iloc[-30] if not pd.isna(atr.iloc[-30]) else atr.iloc[-20]
            atr_now = atr.iloc[-1]

            if not pd.isna(atr_60) and not pd.isna(atr_30) and not pd.isna(atr_now):
                if atr_now < atr_30 < atr_60:
                    vcp_score = 92  # Classic contraction
                elif atr_now < atr_60:
                    vcp_score = 72  # Some contraction
                else:
                    vcp_score = 40  # Expanding volatility

            # Check for volume dry-up during contraction
            vol_ratio = last.get("Volume_Ratio", np.nan)
            recent_avg_vol = df["Volume"].iloc[-10:].mean()
            longer_avg_vol = df["Volume"].iloc[-60:-10].mean()
            if longer_avg_vol > 0 and recent_avg_vol / longer_avg_vol < 0.7:
                vcp_score = min(100, vcp_score + 10)

        # 3. Risk/Entry Quality (20%)
        entry_score = 55
        # Pivot = recent high; check distance
        if len(df) >= 30:
            pivot = df["Close"].iloc[-30:].max()
            dist = (pivot - close) / pivot if pivot > 0 else 1
            if dist < 0.02:
                entry_score = 95  # At pivot
            elif dist < 0.05:
                entry_score = 80
            elif dist < 0.10:
                entry_score = 60
            else:
                entry_score = 35

            # Stop loss distance (below SMA50)
            if not pd.isna(sma50) and sma50 > 0:
                stop_dist = (close - sma50) / close
                if stop_dist < 0.05:
                    entry_score = min(100, entry_score + 5)
                elif stop_dist > 0.10:
                    entry_score = max(0, entry_score - 10)

        # 4. Relative Strength Rank (15%) -- true universe-based percentile.
        # Minervini explicitly requires RS Rank >= 70 IBD-style. We compute
        # the percentile of this name's 3-month return vs the live universe
        # (replaces previous absolute thresholds which mis-fire across regimes).
        rs_rank_score = 60
        ret_63d = last.get("Return_63d", np.nan)
        ret_20d = last.get("Return_20d", np.nan)
        if not pd.isna(ret_63d):
            pct = self.rs_percentile(float(ret_63d))
            # Map IBD-style: >=85 elite, >=70 strong, >=50 average, <30 weak
            if pct >= 90:
                rs_rank_score = 95
            elif pct >= 80:
                rs_rank_score = 88
            elif pct >= 70:
                rs_rank_score = 78
            elif pct >= 55:
                rs_rank_score = 62
            elif pct >= 40:
                rs_rank_score = 48
            else:
                rs_rank_score = 30

        # 5. Earnings Acceleration (10%)
        ea_score = 55
        eg = fund.get("earnings_growth", np.nan)
        if not is_missing_value(eg):
            if eg > 0.30:
                ea_score = 90
            elif eg > 0.15:
                ea_score = 75
            elif eg > 0:
                ea_score = 60
            else:
                ea_score = 35

        return (
            stage_score * 0.30 + vcp_score * 0.25 +
            entry_score * 0.20 + rs_rank_score * 0.15 + ea_score * 0.10
        )

    # ── O'NEIL: CAN SLIM Composite ───────────────────────────────────

    def _score_oneil(self, ticker: str, df: pd.DataFrame, fund: Dict) -> float:
        last = df.iloc[-1]
        close = last["Close"]

        # C: Current Quarterly Earnings (20%)
        c_score = 50
        eg = fund.get("earnings_growth", np.nan)
        if not is_missing_value(eg):
            if eg > 0.40:
                c_score = 95
            elif eg > 0.25:
                c_score = 80
            elif eg > 0.10:
                c_score = 60
            else:
                c_score = 35

        # A: Annual Earnings (15%)
        a_score = 55
        roe = fund.get("return_on_equity", np.nan)
        if not is_missing_value(roe) and roe > 0:
            if roe > 0.25:
                a_score = 90
            elif roe > 0.15:
                a_score = 75
            elif roe > 0.08:
                a_score = 60
            else:
                a_score = 45

        # N: New Factor (15%) -- near highs + catalyst
        n_score = 55
        hi52 = fund.get("52w_high", np.nan)
        if not is_missing_value(hi52) and hi52 > 0:
            pct_from_high = (hi52 - close) / hi52
            if pct_from_high < 0.05:
                n_score = 90  # Near 52w high
            elif pct_from_high < 0.10:
                n_score = 75
            elif pct_from_high < 0.20:
                n_score = 55
            else:
                n_score = 35

        # S: Supply/Demand (15%)
        s_score = 55
        float_shares = fund.get("shares_float", np.nan)
        vol_ratio = last.get("Volume_Ratio", np.nan)
        if not is_missing_value(float_shares):
            if float_shares < 20_000_000:
                s_score = 85  # Tight float
            elif float_shares < 50_000_000:
                s_score = 70
            elif float_shares < 200_000_000:
                s_score = 55
            else:
                s_score = 40
        if not pd.isna(vol_ratio) and vol_ratio > 1.5:
            s_score = min(100, s_score + 15)

        # L: Leader (15%) -- relative strength
        l_score = 60
        ret_63d = last.get("Return_63d", np.nan)
        if not pd.isna(ret_63d):
            if ret_63d > 0.25:
                l_score = 92
            elif ret_63d > 0.15:
                l_score = 78
            elif ret_63d > 0.05:
                l_score = 60
            else:
                l_score = 40

        # I: Institutional Sponsorship (10%)
        i_score = 55
        inst = fund.get("inst_ownership_pct", np.nan)
        if not is_missing_value(inst):
            if 0.30 <= inst <= 0.80:
                i_score = 85  # Sweet spot
            elif inst > 0.80:
                i_score = 65  # Crowded
            elif inst > 0.10:
                i_score = 55
            else:
                i_score = 40

        # M: Market Direction (10%) -- Macro regime composite.
        # O'Neil emphasized following the general market; we use the full
        # live macro snapshot (VIX, yields, DXY, gold, oil, breadth) instead
        # of a single ETF. Falls back to the live SPX MA stack if needed.
        if self.macro is not None:
            m_score = self.macro.panel_m_score()
        else:
            m_score = 60
            if self.benchmark_data is not None and len(self.benchmark_data) >= 50:
                spy_close = self.benchmark_data["Close"]
                spy_sma50 = spy_close.rolling(50).mean()
                spy_sma200 = spy_close.rolling(200).mean()
                if (not pd.isna(spy_sma50.iloc[-1]) and not pd.isna(spy_sma200.iloc[-1]) and
                        spy_close.iloc[-1] > spy_sma50.iloc[-1] > spy_sma200.iloc[-1]):
                    m_score = 90
                elif (not pd.isna(spy_sma50.iloc[-1]) and
                      spy_close.iloc[-1] > spy_sma50.iloc[-1]):
                    m_score = 70
                elif (not pd.isna(spy_sma200.iloc[-1]) and
                      spy_close.iloc[-1] > spy_sma200.iloc[-1]):
                    m_score = 55
                else:
                    m_score = 30

        return (
            c_score * 0.20 + a_score * 0.15 + n_score * 0.15 +
            s_score * 0.15 + l_score * 0.15 + i_score * 0.10 + m_score * 0.10
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 -- OPTIONS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class OptionsEvaluator:
    """
    Evaluate long call options for qualifying stocks.
    Prioritizes better call-trade structure, not just closest-to-ATM heuristics.
    """

    #: Short-to-mid-term horizon: contracts must outlive the ~20-session
    #: equity thesis with buffer, without paying for LEAP-style time value.
    MIN_DTE = 21
    MAX_DTE = 60
    MIN_DELTA = 0.40
    MAX_DELTA = 0.70
    MAX_SPREAD_PCT = 12.0
    MIN_OPEN_INTEREST = 150
    MIN_VOLUME = 20

    @staticmethod
    def evaluate(survivors: List[Dict], fundamentals: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        """
        For each survivor, check options chain and find the best qualifying contract.

        IV-crush guard: if earnings fall *inside* the option's lifetime AND
        within 7 calendar days of today, we skip the contract -- buying
        long calls into earnings is a documented capital killer (post-print
        IV typically collapses 30-60% wiping out gains from intrinsic moves).
        """
        log.info(f"STAGE 6: Options Evaluation -- {len(survivors)} survivors")

        for s in survivors:
            ticker = s.get("ticker")
            fund = (fundamentals or {}).get(ticker, {})
            opt = OptionsEvaluator._find_best_option(s, fund)
            s.update(opt)
            s["trade_setup_score"] = round(OptionsEvaluator._trade_setup_score(s), 1)

        options_found = sum(1 for s in survivors if s.get("option_candidate") == "Y")
        log.info(
            f"STAGE 6 COMPLETE: {options_found} stocks have qualifying options, "
            f"{len(survivors) - options_found} equity-only candidates."
        )
        return survivors

    @staticmethod
    def _find_best_option(candidate: Dict, fund: Optional[Dict] = None) -> Dict:
        """Find the best qualifying long call for a validated stock setup."""
        ticker = candidate["ticker"]
        current_price = normalize_api_scalar(candidate.get("price"))
        fund = fund or {}

        # Compute earnings-window IV-crush guard
        earnings_dt = None
        ed_raw = fund.get("earnings_date")
        if ed_raw:
            try:
                earnings_dt = pd.Timestamp(ed_raw).date()
            except Exception:
                earnings_dt = None

        result = {
            "option_candidate": "N",
            "option_strike": None,
            "option_expiry": None,
            "option_delta": None,
            "option_dte": None,
            "option_bid_ask_spread": None,
            "option_mid": None,
            "option_break_even": None,
            "option_break_even_pct": None,
            "option_iv": None,
            "option_theta": None,
            "option_moneyness_pct": None,
            "option_score": 0.0,
            "option_source": None,
        }

        try:
            # Source priority: MBOUM (primary when credits remain) ->
            # Massive -> Yahoo (free, quotes only).
            chain_data = []
            option_source = None
            if MBOUM_OPTIONS_KEY and ProviderCircuit.get("MBOUM-options").available():
                try:
                    chain_data = OptionsEvaluator._fetch_chain_mboum(ticker, current_price)
                    if chain_data:
                        option_source = "MBOUM"
                except ProviderExhausted:
                    chain_data = []
            if not chain_data:
                chain_data = OptionsEvaluator._fetch_chain_massive(ticker)
                option_source = "Massive" if chain_data else option_source
            if not chain_data:
                chain_data = OptionsEvaluator._fetch_chain_yahoo(ticker)
                option_source = "Yahoo" if chain_data else option_source

            if not chain_data:
                return result
            if option_source:
                DATA_SOURCE_USAGE["options"][option_source] += 1

            profile = OptionsEvaluator._target_profile(candidate)
            today = today_et()
            best_contract = None
            best_score = -1.0

            for contract in chain_data:
                exp_str = contract.get("expiry", "")
                strike = normalize_api_scalar(contract.get("strike"))
                bid = normalize_api_scalar(contract.get("bid"))
                ask = normalize_api_scalar(contract.get("ask"))
                oi = normalize_api_scalar(contract.get("oi"))
                vol = normalize_api_scalar(contract.get("volume"))
                iv = normalize_api_scalar(contract.get("iv"))
                delta = normalize_api_scalar(contract.get("delta"))
                theta = normalize_api_scalar(contract.get("theta"))
                mid = normalize_api_scalar(contract.get("mid"))
                break_even = normalize_api_scalar(contract.get("break_even"))
                contract_underlying = normalize_api_scalar(contract.get("underlying_price"))
                underlying_price = (
                    current_price
                    if not is_missing_value(current_price) and current_price > 0
                    else contract_underlying
                )

                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue

                dte = (exp_date - today).days
                if dte < OptionsEvaluator.MIN_DTE or dte > OptionsEvaluator.MAX_DTE:
                    continue

                # IV-crush guard: skip contracts that span an earnings event
                # within the next 7 days (binary risk + post-print IV collapse).
                # Allow if earnings is >7 days out OR strictly after expiry.
                if earnings_dt is not None:
                    days_to_earn = (earnings_dt - today).days
                    if 0 <= days_to_earn <= 7 and earnings_dt <= exp_date:
                        continue

                if is_missing_value(oi):
                    oi = 0
                if is_missing_value(vol):
                    vol = 0
                if oi < OptionsEvaluator.MIN_OPEN_INTEREST and vol < OptionsEvaluator.MIN_VOLUME:
                    continue

                has_live_quote = (
                    not is_missing_value(bid) and not is_missing_value(ask)
                    and bid > 0 and ask > 0 and ask >= bid
                )
                if not has_live_quote:
                    continue
                if is_missing_value(mid):
                    mid = (bid + ask) / 2
                if mid <= 0:
                    continue

                spread_pct = safe_div(ask - bid, mid, default=np.nan) * 100
                if pd.isna(spread_pct) or spread_pct < 0 or spread_pct > OptionsEvaluator.MAX_SPREAD_PCT:
                    continue

                if is_missing_value(iv) or iv <= 0 or iv > 3.0:
                    continue
                model_iv = max(iv, 0.20)

                if is_missing_value(delta) and underlying_price and underlying_price > 0:
                    delta = OptionsEvaluator._bs_delta(
                        underlying_price, strike, dte / 365.0, model_iv
                    )
                if is_missing_value(delta):
                    continue
                if delta < OptionsEvaluator.MIN_DELTA or delta > OptionsEvaluator.MAX_DELTA:
                    continue

                if is_missing_value(break_even) and underlying_price and underlying_price > 0:
                    break_even = strike + mid

                contract_score = OptionsEvaluator._score_contract(
                    profile=profile,
                    underlying_price=underlying_price,
                    strike=strike,
                    dte=dte,
                    delta=delta,
                    mid=mid,
                    spread_pct=spread_pct,
                    oi=oi,
                    vol=vol,
                    iv=model_iv,
                    theta=theta,
                    break_even=break_even,
                )

                if contract_score > best_score:
                    best_score = contract_score
                    moneyness_pct = (
                        safe_div(strike - underlying_price, underlying_price, default=np.nan)
                        if underlying_price and underlying_price > 0 else np.nan
                    )
                    break_even_pct = (
                        safe_div(break_even - underlying_price, underlying_price, default=np.nan)
                        if break_even and underlying_price and underlying_price > 0 else np.nan
                    )
                    best_contract = {
                        "option_candidate": "Y",
                        "option_strike": strike,
                        "option_expiry": exp_str,
                        "option_delta": round(delta, 3),
                        "option_dte": dte,
                        "option_bid_ask_spread": round(spread_pct, 1),
                        "option_mid": round(mid, 2),
                        "option_break_even": round(break_even, 2) if break_even else None,
                        "option_break_even_pct": (
                            round(break_even_pct * 100, 1)
                            if not pd.isna(break_even_pct) else None
                        ),
                        "option_iv": round(model_iv, 3),
                        "option_theta": round(theta, 4) if not is_missing_value(theta) else None,
                        "option_moneyness_pct": (
                            round(moneyness_pct * 100, 1)
                            if not pd.isna(moneyness_pct) else None
                        ),
                        "option_score": round(contract_score, 1),
                        "option_source": option_source,
                    }

            if best_contract:
                return best_contract

        except Exception as e:
            log.debug(f"  Options error for {ticker}: {e}")

        return result

    @staticmethod
    def _fetch_chain_mboum(ticker: str, underlying_price: Optional[float] = None) -> List[Dict]:
        """Fetch the options chain via MBOUM Pro (options-tier key).

        Strategy:
          1. One meta call returns the list of all expirationDates plus the
             first expiration's chain and a live underlying quote.
          2. Filter expirations to our DTE window [MIN_DTE, MAX_DTE].
          3. Fetch the remaining qualifying expirations in parallel.

        MBOUM does not expose Greeks; delta is back-solved via Black-Scholes
        downstream using the quoted IV (which IS provided per-contract).
        """
        try:
            mboum = MboumAPI()
            meta = mboum.get_options_meta(ticker)
            if not meta:
                return []

            quote = meta.get("quote", {}) or {}
            quote_price = (
                normalize_api_scalar(quote.get("regularMarketPrice"))
                or normalize_api_scalar(quote.get("postMarketPrice"))
                or normalize_api_scalar(quote.get("preMarketPrice"))
            )
            if (not underlying_price or is_missing_value(underlying_price)) and not is_missing_value(quote_price):
                underlying_price = quote_price

            today = datetime.now(ET_TZ).date()
            min_date = today + timedelta(days=OptionsEvaluator.MIN_DTE - 2)
            max_date = today + timedelta(days=OptionsEvaluator.MAX_DTE + 2)
            min_epoch = int(datetime.combine(min_date, dtime.min, tzinfo=ET_TZ).timestamp())
            max_epoch = int(datetime.combine(max_date, dtime.min, tzinfo=ET_TZ).timestamp())

            exp_dates = meta.get("expirationDates", []) or []
            qualifying_epochs = [int(e) for e in exp_dates if min_epoch <= int(e) <= max_epoch]

            # Fast-path: meta itself returns the FIRST expiration's chain.
            chain_entries: List[Dict] = []
            initial_options = meta.get("options", []) or []
            initial_dates = {int(o.get("expirationDate", 0)) for o in initial_options}
            for o in initial_options:
                ep = int(o.get("expirationDate", 0))
                if min_epoch <= ep <= max_epoch:
                    chain_entries.append(o)

            # Fetch the other qualifying expirations in parallel.
            remaining = [e for e in qualifying_epochs if e not in initial_dates]
            if remaining:
                def _fetch_options_for_expiration(expiration_epoch: int) -> Any:
                    worker_mboum = MboumAPI()
                    return worker_mboum.get_options_for_expiration(ticker, expiration_epoch)

                with ThreadPoolExecutor(max_workers=min(8, len(remaining))) as ex:
                    futures = {
                        ex.submit(_fetch_options_for_expiration, e): e
                        for e in remaining
                    }
                    for f in as_completed(futures):
                        try:
                            entry = f.result()
                            if entry:
                                chain_entries.append(entry)
                        except Exception:
                            continue

            contracts: List[Dict] = []
            for entry in chain_entries:
                exp_epoch = int(entry.get("expirationDate", 0))
                if exp_epoch <= 0:
                    continue
                exp_str = datetime.fromtimestamp(exp_epoch, timezone.utc).strftime("%Y-%m-%d")
                for call in (entry.get("calls") or []):
                    bid = normalize_api_scalar(call.get("bid"))
                    ask = normalize_api_scalar(call.get("ask"))
                    last_price = normalize_api_scalar(call.get("lastPrice"))
                    midpoint = None
                    if not is_missing_value(bid) and not is_missing_value(ask) and (bid + ask) > 0:
                        midpoint = (bid + ask) / 2
                    elif not is_missing_value(last_price) and last_price > 0:
                        midpoint = last_price
                    strike = normalize_api_scalar(call.get("strike"))
                    iv = normalize_api_scalar(call.get("impliedVolatility"))
                    contracts.append({
                        "expiry": exp_str,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "mid": midpoint,
                        "oi": normalize_api_scalar(call.get("openInterest")),
                        "volume": normalize_api_scalar(call.get("volume")),
                        "iv": iv,
                        "delta": None,  # MBOUM does not return Greeks; BS-solved later
                        "theta": None,
                        "break_even": (
                            (strike + midpoint)
                            if not is_missing_value(strike) and not is_missing_value(midpoint)
                            else None
                        ),
                        "underlying_price": underlying_price,
                    })

            return contracts
        except ProviderExhausted:
            raise
        except Exception:
            return []

    @staticmethod
    def _fetch_chain_massive(ticker: str) -> List[Dict]:
        """Fetch options chain from Massive."""
        try:
            url = f"https://api.massive.com/v3/snapshot/options/{ticker}"
            today = today_et()
            params = {
                "contract_type": "call",
                "expiration_date.gte": (
                    today + timedelta(days=OptionsEvaluator.MIN_DTE)
                ).isoformat(),
                "expiration_date.lte": (
                    today + timedelta(days=OptionsEvaluator.MAX_DTE)
                ).isoformat(),
                "limit": 250,
                "sort": "expiration_date",
                "order": "asc",
                "apiKey": MASSIVE_API_KEY,
            }

            raw_results = []
            while url:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code != 200:
                    return []

                payload = r.json()
                results = payload.get("results", [])
                if not isinstance(results, list):
                    return []

                raw_results.extend(results)
                url = payload.get("next_url")
                params = {"apiKey": MASSIVE_API_KEY} if url else None

            contracts = []
            for opt in raw_results:
                details = opt.get("details", {})
                quote = opt.get("last_quote", {})
                day = opt.get("day", {})
                greeks = opt.get("greeks", {})
                underlying = opt.get("underlying_asset", {})

                if details.get("contract_type") != "call":
                    continue

                bid = normalize_api_scalar(quote.get("bid"))
                ask = normalize_api_scalar(quote.get("ask"))
                midpoint = normalize_api_scalar(quote.get("midpoint"))
                if is_missing_value(midpoint) and not is_missing_value(bid) and not is_missing_value(ask):
                    midpoint = (bid + ask) / 2

                if is_missing_value(bid):
                    bid = midpoint
                if is_missing_value(ask):
                    ask = midpoint
                if is_missing_value(bid) or is_missing_value(ask):
                    continue

                volume = normalize_api_scalar(day.get("volume"))
                if is_missing_value(volume):
                    volume = 0
                open_interest = normalize_api_scalar(opt.get("open_interest"))
                if is_missing_value(open_interest):
                    open_interest = 0

                contracts.append({
                    "expiry": details.get("expiration_date", ""),
                    "strike": normalize_api_scalar(details.get("strike_price")),
                    "bid": bid,
                    "ask": ask,
                    "mid": midpoint,
                    "oi": open_interest,
                    "volume": volume,
                    "iv": normalize_api_scalar(opt.get("implied_volatility")),
                    "delta": normalize_api_scalar(greeks.get("delta")),
                    "theta": normalize_api_scalar(greeks.get("theta")),
                    "break_even": normalize_api_scalar(opt.get("break_even_price")),
                    "underlying_price": normalize_api_scalar(underlying.get("price")),
                })

            return contracts
        except Exception:
            return []

    @staticmethod
    def _fetch_chain_yahoo(ticker: str) -> List[Dict]:
        """Fetch options chain from yfinance-backed Yahoo data."""
        try:
            ticker_obj = yf.Ticker(ticker)
            expirations = ticker_obj.options or []
            contracts = []
            underlying_price = None
            try:
                fast_info = getattr(ticker_obj, "fast_info", {}) or {}
                underlying_price = normalize_api_scalar(fast_info.get("lastPrice"))
            except Exception:
                underlying_price = None

            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                dte = (exp_date - today_et()).days
                if dte < OptionsEvaluator.MIN_DTE or dte > OptionsEvaluator.MAX_DTE:
                    continue

                try:
                    chain = ticker_obj.option_chain(exp_str)
                except Exception:
                    continue
                if chain is None or getattr(chain, "calls", None) is None or chain.calls.empty:
                    continue

                for _, call in chain.calls.iterrows():
                    bid = normalize_api_scalar(call.get("bid"))
                    ask = normalize_api_scalar(call.get("ask"))
                    last_price = normalize_api_scalar(call.get("lastPrice"))
                    midpoint = None
                    if not is_missing_value(bid) and not is_missing_value(ask) and (bid + ask) > 0:
                        midpoint = (bid + ask) / 2
                    elif not is_missing_value(last_price) and last_price > 0:
                        midpoint = last_price

                    strike = normalize_api_scalar(call.get("strike"))
                    contracts.append({
                        "expiry": exp_str,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "mid": midpoint,
                        "oi": normalize_api_scalar(call.get("openInterest")),
                        "volume": normalize_api_scalar(call.get("volume")),
                        "iv": normalize_api_scalar(call.get("impliedVolatility")),
                        "delta": None,
                        "theta": None,
                        "break_even": (
                            strike + midpoint
                            if not is_missing_value(strike) and not is_missing_value(midpoint)
                            else None
                        ),
                        "underlying_price": underlying_price,
                    })

            return contracts
        except Exception:
            return []

    @staticmethod
    def _bs_delta(S, K, T, sigma, r=0.05):
        """Black-Scholes call delta estimation."""
        try:
            from scipy.stats import norm
            import math

            if sigma <= 0 or S <= 0 or K <= 0 or T <= 0:
                return None
            d1 = (
                math.log(S / K) + (r + 0.5 * sigma ** 2) * T
            ) / (sigma * math.sqrt(T))
            delta = norm.cdf(d1)
            return delta if 0 < delta < 1 else None
        except Exception:
            return None

    @staticmethod
    def _target_profile(candidate: Dict) -> Dict[str, float]:
        """Tune target contract shape to overall setup quality."""
        ml = normalize_api_scalar(candidate.get("ml_ensemble_score"))
        panel = normalize_api_scalar(candidate.get("panel_composite_score"))
        ret_20d = normalize_api_scalar(candidate.get("return_20d"))
        rsi = normalize_api_scalar(candidate.get("rsi_14"))

        ml = ml if not is_missing_value(ml) else 0.5
        panel = (panel / 100.0) if not is_missing_value(panel) else 0.60
        ret_20d = ret_20d if not is_missing_value(ret_20d) else 0.0
        rsi = rsi if not is_missing_value(rsi) else 55.0

        setup_strength = clamp(
            0.50 * ml + 0.35 * panel + 0.15 * clamp(ret_20d / 0.20, 0.0, 1.0),
            0.0,
            1.0,
        )

        if setup_strength >= 0.78 and 48 <= rsi <= 67 and ret_20d > 0.06:
            return {
                "target_delta": 0.50,
                "target_dte": 30,
                "target_moneyness": 0.01,
                "target_premium_pct": 0.08,
                "max_break_even_pct": 0.10,
            }
        if setup_strength >= 0.65:
            return {
                "target_delta": 0.55,
                "target_dte": 42,
                "target_moneyness": -0.02,
                "target_premium_pct": 0.10,
                "max_break_even_pct": 0.12,
            }
        return {
            "target_delta": 0.60,
            "target_dte": 52,
            "target_moneyness": -0.05,
            "target_premium_pct": 0.12,
            "max_break_even_pct": 0.15,
        }

    @staticmethod
    def _score_contract(
        profile: Dict[str, float],
        underlying_price: float,
        strike: float,
        dte: int,
        delta: float,
        mid: float,
        spread_pct: float,
        oi: float,
        vol: float,
        iv: float,
        theta: Optional[float],
        break_even: Optional[float],
    ) -> float:
        """Return a 0-100 contract quality score for long calls."""
        if not underlying_price or underlying_price <= 0 or not strike or strike <= 0:
            return -1.0

        moneyness_pct = safe_div(strike - underlying_price, underlying_price, default=np.nan)
        premium_pct = safe_div(mid, underlying_price, default=np.nan)
        break_even_pct = (
            safe_div(break_even - underlying_price, underlying_price, default=np.nan)
            if break_even else np.nan
        )

        delta_fit = 1.0 - clamp(abs(delta - profile["target_delta"]) / 0.18, 0.0, 1.0)
        dte_fit = 1.0 - clamp(abs(dte - profile["target_dte"]) / 28.0, 0.0, 1.0)
        moneyness_fit = 1.0 - clamp(
            abs(moneyness_pct - profile["target_moneyness"]) / 0.12,
            0.0,
            1.0,
        )
        spread_score = 1.0 - clamp(
            spread_pct / OptionsEvaluator.MAX_SPREAD_PCT,
            0.0,
            1.0,
        )
        liquidity_score = (
            0.5 * clamp(np.log10(max(oi, 0) + 1) / 3.0, 0.0, 1.0) +
            0.5 * clamp(np.log10(max(vol, 0) + 1) / 2.5, 0.0, 1.0)
        )

        premium_score = 0.55
        if not pd.isna(premium_pct):
            premium_score = 1.0 - clamp(
                max(premium_pct - profile["target_premium_pct"], 0.0) / 0.12,
                0.0,
                1.0,
            )

        break_even_score = 0.50
        if not pd.isna(break_even_pct):
            break_even_score = 1.0 - clamp(
                max(break_even_pct - profile["max_break_even_pct"], 0.0) / 0.15,
                0.0,
                1.0,
            )

        iv_score = 1.0 - clamp(max(iv - 0.60, 0.0) / 0.90, 0.0, 1.0)

        theta_score = 0.55
        if not is_missing_value(theta):
            theta_score = 1.0 - clamp(abs(theta) / 0.15, 0.0, 1.0)

        raw_score = (
            0.22 * delta_fit +
            0.18 * dte_fit +
            0.14 * moneyness_fit +
            0.14 * spread_score +
            0.12 * liquidity_score +
            0.08 * premium_score +
            0.06 * break_even_score +
            0.04 * iv_score +
            0.02 * theta_score
        )
        return 100.0 * clamp(raw_score, 0.0, 1.0)

    #: Flags that describe opportunity rather than risk -- they must not be
    #: subtracted from the confidence score.
    INFORMATIONAL_FLAGS = frozenset({
        "HIGH_CROWD_INTEREST",
        "INSIDER_BUYING",
        "HIGH_SHORT_INTEREST",
    })

    @staticmethod
    def _trade_setup_score(candidate: Dict) -> float:
        """
        Blend stock, macro-panel, ML, hard-rule breadth, and option quality.

        Strategy bias: reward live insider buying and genuine crowd/momentum
        interest, but penalise names that have already gone vertical so the
        book is not built on runners that are peaking.
        """
        ml = normalize_api_scalar(candidate.get("ml_ensemble_score"))
        panel = normalize_api_scalar(candidate.get("panel_composite_score"))
        option_score = normalize_api_scalar(candidate.get("option_score"))
        rules_passed = normalize_api_scalar(candidate.get("rules_passed"))
        ret_20d = normalize_api_scalar(candidate.get("return_20d"))
        close_vs_sma50 = normalize_api_scalar(candidate.get("close_vs_sma50"))
        ema20_vs_ema50 = normalize_api_scalar(candidate.get("ema20_vs_ema50"))
        avg_dollar_volume = normalize_api_scalar(candidate.get("avg_dollar_volume"))
        hype = normalize_api_scalar(candidate.get("hype_score"))
        insider = normalize_api_scalar(candidate.get("insider_score"))
        squeeze = normalize_api_scalar(candidate.get("squeeze_score"))
        exhaustion = normalize_api_scalar(candidate.get("exhaustion_score"))

        ml = ml if not is_missing_value(ml) else 0.5
        panel = (panel / 100.0) if not is_missing_value(panel) else 0.60
        rule_component = (
            clamp(rules_passed / 10.0, 0.0, 1.0)
            if not is_missing_value(rules_passed) else 0.0
        )
        momentum_component = np.mean([
            clamp((ret_20d if not is_missing_value(ret_20d) else 0.0) / 0.15, 0.0, 1.0),
            clamp((close_vs_sma50 if not is_missing_value(close_vs_sma50) else 0.0) / 0.12, 0.0, 1.0),
            clamp((ema20_vs_ema50 if not is_missing_value(ema20_vs_ema50) else 0.0) / 0.08, 0.0, 1.0),
        ])
        liquidity_component = (
            clamp((math.log10(max(avg_dollar_volume, 1.0)) - 6.0) / 3.0, 0.0, 1.0)
            if not is_missing_value(avg_dollar_volume) else 0.0
        )
        option_component = (
            (option_score / 100.0)
            if candidate.get("option_candidate") == "Y" and not is_missing_value(option_score)
            else 0.0
        )

        hype_component = (
            clamp(hype / 100.0, 0.0, 1.0)
            if not is_missing_value(hype) else MomentumQuality.NEUTRAL / 100.0
        )
        squeeze_component = (
            clamp(squeeze / 100.0, 0.0, 1.0)
            if not is_missing_value(squeeze) else MomentumQuality.NEUTRAL / 100.0
        )
        # Insider conviction dominates, with short-interest fuel as a kicker.
        insider_component = (
            clamp(insider / 100.0, 0.0, 1.0)
            if not is_missing_value(insider) else MomentumQuality.NEUTRAL / 100.0
        )
        conviction_component = 0.75 * insider_component + 0.25 * squeeze_component

        # Anti-chase: nothing is subtracted at or below the neutral reading,
        # then the penalty ramps up to 18 points for fully parabolic names.
        exhaustion_penalty = 0.0
        if not is_missing_value(exhaustion):
            exhaustion_penalty = 0.18 * clamp(
                (exhaustion - MomentumQuality.NEUTRAL) / (100.0 - MomentumQuality.NEUTRAL),
                0.0,
                1.0,
            )

        flags = candidate.get("flags", [])
        if isinstance(flags, str):
            flags = [f for f in flags.split(", ") if f]
        risk_flags = [
            f for f in flags
            if f not in OptionsEvaluator.INFORMATIONAL_FLAGS
        ]
        flag_penalty = 0.015 * len(risk_flags)

        score = clamp(
            0.24 * panel +
            0.24 * ml +
            0.14 * rule_component +
            0.08 * momentum_component +
            0.10 * hype_component +
            0.10 * conviction_component +
            0.04 * liquidity_component +
            0.06 * option_component -
            exhaustion_penalty -
            flag_penalty,
            0.0,
            1.0,
        )
        confidence = 100.0 * score
        candidate["overall_confidence_score"] = round(confidence, 1)
        candidate["holding_horizon_days"] = HOLDING_HORIZON_DAYS
        return confidence


# ═══════════════════════════════════════════════════════════════════════════════
# HARD SELL/EXIT RULES -- Position Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

class SellMonitor:
    """
    Checks existing positions against hard sell rules.
    ANY single trigger = immediate exit signal.
    """

    @staticmethod
    def check_exits(
        positions: List[Dict], all_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        For each position, check all sell rules.
        Returns list of exit signals.
        """
        exits = []
        for pos in positions:
            ticker = pos.get("ticker")
            entry_date = pos.get("entry_date")
            entry_price = pos.get("entry_price", 0)

            df = all_data.get(ticker)
            if df is None:
                exits.append({"ticker": ticker, "reason": "No data available"})
                continue

            last = df.iloc[-1]
            close = last["Close"]
            ema20 = last.get("EMA_20", np.nan)
            ema10 = last.get("EMA_10", np.nan)
            rsi = last.get("RSI_14", np.nan)

            # SELL_01: EMA Breakdown -- Close < EMA(20)
            if not pd.isna(ema20) and close < ema20:
                exits.append({
                    "ticker": ticker,
                    "reason": "SELL_01: Close < EMA(20)",
                    "price": close,
                })
                continue

            # SELL_02: RSI Deterioration -- RSI < 45
            if not pd.isna(rsi) and rsi < 45:
                exits.append({
                    "ticker": ticker,
                    "reason": f"SELL_02: RSI={rsi:.1f} < 45",
                    "price": close,
                })
                continue

            # SELL_03: Time Stop -- > 20 trading days
            if entry_date:
                try:
                    ed = pd.Timestamp(entry_date)
                    days_held = len(df.loc[ed:]) - 1
                    if days_held > 20:
                        exits.append({
                            "ticker": ticker,
                            "reason": f"SELL_03: Held {days_held} days (> 20)",
                            "price": close,
                        })
                        continue
                except Exception:
                    pass

            # SELL_04: Trailing Profit Lock
            if entry_price > 0:
                gain = (close - entry_price) / entry_price
                if gain >= 0.10 and not pd.isna(ema10) and close < ema10:
                    exits.append({
                        "ticker": ticker,
                        "reason": f"SELL_04: +{gain:.1%} gain, Close < EMA(10)",
                        "price": close,
                    })

        return exits


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

class OutputFormatter:
    """Formats and saves the final ranked output."""

    @staticmethod
    def format_and_save(
        survivors: List[Dict],
        stage_counts: Dict[str, int],
        ml_params: Dict,
        feature_importances: Dict,
        macro: Optional["MacroRegime"] = None,
    ) -> pd.DataFrame:
        """
        Create the final ranked table, save to CSV and JSON.
        """
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not survivors:
            log.warning("No survivors to output!")
            return pd.DataFrame()

        # Sort by best call-trade quality first, then underlying quality.
        survivors_sorted = sorted(
            survivors,
            key=lambda x: (
                x.get("overall_confidence_score", x.get("trade_setup_score", 0)),
                x.get("hard_buy_pass", False),
                1 if x.get("option_candidate") == "Y" else 0,
                x.get("panel_composite_score", 0),
                x.get("ml_ensemble_score", 0),
                x.get("rules_passed", 0),
                x.get("option_score", 0),
            ),
            reverse=True,
        )

        # Assign ranks
        for i, s in enumerate(survivors_sorted):
            s["rank"] = i + 1

        # Build DataFrame
        columns = [
            "rank", "ticker", "name", "sector", "price", "market_cap",
            "rsi_14", "macd_histogram", "volume_ratio",
            "stop_loss", "target_price", "risk_per_share",
            "reward_risk_ratio", "pct_equity_risk", "shares_per_10k_risk",
            "ml_score_xgb", "ml_score_rf", "ml_ensemble_score", "lstm_score",
            "panel_composite_score", "panel_consensus", "trade_setup_score",
            "overall_confidence_score", "rules_passed", "rules_failed",
            "hard_buy_pass", "failed_rules",
            "panel_livermore", "panel_druckenmiller", "panel_lynch",
            "panel_minervini", "panel_oneil",
            "hype_score", "exhaustion_score", "insider_score", "squeeze_score",
            "insider_net_purchase_pct", "insider_buy_transactions",
            "insider_sell_transactions", "short_pct_float",
            "rvol_5", "stretch_atr", "pct_from_52w_high",
            "holding_horizon_days",
            "recommended_action",
            "option_candidate", "option_strike", "option_expiry",
            "option_delta", "option_dte", "option_bid_ask_spread",
            "option_mid", "option_break_even", "option_break_even_pct",
            "option_iv", "option_theta", "option_moneyness_pct",
            "option_score", "option_source",
            "flags",
        ]

        rows = []
        for s in survivors_sorted:
            # Determine recommended action
            ml = s.get("ml_ensemble_score", 0)
            panel = s.get("panel_composite_score", 0)
            trade = s.get("trade_setup_score", 0)
            option_score = s.get("option_score", 0)
            if s.get("option_candidate") == "Y" and trade >= 78 and option_score >= 70:
                action = "BEST CALL"
            elif s.get("option_candidate") == "Y" and trade >= 68 and option_score >= 60:
                action = "CALL BUY"
            elif trade >= 75 and ml >= 0.60 and panel >= 70:
                action = "STRONG BUY"
            elif trade >= 62 and panel >= 60:
                action = "BUY"
            else:
                action = "WATCH"

            s["recommended_action"] = action
            s["flags"] = ", ".join(s.get("flags", []))
            if isinstance(s.get("failed_rules"), list):
                s["failed_rules"] = ", ".join(s.get("failed_rules", []))
            s.setdefault("holding_horizon_days", HOLDING_HORIZON_DAYS)

            row = {col: s.get(col) for col in columns}
            rows.append(row)

        df = pd.DataFrame(rows, columns=columns)

        # Format numeric columns
        numeric_fmt = {
            "price": "{:.2f}",
            "rsi_14": "{:.1f}",
            "macd_histogram": "{:.4f}",
            "volume_ratio": "{:.2f}",
            "ml_score_xgb": "{:.4f}",
            "ml_score_rf": "{:.4f}",
            "ml_ensemble_score": "{:.4f}",
            "panel_composite_score": "{:.1f}",
            "trade_setup_score": "{:.1f}",
            "overall_confidence_score": "{:.1f}",
            "option_score": "{:.1f}",
            "option_mid": "{:.2f}",
            "option_break_even": "{:.2f}",
            "option_break_even_pct": "{:.1f}",
            "option_iv": "{:.3f}",
            "option_theta": "{:.4f}",
            "option_moneyness_pct": "{:.1f}",
        }

        # Save CSV
        csv_path = OUTPUT_DIR / f"scan_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Results saved to {csv_path}")

        # Save JSON report
        report = {
            "engine": ENGINE_NAME,
            "engine_version": ENGINE_VERSION,
            "scan_timestamp": now_et(),
            "strategy": {
                "objective": "maximum risk-adjusted profit over a short-to-mid-term hold",
                "holding_horizon_days": HOLDING_HORIZON_DAYS,
                "ml_target_return": ML_TARGET_RETURN,
                "option_dte_window": [OptionsEvaluator.MIN_DTE, OptionsEvaluator.MAX_DTE],
                "follows": ["insider buying", "short interest fuel", "crowd/volume momentum"],
                "avoids": [
                    f"extension > {MAX_EXTENSION_ABOVE_SMA50:.0%} above the 50DMA",
                    f"blow-off RSI > {MAX_EXHAUSTION_RSI:.0f}",
                    f"5-day vertical spikes > {MAX_SPIKE_RETURN_5D:.0%}",
                    f"20-day parabolic runs > {MAX_SPIKE_RETURN_20D:.0%}",
                ],
            },
            "attestation": (
                "This scan used live data only. The equity universe is discovered "
                "fresh from the exchange listing each run. Prior scan_results are "
                "never read as input. No hardcoded tickers, no watchlists, no "
                "preset baskets, no fabricated values, no demo data. Relative "
                "strength uses the live S&P 500 index, not a pre-selected ETF. "
                "Live headlines, US market status, VIX/energy/gold gauges, and "
                "the near-term earnings calendar contextualize regime and "
                "annotate survivors; they do not select names. Intraday partial "
                "bars are trimmed during RTH."
            ),
            "data_sources": {
                "ohlcv": dict(DATA_SOURCE_USAGE["ohlcv"]),
                "fundamentals": dict(DATA_SOURCE_USAGE["fundamentals"]),
                "options": dict(DATA_SOURCE_USAGE["options"]),
            },
            "macro_regime": macro.to_dict() if macro is not None else None,
            "stage_counts": stage_counts,
            "ml_hyperparameters": ml_params,
            "feature_importances": {
                k: round(v, 4) for k, v in sorted(
                    feature_importances.items(), key=lambda x: -x[1]
                )
            },
            "total_survivors": len(survivors_sorted),
            "top_25": [
                {
                    "rank": s["rank"],
                    "ticker": s["ticker"],
                    "name": s.get("name", ""),
                    "ml_ensemble": round(s.get("ml_ensemble_score", 0), 4),
                    "panel_composite": round(s.get("panel_composite_score", 0), 1),
                    "trade_setup_score": round(s.get("trade_setup_score", 0), 1),
                    "overall_confidence": round(s.get("overall_confidence_score", 0), 1),
                    "hype_score": s.get("hype_score"),
                    "exhaustion_score": s.get("exhaustion_score"),
                    "insider_score": s.get("insider_score"),
                    "squeeze_score": s.get("squeeze_score"),
                    "holding_horizon_days": s.get("holding_horizon_days", HOLDING_HORIZON_DAYS),
                    "rules_passed": s.get("rules_passed"),
                    "hard_buy_pass": s.get("hard_buy_pass"),
                    "option_score": round(s.get("option_score", 0), 1),
                    "action": s.get("recommended_action", ""),
                    "live_earnings_date": s.get("live_earnings_date"),
                    "live_headlines": s.get("live_headlines") or [],
                }
                for s in survivors_sorted[:25]
            ],
        }
        json_path = OUTPUT_DIR / f"scan_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Report saved to {json_path}")

        # Console output
        OutputFormatter._print_results(df, stage_counts, report)

        return df

    @staticmethod
    def _print_results(df: pd.DataFrame, stage_counts: Dict, report: Dict):
        """Pretty-print results to console."""
        print("\n" + "=" * 100)
        print(f"  STOCK UNIVERSE SCAN -- FINAL RESULTS")
        print(f"  Engine: {report.get('engine', ENGINE_NAME)} v{report.get('engine_version', ENGINE_VERSION)}")
        print(f"  Scan Time: {report['scan_timestamp']}")
        print("=" * 100)

        # Macro regime context
        macro_dict = report.get("macro_regime") or {}
        if macro_dict:
            print("\n  ┌─ MACRO REGIME ────────────────────────────────────┐")
            print(f"  │  {('Regime: ' + str(macro_dict.get('regime_label', ''))) :<50} │")
            print(f"  │  Score: {macro_dict.get('regime_score', '')}/100" + " " * 38 + "│")
            for n in macro_dict.get("notes", []):
                txt = str(n)[:48]
                print(f"  │  - {txt:<48} │")
            print(f"  └──────────────────────────────────────────────────┘")

        world = macro_dict.get("world_context") or report.get("world_context") or {}
        headlines = world.get("headlines") or []
        if headlines or world.get("event_risk") is not None:
            print("\n  ┌─ LIVE WORLD CONTEXT (does not pick tickers) ─────┐")
            print(
                f"  │  Event risk: {world.get('event_risk', '--')}/100"
                + " " * 32 + "│"
            )
            for item in headlines[:5]:
                txt = str(item.get("headline") or item)[:48]
                print(f"  │  - {txt:<48} │")
            print(f"  └──────────────────────────────────────────────────┘")

        # Pipeline funnel
        print("\n  ┌─ PIPELINE FUNNEL ─────────────────────────────────┐")
        for stage, count in stage_counts.items():
            print(f"  │  {stage:<40} {count:>6} │")
        print(f"  └──────────────────────────────────────────────────┘")

        # Top 25 ranked table
        print(f"\n  TOP {min(25, len(df))} CANDIDATES:")
        print("  " + "-" * 110)
        header = (
            f"  {'#':>3} {'Ticker':<7} {'Name':<22} {'Price':>8} "
            f"{'Conf':>6} {'Rules':>5} {'ML':>6} {'Panel':>6} "
            f"{'Hype':>5} {'Exh':>5} {'Insdr':>6} {'Action':<12} {'Opt':>3}"
        )
        print(header)
        print("  " + "-" * 110)

        for _, row in df.head(25).iterrows():
            mc = row.get("market_cap")
            mc_str = ""
            if pd.notna(mc) and mc > 0:
                if mc >= 1e12:
                    mc_str = f"${mc/1e12:.1f}T"
                elif mc >= 1e9:
                    mc_str = f"${mc/1e9:.1f}B"
                elif mc >= 1e6:
                    mc_str = f"${mc/1e6:.0f}M"

            name = str(row.get("name", ""))[:21]
            trade = row.get("trade_setup_score", 0)
            conf = row.get("overall_confidence_score", trade)
            rules = row.get("rules_passed", 0)
            ml = row.get("ml_ensemble_score", 0)
            panel = row.get("panel_composite_score", 0)
            action = row.get("recommended_action", "")
            opt = row.get("option_candidate", "N")
            hype = row.get("hype_score")
            exh = row.get("exhaustion_score")
            insdr = row.get("insider_score")

            def _sig(value) -> str:
                return "  --" if is_missing_value(value) else f"{float(value):4.0f}"

            print(
                f"  {int(row['rank']):>3} {row['ticker']:<7} {name:<22} "
                f"${float(row['price']):>7.2f} "
                f"{float(conf):>5.1f} {int(rules):>2}/10 "
                f"{float(ml):>5.3f} {float(panel):>5.1f} "
                f"{_sig(hype):>5} {_sig(exh):>5} {_sig(insdr):>6} "
                f"{action:<12} {opt:>3}"
            )

        # Flags summary
        flagged = df[df["flags"].str.len() > 0]
        if not flagged.empty:
            print(f"\n  FLAGS:")
            for _, row in flagged.iterrows():
                print(f"    {row['ticker']}: {row['flags']}")

        # Options detail
        opts = df[df["option_candidate"] == "Y"]
        if not opts.empty:
            print(f"\n  QUALIFYING OPTIONS ({len(opts)} contracts):")
            print(f"  {'Ticker':<7} {'Strike':>8} {'Expiry':<12} "
                  f"{'Delta':>6} {'DTE':>4} {'Spread%':>8} {'OptSc':>6}")
            for _, row in opts.iterrows():
                print(
                    f"  {row['ticker']:<7} ${float(row.get('option_strike', 0)):>7.2f} "
                    f"{row.get('option_expiry', ''):>12} "
                    f"{float(row.get('option_delta', 0)):>5.3f} "
                    f"{int(row.get('option_dte', 0)):>4} "
                    f"{float(row.get('option_bid_ask_spread', 0)):>7.1f}% "
                    f"{float(row.get('option_score', 0)):>5.1f}"
                )

        print("\n" + "=" * 100)
        print(f"  ATTESTATION: {report['attestation']}")
        print("=" * 100 + "\n")

    @staticmethod
    def save_near_misses(
        near_misses: List[Dict], stage_counts: Dict[str, int]
    ) -> Optional[pd.DataFrame]:
        """
        Save and display the near-miss report when zero survivors emerge.
        Outputs top 3 tickers closest to passing all 10 hard buy rules.
        """
        if not near_misses:
            log.info("No near-miss candidates to report.")
            return None

        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build DataFrame
        rows = []
        for nm in near_misses:
            rows.append({
                "near_miss_rank": nm["near_miss_rank"],
                "ticker": nm["ticker"],
                "price": nm["price"],
                "rules_passed": nm["rules_passed"],
                "rules_failed": nm["rules_failed"],
                "passed_rules": ", ".join(nm["passed_rules"]),
                "failed_rules": ", ".join(nm["failed_rules"]),
                "rsi_14": nm.get("rsi_14"),
                "macd_histogram": nm.get("macd_histogram"),
                "volume_ratio": nm.get("volume_ratio"),
                "return_20d": nm.get("return_20d"),
            })

        df = pd.DataFrame(rows)

        # Save CSV
        csv_path = OUTPUT_DIR / f"near_misses_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Near-miss report saved to {csv_path}")

        # Save JSON
        json_path = OUTPUT_DIR / f"near_misses_{timestamp}.json"
        report = {
            "scan_timestamp": now_et(),
            "reason": "Zero survivors -- hard buy rules too tight for current market conditions",
            "stage_counts": stage_counts,
            "near_misses": [
                {
                    "rank": nm["near_miss_rank"],
                    "ticker": nm["ticker"],
                    "price": nm["price"],
                    "rules_passed": nm["rules_passed"],
                    "passed": nm["passed_rules"],
                    "failed": nm["failed_rules"],
                }
                for nm in near_misses
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # Console output
        print("\n" + "=" * 100)
        print("  ZERO SURVIVORS -- NEAR-MISS REPORT")
        print(f"  Scan Time: {now_et()}")
        print("  The market did not produce any tickers passing all 10 hard buy rules.")
        print("  Below are the 3 closest contenders, ranked by rules passed.")
        print("=" * 100)

        print(f"\n  {'#':>3} {'Ticker':<7} {'Price':>8} {'Passed':>7} "
              f"{'Failed Rules':<55}")
        print("  " + "-" * 85)

        for nm in near_misses:
            failed_str = ", ".join(nm["failed_rules"])
            print(
                f"  {nm['near_miss_rank']:>3} {nm['ticker']:<7} "
                f"${nm['price']:>7.2f} "
                f"{nm['rules_passed']:>3}/10  "
                f"{failed_str:<55}"
            )

        # Detail breakdown
        print("\n  DETAILED BREAKDOWN:")
        rule_names = {
            "BUY_01": "Trend Filter (Close > SMA200, SMA50 > SMA200)",
            "BUY_02": "50-Day MA Bullish Slope",
            "BUY_03": "Bollinger Breakout / 20-Day High",
            "BUY_04": "MACD Bullish (line > signal, histogram > 0)",
            "BUY_05": "SMA(10)/SMA(30) Crossover within 5 sessions",
            "BUY_06": "VWAP Confirmation",
            "BUY_07": "EMA Stack (EMA20 > EMA50)",
            "BUY_08": "RSI Sweet Spot (40-70)",
            "BUY_09": "Volume Surge (>= 1.25x avg)",
            "BUY_10": "No Penny Stocks (>= $5)",
        }
        for nm in near_misses:
            print(f"\n  {nm['ticker']} ({nm['rules_passed']}/10):")
            for rule_id in [f"BUY_{i:02d}" for i in range(1, 11)]:
                passed = rule_id in nm["passed_rules"]
                mark = "[PASS]" if passed else "[FAIL]"
                name = rule_names.get(rule_id, rule_id)
                print(f"    {mark} {rule_id}: {name}")

            # Show key metrics
            rsi = nm.get("rsi_14")
            macd = nm.get("macd_histogram")
            vr = nm.get("volume_ratio")
            ret = nm.get("return_20d")
            metrics = []
            if rsi is not None:
                metrics.append(f"RSI={rsi:.1f}")
            if macd is not None:
                metrics.append(f"MACD_hist={macd:.4f}")
            if not pd.isna(vr):
                metrics.append(f"VolRatio={vr:.2f}")
            if not pd.isna(ret):
                metrics.append(f"20d_ret={ret:.2%}")
            if metrics:
                print(f"    Metrics: {' | '.join(metrics)}")

        print("\n" + "=" * 100)
        print(f"  Near-miss CSV: {csv_path}")
        print(f"  Near-miss JSON: {json_path}")
        print("=" * 100 + "\n")

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Execute the full 6-stage scanning pipeline."""
    pipeline_start = time.time()
    clock = PipelineClock(PIPELINE_BUDGET_MINUTES)
    log.info("=" * 70)
    log.info("  STOCK UNIVERSE SCAN PIPELINE -- STARTING")
    log.info(f"  Engine: {ENGINE_NAME} v{ENGINE_VERSION}")
    log.info(f"  Time: {now_et()}")
    log.info(f"  Market Open Now (ET RTH): {is_market_open_now()}")
    log.info(f"  Wall-clock budget: {PIPELINE_BUDGET_MINUTES:.0f} min")
    log.info(
        f"  Strategy: {HOLDING_HORIZON_DAYS}-session horizon, target "
        f"+{ML_TARGET_RETURN:.0%}, insider/hype weighted, anti-chase guard on"
    )
    log.info("=" * 70)

    stage_counts = {}
    ml_params = {}
    feature_importances = {}
    near_misses: List[Dict] = []

    try:
        reset_market_router()
        verify_api_credentials()
        log.info(
            "Fresh run: live universe discovery only -- prior scan_results "
            "are never read, and no ticker basket is preloaded."
        )

        # ── STAGE 0: Macro Regime Snapshot ───────────────────────────
        # Establishes geopolitical/macro context BEFORE any equity work
        # (the panel review's #1 demand: "consider current context").
        macro = MacroRegime()
        macro.load()
        stage_counts["Stage 0: Macro regime"] = round(macro.regime_score)

        # ── STAGE 1: Universe Discovery ──────────────────────────────
        clock.check("Stage 1: Universe Discovery")
        discovery = UniverseDiscovery(MASSIVE_API_KEY)
        tickers = discovery.discover()
        stage_counts["Stage 1: Universe Discovered"] = len(tickers)

        # ── DATA FETCH: OHLCV via MBOUM primary + live fallbacks ──────
        clock.check("Data Fetch")
        fetcher = DataFetcher()
        all_data = fetcher.fetch_ohlcv(tickers)
        stage_counts["Data Fetch: Tickers with OHLCV"] = len(all_data)

        # ── COMPUTE TECHNICALS ───────────────────────────────────────
        log.info("Computing technical indicators for all tickers...")
        for ticker in list(all_data.keys()):
            try:
                all_data[ticker] = TechnicalEngine.compute_all(all_data[ticker])
            except Exception as e:
                log.debug(f"  Tech calc failed for {ticker}: {e}")
                del all_data[ticker]
        stage_counts["Technicals Computed"] = len(all_data)

        # ── STAGE 2: Execution Guards ────────────────────────────────
        clock.check("Stage 2: Execution Guards")
        guarded_data, guard_rejected = ExecutionGuards.apply(all_data)
        stage_counts["Stage 2: Passed Guards"] = len(guarded_data)

        if len(guarded_data) == 0:
            raise PipelineError("No tickers survived execution guards. Pipeline STOPPED.")

        # ── STAGE 3: Hard Buy Rules ──────────────────────────────────
        clock.check("Stage 3: Hard Buy Rules")
        strict_survivors, buy_rejected = HardBuyRules.apply(guarded_data)
        stage_counts["Stage 3: Strict Hard Buy Rules"] = len(strict_survivors)

        if len(strict_survivors) < TARGET_FINAL_CANDIDATES:
            log.warning(
                f"Only {len(strict_survivors)} tickers passed all 10 hard buy rules; "
                "backfilling from the strongest live near-misses to produce a top-7 ranking."
            )
            survivors, near_misses = HardBuyRules.build_rank_pool(
                guarded_data,
                strict_survivors,
                target_size=TARGET_FINAL_CANDIDATES,
                max_pool_size=MAX_PANEL_CANDIDATES,
            )
        else:
            survivors = strict_survivors
            near_misses = []
        stage_counts["Stage 3B: Ranked Candidate Pool"] = len(survivors)

        if len(survivors) == 0:
            # The near-miss report is the only diagnostic artifact when the
            # rule set is too tight for the current regime -- always write it.
            OutputFormatter.save_near_misses(near_misses, stage_counts)
            raise PipelineError("No rankable candidates after hard-rule scoring. Pipeline STOPPED.")

        # ── STAGE 4: ML Ranking ──────────────────────────────────────
        # Train on a broad price/liquidity-screened universe (~50x more
        # samples than survivors-only) for true discriminative power.
        clock.check("Stage 4: ML Ranking")
        ranker = MLRanker()
        # Fit on the liquidity/price-screened universe, not the guarded set:
        # the guards condition on a name's latest 63-day return, which is
        # look-ahead relative to the historical rows being labelled.
        ml_training_pool = ExecutionGuards.ml_training_pool(all_data)
        log.info(
            f"  ML training universe: {len(ml_training_pool)} tickers "
            f"(price/liquidity screen on {len(all_data)}), scoring "
            f"{len(survivors)} guarded candidates."
        )
        survivors = ranker.rank(survivors, all_data, training_universe=ml_training_pool)
        ml_params = {
            "XGBoost": "n_estimators=200, max_depth=6, lr=0.05, subsample=0.8",
            "RandomForest": "n_estimators=200, max_depth=8, min_samples_leaf=20",
            "LSTM": "Available (PyTorch)" if LSTM_AVAILABLE else "Not installed",
            "XGB_available": XGB_AVAILABLE,
            "training_universe_size": len(ml_training_pool),
            "training_universe_screen": "price >= $5 and 20d avg $vol >= $5M",
            "degraded_models": list(ranker.degraded_models),
        }
        if ranker.degraded_models:
            log.warning(
                "  ML ensemble degraded this run -- "
                f"{', '.join(ranker.degraded_models)} scored 0.5 for every "
                "survivor. Ranking leans on the hard rules and panel."
            )
        feature_importances = ranker.feature_importances

        # ── FUNDAMENTALS FETCH (survivors only) ──────────────────────
        clock.check("Fundamentals + insider fetch")
        survivor_tickers = [s["ticker"] for s in survivors]
        fund_fetcher = FundamentalsFetcher(MASSIVE_API_KEY)
        fundamentals = fund_fetcher.fetch_batch(survivor_tickers)

        # ── STAGE 5: 5-Investor Panel ────────────────────────────────
        clock.check("Stage 5: Investor Panel")
        panel = InvestorPanel(macro=macro)
        panel.set_universe_returns(guarded_data)  # IBD-style RS percentile
        survivors = panel.score_all(
            survivors, all_data, fundamentals, apply_filter=False
        )
        panel_qualified = [
            s for s in survivors
            if s.get("panel_composite_score", 0) >= 60 and s.get("panel_consensus", 0) >= 3
        ]
        stage_counts["Stage 5: Panel Scored"] = len(survivors)
        stage_counts["Stage 5: Panel Qualified"] = len(panel_qualified)

        if len(survivors) == 0:
            log.warning("No tickers could be panel-scored.")
            OutputFormatter.save_near_misses(near_misses, stage_counts)
            OutputFormatter.format_and_save([], stage_counts, ml_params, feature_importances, macro=macro)
            return

        for s in survivors:
            s["trade_setup_score"] = round(OptionsEvaluator._trade_setup_score(s), 1)

        pre_option_pool = sorted(
            panel_qualified or survivors,
            key=lambda x: (
                x.get("trade_setup_score", 0),
                x.get("hard_buy_pass", False),
                x.get("panel_composite_score", 0),
                x.get("ml_ensemble_score", 0),
                x.get("rules_passed", 0),
            ),
            reverse=True,
        )[:MAX_OPTIONS_EVAL_CANDIDATES]
        if len(pre_option_pool) < TARGET_FINAL_CANDIDATES and len(survivors) >= TARGET_FINAL_CANDIDATES:
            seen_pre = {s["ticker"] for s in pre_option_pool}
            for s in sorted(
                survivors,
                key=lambda x: (
                    x.get("trade_setup_score", 0),
                    x.get("panel_composite_score", 0),
                    x.get("rules_passed", 0),
                ),
                reverse=True,
            ):
                if s["ticker"] not in seen_pre:
                    pre_option_pool.append(s)
                    seen_pre.add(s["ticker"])
                if len(pre_option_pool) >= TARGET_FINAL_CANDIDATES:
                    break

        # ── STAGE 6: Options Evaluation ──────────────────────────────
        clock.check("Stage 6: Options Evaluation")
        survivors = OptionsEvaluator.evaluate(pre_option_pool, fundamentals=fundamentals)
        stage_counts["Stage 6: Options Evaluated"] = len(survivors)

        # Enrich with risk-management fields (ATR-stop, target, sizing).
        for s in survivors:
            df = all_data.get(s["ticker"])
            if df is not None and len(df) > 0:
                last = df.iloc[-1]
                close = float(last.get("Close", 0.0)) or 0.0
                atr = float(last.get("ATR_14", 0.0)) or 0.0
                sma50 = float(last.get("SMA_50", 0.0)) or 0.0
                # Stop = max(SMA50, close - 2.5 * ATR) -- tighter of structural
                # support and volatility-based stop. Matches Minervini protocol.
                vol_stop = close - 2.5 * atr
                stop_loss = max(sma50, vol_stop, 0.0) if sma50 > 0 else vol_stop
                if stop_loss <= 0 or stop_loss >= close:
                    stop_loss = close * 0.92  # 8% fallback
                # Target: 2.5x risk (asymmetric R/R Druckenmiller-style)
                risk = max(close - stop_loss, 0.01)
                target = close + 2.5 * risk
                # 1% risk-of-equity sizing scaled by macro regime
                regime_scalar = macro.position_sizing_scalar()
                pct_of_equity = round(1.0 * regime_scalar, 2)  # % of portfolio risked
                shares_per_10k = math.floor(
                    (10000.0 * (pct_of_equity / 100.0)) / max(risk, 0.01)
                )
                s["stop_loss"] = round(stop_loss, 2)
                s["target_price"] = round(target, 2)
                s["risk_per_share"] = round(risk, 2)
                s["reward_risk_ratio"] = round((target - close) / risk, 2)
                s["pct_equity_risk"] = pct_of_equity
                s["shares_per_10k_risk"] = int(shares_per_10k)

        # ── PRE-OUTPUT VERIFICATION ──────────────────────────────────
        log.info("Running pre-output verification...")
        verified = []
        for s in survivors:
            ticker = s["ticker"]
            df = all_data.get(ticker)
            if df is None:
                continue

            # Re-verify critical values
            last = df.iloc[-1]
            rsi = s.get("rsi_14", np.nan)
            if not pd.isna(rsi) and not (0 <= rsi <= 100):
                log.warning(f"  {ticker}: RSI out of range ({rsi}), excluding")
                continue

            ml = s.get("ml_ensemble_score", np.nan)
            if not pd.isna(ml) and not (0 <= ml <= 1):
                log.warning(f"  {ticker}: ML score out of range ({ml}), excluding")
                continue

            panel_score = s.get("panel_composite_score", 0)
            if not (0 <= panel_score <= 100):
                log.warning(f"  {ticker}: Panel score out of range ({panel_score}), excluding")
                continue

            # Check for extreme single-day move
            ret_1d = last.get("Return_1d", 0)
            if not pd.isna(ret_1d) and abs(ret_1d) > 0.20:
                if "flags" not in s or not isinstance(s["flags"], list):
                    s["flags"] = s.get("flags", "").split(", ") if isinstance(s.get("flags"), str) else []
                s["flags"].append("EXTREME_VOLATILITY")

            # Check low float
            fund = fundamentals.get(ticker, {})
            float_shares = fund.get("shares_float", np.nan)
            if not is_missing_value(float_shares) and float_shares < 5_000_000:
                if isinstance(s.get("flags"), list):
                    s["flags"].append("LOW_FLOAT")
                elif isinstance(s.get("flags"), str):
                    s["flags"] = s["flags"] + ", LOW_FLOAT" if s["flags"] else "LOW_FLOAT"

            verified.append(s)

        # Remove duplicate tickers
        seen = set()
        final = []
        for s in verified:
            if s["ticker"] not in seen:
                seen.add(s["ticker"])
                final.append(s)

        final = sorted(
            final,
            key=lambda x: (
                x.get("overall_confidence_score", x.get("trade_setup_score", 0)),
                x.get("hard_buy_pass", False),
                x.get("panel_composite_score", 0),
                x.get("ml_ensemble_score", 0),
                x.get("rules_passed", 0),
            ),
            reverse=True,
        )[:TARGET_FINAL_CANDIDATES]

        stage_counts["Verified Final Output"] = len(final)
        if len(final) < TARGET_FINAL_CANDIDATES:
            log.warning(
                f"Only {len(final)} verified candidates available for top-{TARGET_FINAL_CANDIDATES} output."
            )

        try:
            macro.world.annotate(final)
        except Exception as exc:
            log.debug(f"  Live news annotation skipped: {exc}")

        # ── FORMAT AND SAVE OUTPUT ───────────────────────────────────
        result_df = OutputFormatter.format_and_save(
            final, stage_counts, ml_params, feature_importances, macro=macro
        )

        elapsed = time.time() - pipeline_start
        log.info(f"Pipeline completed in {elapsed:.1f} seconds")
        log.info(f"Final candidates: {len(final)}")
        log.info(f"Engine: {ENGINE_NAME} v{ENGINE_VERSION}")

    except PipelineBudgetExceeded as e:
        # Out of wall-clock time: emit whatever diagnostics exist rather than
        # letting the Actions runner kill the job with no artifacts at all.
        log.error(f"PIPELINE BUDGET EXCEEDED: {e}")
        try:
            OutputFormatter.save_near_misses(near_misses, stage_counts)
        except Exception as inner:
            log.error(f"  Could not save near-miss diagnostics: {inner}")
        print(f"\n*** PIPELINE BUDGET EXCEEDED ***\n{e}\n")
        sys.exit(1)

    except PipelineError as e:
        log.error(f"PIPELINE STOPPED: {e}")
        print(f"\n*** PIPELINE STOPPED ***\n{e}\n")
        sys.exit(1)

    except Exception as e:
        log.error(f"Unexpected error: {e}")
        log.error(traceback.format_exc())
        print(f"\n*** UNEXPECTED ERROR ***\n{e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
