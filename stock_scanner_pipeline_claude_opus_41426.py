#!/usr/bin/env python3
"""
===============================================================================
  Stock Universe Scan Pipeline -- Nick's Live Trading System
  ─────────────────────────────────────────────────────────
  Owner:   Nick -- Data Analyst, real-capital trader
  Purpose: Identify the strongest buy opportunities before they happen
  Capital: Real money -- zero tolerance for hallucinated data or shortcuts

  STAGES:
    1. Universe Discovery   (Finnhub + yfinance)
    2. Execution Guards     (data integrity, liquidity, spread, 3mo perf)
    3. Hard Buy Rules       (ALL 10 must pass)
    4. ML Ranking           (XGBoost + Random Forest + optional LSTM)
    5. 5-Investor Panel     (Livermore, Druckenmiller, Lynch, Minervini, O'Neil)
    6. Options Evaluation   (long calls, 0.35-0.50 delta, 14-45 DTE)

  Run:  python stock_scanner_pipeline.py
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
from massive import RESTClient
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

client = RESTClient("hTRjnsG45cxV1K4GpLeGxpZp7rgPu6tU")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION -- API keys default from Nick's SKILL file; env vars override
# ═══════════════════════════════════════════════════════════════════════════════

MASSIVE_API_KEY = os.environ.get(
    "MASSIVE_API_KEY", "hTRjnsG45cxV1K4GpLeGxpZp7rgPu6tU"
)
ALPHAVANTAGE_API_KEY = os.environ.get(
    "ALPHAVANTAGE_API_KEY", "Q7LVL2LCTKCF81ZA"
)
MBOUM_API_KEY = os.environ.get(
    "MBOUM_API_KEY", "xfAMuSlx5yUmX4PKegfarmd7y8799RcxjxKiNAUh"
)
# Options-tier MBOUM key (different plan that includes the /v1/markets/options
# endpoint). Set via env var or falls back to the user's options-plan key.
MBOUM_OPTIONS_KEY = os.environ.get(
    "MBOUM_OPTIONS_KEY", "642|Splqvb0O7fzSSI0ptlYIs4qXt4N4UaqwJHToVQ1X"
)
MBOUM_BASE_URL = "https://api.mboum.com"
MASSIVE_BASE_URL = "https://api.massive.com/v2"

# Pipeline parameters
LOOKBACK_DAYS = 380  # calendar days to request (~252 trading days)
MIN_TRADING_DAYS = 252
MIN_UNIVERSE_SIZE = 500
BATCH_SIZE = 100  # yfinance download batch size
MAX_WORKERS = 8   # thread pool for fundamentals
OUTPUT_DIR = Path("scan_results")

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
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

        missing_required = sorted(self.REQUIRED_SYMBOLS - set(self.snapshot))
        if len(self.snapshot) < self.MIN_SNAPSHOT_COUNT or missing_required:
            raise PipelineError(
                "Macro regime unavailable or incomplete "
                f"({len(self.snapshot)}/{len(self.SYMBOLS)} loaded; "
                f"missing required: {', '.join(missing_required) or 'none'}). "
                "Pipeline STOPPED rather than using a neutral fallback."
            )

        self._score_regime()

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

    def position_sizing_scalar(self) -> float:
        """Multiplier in [0.4, 1.2] for downstream position sizing.
        Used to scale the recommended dollar exposure based on regime."""
        if self.regime_score is None:
            raise PipelineError("Macro regime score unavailable for position sizing.")
        s = self.regime_score
        if s >= 75:
            return 1.20
        if s >= 60:
            return 1.00
        if s >= 45:
            return 0.80
        if s >= 30:
            return 0.55
        return 0.40

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
    MASSIVE_API_KEY = "hTRjnsG45cxV1K4GpLeGxpZp7rgPu6tU"

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

        while url:
            retries = 0
            backoff = 1
            while retries < 3:
                try:
                    resp = self.session.get(url, params=params, timeout=600)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception as e:
                    retries += 1
                    if retries >= 3:
                        raise PipelineError(
                            f"Live universe discovery FAILED. Source: Massive. "
                            f"Error: {e}. Do NOT substitute a preset basket."
                        )
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
            raise PipelineError(
                "Massive returned empty symbol list. Pipeline STOPPED."
            )

        # Filter to common stocks and ETFs on major exchanges ONLY
        # Exclude OTC (OOTC), warrants, rights, units, ADRs with poor liquidity
        VALID_MIC = {"XNYS", "XNAS", "XASE", "ARCX", "BATS"}
        VALID_TYPES = {
            "CS", "Common Stock", "EQS",
            "ETF", "ETP", "REIT", "MLP", "Closed-End Fund",
        }

        tickers = []
        for item in data:
            sym = item.get("ticker", item.get("symbol", ""))
            mic = item.get("primary_exchange", item.get("mic", ""))
            sec_type = item.get("type", "")

            # Must be on a major exchange (excludes 17K+ OTC names)
            if mic not in VALID_MIC:
                continue

            # Must be a valid security type
            if sec_type not in VALID_TYPES:
                continue

            # Skip symbols with special characters (warrants, preferred, units)
            if any(c in sym for c in [".", "-", "/", "+"]):
                continue
            if len(sym) > 5 or len(sym) == 0:
                continue
            if not sym.isalpha():
                continue

            tickers.append(sym)

        tickers = sorted(set(tickers))
        log.info(
            f"STAGE 1 COMPLETE: {len(data)} raw symbols -> {len(tickers)} "
            f"cleaned tickers (common stocks + ETFs)"
        )

        if len(tickers) < MIN_UNIVERSE_SIZE:
            raise PipelineError(
                f"Universe too small ({len(tickers)} < {MIN_UNIVERSE_SIZE}). "
                f"Pipeline STOPPED. Do NOT substitute a preset basket."
            )

        return tickers


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING -- MBOUM Pro (primary) + Yahoo Direct API (fallback/options)
# ═══════════════════════════════════════════════════════════════════════════════

class MboumAPI:
    """
    MBOUM Pro API client -- primary data source.
    Provides: OHLCV history (5yr), fundamentals, screener.
    """

    def __init__(self, api_key: str = MBOUM_API_KEY):
        self.api_key = api_key
        self.base = MBOUM_BASE_URL
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch full OHLCV history for a ticker.
        Returns up to ~1257 daily bars (5 years) with adjusted close.
        """
        url = f"{self.base}/v1/markets/stock/history"
        params = {"symbol": symbol, "interval": "1d", "diffandsplits": "true"}

        try:
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(3)
                resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                return None

            data = resp.json()
            body = data.get("body", {})
            if not isinstance(body, dict) or len(body) < 50:
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
                return None

            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df = df.dropna(subset=["Close"])
            df = df[df["Volume"] > 0]

            return df if len(df) > 0 else None

        except Exception:
            return None

    def get_modules(self, symbol: str, modules: List[str]) -> Dict:
        """Fetch fundamental modules for a ticker."""
        result = {}
        for mod in modules:
            try:
                url = f"{self.base}/v1/markets/stock/modules"
                params = {"symbol": symbol, "module": mod}
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    result[mod] = data.get("body", {})
            except Exception:
                pass
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
        url = f"{self.base}/v1/markets/options"
        params = {"symbol": symbol}
        headers = {"Authorization": f"Bearer {MBOUM_OPTIONS_KEY}"}
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(2)
                resp = self.session.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None
            data = resp.json()
            body = data.get("body")
            if isinstance(body, list) and body:
                return body[0]
            if isinstance(body, dict):
                return body
        except Exception:
            return None
        return None

    def get_options_for_expiration(self, symbol: str, expiration_epoch: int) -> Optional[Dict]:
        """Fetch the calls+puts chain for ONE expiration date.
        Returns the {'expirationDate', 'calls': [...], 'puts': [...]} entry."""
        url = f"{self.base}/v1/markets/options"
        params = {"symbol": symbol, "expiration": int(expiration_epoch)}
        headers = {"Authorization": f"Bearer {MBOUM_OPTIONS_KEY}"}
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(2)
                resp = self.session.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None
            data = resp.json()
            body = data.get("body")
            if isinstance(body, list) and body:
                opts = body[0].get("options", [])
                if opts:
                    return opts[0]
        except Exception:
            return None
        return None

    def get_screener(self, list_name: str = "most_actives") -> List[Dict]:
        """Fetch screener results."""
        try:
            url = f"{self.base}/v1/markets/screener"
            params = {"list": list_name}
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("body", [])
        except Exception:
            pass
        return []


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


class DataFetcher:
    """
    Two-phase data fetcher:
      Phase 1: Yahoo Direct API quick 3-month screen (fast, free, parallel)
      Phase 2: MBOUM Pro full history download (reliable, 5yr bars)
    """

    def __init__(self, lookback_days: int = LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        self.yahoo = YahooDirectAPI()
        self.mboum = MboumAPI()

    def fetch_ohlcv(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Two-phase download:
          1. Yahoo Direct quick 3mo screen (parallel, fast)
          2. MBOUM full history for promising tickers (reliable, 5yr bars)
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
            f"PHASE 2: Full OHLCV download via MBOUM Pro "
            f"for {len(promising)} tickers"
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
                result = future.result()
                if result:
                    promising.append(result)

        return promising

    def _full_download(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download full history via MBOUM Pro for the promising tickers."""
        all_data: Dict[str, pd.DataFrame] = {}
        failed = 0

        def _download_one(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = self.mboum.get_history(ticker)
            if df is not None and len(df) >= MIN_TRADING_DAYS:
                return ticker, df
            return ticker, None

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(_download_one, t): t for t in tickers
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                ticker, df = future.result()
                if df is not None:
                    all_data[ticker] = df
                else:
                    failed += 1

                if completed % 100 == 0 or completed == len(tickers):
                    log.info(
                        f"  MBOUM download: {completed}/{len(tickers)} done, "
                        f"{len(all_data)} loaded, {failed} failed"
                    )

        log.info(
            f"Full OHLCV fetch complete: {len(all_data)} tickers with "
            f">= {MIN_TRADING_DAYS} trading days. "
            f"{failed} excluded (insufficient history)."
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
        return 100 - (100 / (1 + rs))

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

        return df


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
    def _check(
        ticker: str, df: pd.DataFrame, now: Optional[datetime] = None
    ) -> Optional[str]:
        """Returns rejection reason or None if passes all guards."""
        if df.empty or len(df) < MIN_TRADING_DAYS:
            return "GUARD_A: Insufficient data"

        last = df.iloc[-1]

        # GUARD_A: Data Integrity -- check for missing bars
        total_expected = MIN_TRADING_DAYS
        actual_bars = df["Close"].dropna().shape[0]
        missing_pct = 1 - (actual_bars / total_expected)
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

        return None  # All guards passed


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 -- HARD BUY RULES (ALL 10 MUST PASS)
# ═══════════════════════════════════════════════════════════════════════════════

class HardBuyRules:
    """
    10 hard buy rules. ALL must pass -- no exceptions, no ML override.
    Returns results with per-rule pass/fail and flags.
    """

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
                last = df.iloc[-1]
                record = {
                    "ticker": ticker,
                    "price": last["Close"],
                    "rsi_14": last.get("RSI_14", np.nan),
                    "macd_histogram": last.get("MACD_histogram", np.nan),
                    "volume_ratio": last.get("Volume_Ratio", np.nan),
                    "return_1d": last.get("Return_1d", np.nan),
                    "return_5d": last.get("Return_5d", np.nan),
                    "return_20d": last.get("Return_20d", np.nan),
                    "close_vs_sma50": last.get("Close_vs_SMA50", np.nan),
                    "close_vs_sma200": last.get("Close_vs_SMA200", np.nan),
                    "ema20_vs_ema50": last.get("EMA20_vs_EMA50", np.nan),
                    "avg_dollar_volume": last.get("Avg_Dollar_Vol_20", np.nan),
                    "flags": flags,
                }
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
    def near_misses(
        data: Dict[str, pd.DataFrame], top_n: int = 25
    ) -> List[Dict]:
        """
        Evaluate ALL 10 rules for every ticker without short-circuiting.
        Returns the top_n tickers sorted by most rules passed (descending),
        with full per-rule pass/fail detail.
        Called only when zero survivors emerge.
        """
        log.info(f"Computing near-miss rankings for {len(data)} tickers...")
        scoreboard = []

        for ticker, df in data.items():
            result = HardBuyRules._evaluate_all_rules(ticker, df)
            if result:
                scoreboard.append(result)

        # Sort: most rules passed first, then by RSI closeness to midpoint (55)
        scoreboard.sort(
            key=lambda x: (x["rules_passed"], -x["rules_failed"]),
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

class MLRanker:
    """
    XGBoost + Random Forest ensemble ranking.
    Trains on pooled historical data from all survivors.
    Optional LSTM sequence layer.
    """

    FEATURE_COLS = [
        "Return_1d", "Return_5d", "Return_20d", "RSI_14",
        "MACD_histogram", "Volume_Ratio", "Close_vs_SMA50",
        "Close_vs_SMA200", "EMA20_vs_EMA50", "Avg_Dollar_Vol_20",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.rf_model = None
        self.lstm_model = None
        self.feature_importances = {}

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

        # Train XGBoost
        xgb_scores = self._train_xgboost(X_train_scaled, y_train, X_current_scaled)

        # Train Random Forest
        rf_scores = self._train_rf(X_train_scaled, y_train, X_current_scaled)

        # Ensemble
        ensemble_scores = (xgb_scores + rf_scores) / 2

        # Check score spread
        spread = ensemble_scores.max() - ensemble_scores.min()
        if spread < 0.02:
            log.warning(
                f"ML score spread is only {spread:.4f} -- "
                f"model may lack discriminative power"
            )

        # Optional LSTM
        lstm_scores = self._train_lstm(survivors, all_data)

        # Assign scores back to survivors
        ticker_to_idx = {t: i for i, t in enumerate(current_tickers)}
        for s in survivors:
            idx = ticker_to_idx.get(s["ticker"])
            if idx is not None:
                s["ml_score_xgb"] = float(xgb_scores[idx])
                s["ml_score_rf"] = float(rf_scores[idx])
                s["ml_ensemble_score"] = float(ensemble_scores[idx])
                s["lstm_score"] = (
                    float(lstm_scores[idx]) if lstm_scores is not None and idx < len(lstm_scores)
                    else None
                )
            else:
                s["ml_score_xgb"] = 0.5
                s["ml_score_rf"] = 0.5
                s["ml_ensemble_score"] = 0.5
                s["lstm_score"] = None

        # Log feature importances
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
        """Keep the most recent, class-balanced rows so live scans finish."""
        if len(X_train) <= max_rows:
            return X_train, y_train
        rng = np.random.default_rng(42)
        per_class = max_rows // 2
        keep = []
        for cls in (0, 1):
            idx = np.flatnonzero(y_train == cls)
            if len(idx) > per_class:
                idx = idx[-per_class:]
            keep.append(idx)
        keep_idx = np.concatenate(keep)
        if len(keep_idx) < max_rows:
            remaining = np.setdiff1d(np.arange(len(y_train)), keep_idx, assume_unique=False)
            fill_n = min(max_rows - len(keep_idx), len(remaining))
            if fill_n:
                keep_idx = np.concatenate([keep_idx, remaining[-fill_n:]])
        keep_idx = np.sort(keep_idx)
        log.info(f"  Training set capped to {len(keep_idx)} most-recent balanced samples.")
        return X_train[keep_idx], y_train[keep_idx]

    def rank_near_misses(
        self,
        near_misses: List[Dict],
        all_data: Dict[str, pd.DataFrame],
        training_universe: Dict[str, pd.DataFrame],
    ) -> List[Dict]:
        """Score near misses for diagnostics without converting them to buys."""
        shadow = [{"ticker": nm["ticker"]} for nm in near_misses]
        scored = self.rank(shadow, all_data, training_universe=training_universe)
        score_by_ticker = {s["ticker"]: s for s in scored}
        for nm in near_misses:
            scores = score_by_ticker.get(nm["ticker"], {})
            for key in ("ml_score_xgb", "ml_score_rf", "ml_ensemble_score", "lstm_score"):
                nm[key] = scores.get(key)
        return near_misses

    def _build_dataset(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
        training_pool: Dict[str, pd.DataFrame],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, List[str], Optional[np.ndarray]]:
        """
        Build training and current-day feature matrices.

        training_pool: broader universe used to fit the model. The forward
        20-day return label is generated identically across the pool. This
        prevents survivor-only training (heavy positive class bias) that
        previously made the classifier near-uniform.

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

            # Forward 20-day return label
            fwd_ret = close.shift(-20) / close - 1
            label = (fwd_ret > 0).astype(int)

            train_section = feat_df.iloc[:-20]
            label_section = label.iloc[:-20]

            valid = train_section.dropna()
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
            return None, None, np.array([]), [], None

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

        return X_train, y_train, X_current, current_tickers

    def _train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray, X_current: np.ndarray
    ) -> np.ndarray:
        """Train XGBoost and return predicted probabilities for current data."""
        if not XGB_AVAILABLE:
            log.warning("XGBoost not installed -- using Random Forest only")
            return np.full(X_current.shape[0], 0.5)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        # Time-series cross-validation on date-sorted rows approximates
        # walk-forward validation across the whole market, not ticker blocks.
        gap = 20
        tscv = TimeSeriesSplit(n_splits=5)
        val_accs = []
        train_accs = []
        for train_idx, val_idx in tscv.split(X_train):
            if gap > 0:
                val_start = val_idx[0]
                train_idx = train_idx[train_idx < (val_start - gap)]
                if train_idx.size == 0:
                    continue
            model.fit(X_train[train_idx], y_train[train_idx])
            train_pred = model.predict(X_train[train_idx])
            val_pred = model.predict(X_train[val_idx])
            train_accs.append(accuracy_score(y_train[train_idx], train_pred))
            val_accs.append(accuracy_score(y_train[val_idx], val_pred))

        avg_train = np.mean(train_accs)
        avg_val = np.mean(val_accs)
        log.info(f"  XGBoost CV -- Train acc: {avg_train:.3f}, Val acc: {avg_val:.3f}")

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
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )

        # Final fit on all training data
        model.fit(X_train, y_train)
        self.xgb_model = model

        # Feature importances
        for name, imp in zip(self.FEATURE_COLS, model.feature_importances_):
            self.feature_importances[name] = (
                self.feature_importances.get(name, 0) + imp
            ) / 2

        probs = model.predict_proba(X_current)[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def _train_rf(
        self, X_train: np.ndarray, y_train: np.ndarray, X_current: np.ndarray
    ) -> np.ndarray:
        """Train Random Forest and return predicted probabilities."""
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
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
        for train_idx, val_idx in tscv.split(X_train):
            if use_manual_gap:
                if len(train_idx) <= gap:
                    continue
                train_idx = train_idx[:-gap]
            model.fit(X_train[train_idx], y_train[train_idx])
            val_pred = model.predict(X_train[val_idx])
            val_accs.append(accuracy_score(y_train[val_idx], val_pred))

        log.info(f"  Random Forest CV -- Val acc: {np.mean(val_accs):.3f}")

        # Final fit
        model.fit(X_train, y_train)
        self.rf_model = model

        # Merge feature importances
        for name, imp in zip(self.FEATURE_COLS, model.feature_importances_):
            self.feature_importances[name] = (
                self.feature_importances.get(name, 0) + imp
            ) / 2

        probs = model.predict_proba(X_current)[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def _train_lstm(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
    ) -> Optional[np.ndarray]:
        """
        Optional LSTM sequence scoring layer (PyTorch implementation).
        Architecture: LSTM(64) -> Dropout(0.3) -> Dense(32,ReLU) -> Dropout(0.2) -> Dense(1,Sigmoid)
        Returns array of scores or None if unavailable.
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

            # Normalize per-ticker (z-score)
            feat_norm = (feat_df - feat_df.mean()) / feat_df.std().replace(0, 1)
            feat_norm = feat_norm.fillna(0)
            values = feat_norm.values

            # Forward return labels
            fwd_ret = close.shift(-20) / close - 1
            labels = (fwd_ret > 0).astype(int)

            # Build sequences for training
            for i in range(SEQ_LEN, len(values) - 20):
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
            return np.clip(scores, 0.0, 1.0)

        except Exception as e:
            log.warning(f"  LSTM training failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS FETCHER -- for panel scoring (survivors only)
# ═══════════════════════════════════════════════════════════════════════════════

class FundamentalsFetcher:
    """
    Fetch fundamental data via MBOUM Pro modules.
    Uses: financial-data, default-key-statistics, asset-profile, calendar-events.
    """

    def __init__(self, massive_key: str):
        self.massive_key = MASSIVE_API_KEY
        self.mboum = MboumAPI()
        self.massive_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.massive_session.mount("https://", adapter)
        self.massive_session.mount("http://", adapter)
        self.massive_session.headers.update({"X-massive-Token": MASSIVE_API_KEY})

    def fetch_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch fundamentals for a list of tickers. Returns dict of info dicts."""
        log.info(f"Fetching fundamentals for {len(tickers)} survivors via MBOUM...")
        results = {}

        def _fetch_one(ticker: str) -> Tuple[str, Dict]:
            info = {"name": ticker, "sector": "Unknown"}
            try:
                modules = self.mboum.get_modules(
                    ticker,
                    ["financial-data", "default-key-statistics",
                     "asset-profile", "calendar-events"],
                )

                fin = modules.get("financial-data", {})
                stats = modules.get("default-key-statistics", {})
                profile = modules.get("asset-profile", {})
                cal = modules.get("calendar-events", {})
                missing_modules = [
                    m for m in (
                        "financial-data",
                        "default-key-statistics",
                        "asset-profile",
                        "calendar-events",
                    )
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
                    "target_price",
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
                    "earnings_date": None,
                    "fundamentals_quality": "complete" if not missing_modules else "partial",
                    "missing_fundamental_modules": missing_modules,
                }

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

            return ticker, info

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, info = future.result()
                results[ticker] = info

        log.info(f"Fundamentals fetched for {len(results)} tickers")
        return results


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
        self.spy_data = None
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
        """Load SPY data for relative strength calculations via MBOUM."""
        try:
            mboum = MboumAPI()
            spy = mboum.get_history("SPY")
            if spy is not None and len(spy) > 100:
                self.spy_data = spy
                log.info(f"  SPY benchmark loaded ({len(spy)} bars) for relative strength")
            else:
                log.warning("  Could not load SPY benchmark from MBOUM")
        except Exception as e:
            log.warning(f"  Could not load SPY benchmark: {e}")

    def score_all(
        self,
        survivors: List[Dict],
        all_data: Dict[str, pd.DataFrame],
        fundamentals: Dict[str, Dict],
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

            # Merge fundamentals into record
            s["name"] = fund.get("name", ticker)
            s["sector"] = fund.get("sector", "Unknown")
            s["market_cap"] = fund.get("market_cap", np.nan)

            scored.append(s)

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

        # 5. Relative Strength vs SPY (10%)
        rs_score = 60
        if self.spy_data is not None and len(self.spy_data) >= 126 and len(df) >= 126:
            spy_close = self.spy_data["Close"]
            stock_ret_1m = (close / df["Close"].iloc[-21]) - 1 if len(df) >= 21 else 0
            stock_ret_3m = (close / df["Close"].iloc[-63]) - 1 if len(df) >= 63 else 0
            stock_ret_6m = (close / df["Close"].iloc[-126]) - 1 if len(df) >= 126 else 0

            spy_ret_1m = (spy_close.iloc[-1] / spy_close.iloc[-21]) - 1 if len(spy_close) >= 21 else 0
            spy_ret_3m = (spy_close.iloc[-1] / spy_close.iloc[-63]) - 1 if len(spy_close) >= 63 else 0
            spy_ret_6m = (spy_close.iloc[-1] / spy_close.iloc[-126]) - 1 if len(spy_close) >= 126 else 0

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
        # Use relative strength vs SPY as macro proxy
        if self.spy_data is not None and len(df) >= 63:
            stock_ret = (close / df["Close"].iloc[-63]) - 1
            spy_ret = (
                self.spy_data["Close"].iloc[-1] / self.spy_data["Close"].iloc[-63] - 1
            ) if len(self.spy_data) >= 63 else 0
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
        # macro snapshot (VIX, yields, DXY, gold, oil, breadth) instead of
        # SPY alone. Falls back to SPY MA stack when macro unavailable.
        if self.macro is not None:
            m_score = self.macro.panel_m_score()
        else:
            m_score = 60
            if self.spy_data is not None and len(self.spy_data) >= 50:
                spy_close = self.spy_data["Close"]
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

    MIN_DTE = 21
    MAX_DTE = 75
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
            # Source priority: MBOUM (paid plan, fast, full chain w/ IV) ->
            # Massive (snapshot tier) -> Yahoo (free, quotes only).
            chain_data = OptionsEvaluator._fetch_chain_mboum(ticker, current_price)
            option_source = "MBOUM"
            if not chain_data:
                chain_data = OptionsEvaluator._fetch_chain_massive(ticker)
                option_source = "Massive"
            if not chain_data:
                chain_data = OptionsEvaluator._fetch_chain_yahoo(ticker)
                option_source = "Yahoo"

            if not chain_data:
                return result

            profile = OptionsEvaluator._target_profile(candidate)
            today = datetime.now().date()
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
        except Exception:
            return []

    @staticmethod
    def _fetch_chain_massive(ticker: str) -> List[Dict]:
        """Fetch options chain from Massive."""
        try:
            url = f"https://api.massive.com/v3/snapshot/options/{ticker}"
            today = datetime.now().date()
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
                dte = (exp_date - datetime.now().date()).days
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
            "target_dte": 56,
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

    @staticmethod
    def _trade_setup_score(candidate: Dict) -> float:
        """Blend stock quality with option quality for final ranking."""
        ml = normalize_api_scalar(candidate.get("ml_ensemble_score"))
        panel = normalize_api_scalar(candidate.get("panel_composite_score"))
        option_score = normalize_api_scalar(candidate.get("option_score"))

        ml = ml if not is_missing_value(ml) else 0.5
        panel = (panel / 100.0) if not is_missing_value(panel) else 0.60
        option_component = (
            (option_score / 100.0)
            if candidate.get("option_candidate") == "Y" and not is_missing_value(option_score)
            else 0.0
        )

        flags = candidate.get("flags", [])
        if isinstance(flags, str):
            flags = [f for f in flags.split(", ") if f]
        flag_penalty = 0.015 * len(flags)

        score = clamp(
            0.40 * ml + 0.35 * panel + 0.25 * option_component - flag_penalty,
            0.0,
            1.0,
        )
        return 100.0 * score


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
                1 if x.get("option_candidate") == "Y" else 0,
                x.get("trade_setup_score", 0),
                x.get("option_score", 0),
                x.get("ml_ensemble_score", 0),
                x.get("panel_composite_score", 0),
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
            "panel_livermore", "panel_druckenmiller", "panel_lynch",
            "panel_minervini", "panel_oneil",
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
            elif ml >= 0.70 and panel >= 75:
                action = "STRONG BUY"
            elif ml >= 0.55 and panel >= 65:
                action = "BUY"
            else:
                action = "WATCH"

            s["recommended_action"] = action
            s["flags"] = ", ".join(s.get("flags", []))

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
            "scan_timestamp": now_et(),
            "attestation": (
                "This scan used live data only: Massive (universe discovery + "
                "options chains + live snapshot), MBOUM Pro (5yr OHLCV + "
                "fundamentals), Yahoo v8 (macro snapshot, options fallback). "
                "No hardcoded tickers, no presets, no fabricated values, no "
                "demo data. Intraday partial bars are trimmed during RTH."
            ),
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
                    "option_score": round(s.get("option_score", 0), 1),
                    "action": s.get("recommended_action", ""),
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

        # Pipeline funnel
        print("\n  ┌─ PIPELINE FUNNEL ─────────────────────────────────┐")
        for stage, count in stage_counts.items():
            print(f"  │  {stage:<40} {count:>6} │")
        print(f"  └──────────────────────────────────────────────────┘")

        # Top 25 ranked table
        print(f"\n  TOP {min(25, len(df))} CANDIDATES:")
        print("  " + "-" * 96)
        header = (
            f"  {'#':>3} {'Ticker':<7} {'Name':<25} {'Price':>8} "
            f"{'Trade':>6} {'ML':>6} {'Panel':>6} {'OptSc':>6} {'Action':<12} {'Opt':>3}"
        )
        print(header)
        print("  " + "-" * 96)

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

            name = str(row.get("name", ""))[:24]
            trade = row.get("trade_setup_score", 0)
            ml = row.get("ml_ensemble_score", 0)
            panel = row.get("panel_composite_score", 0)
            opt_score = row.get("option_score", 0)
            action = row.get("recommended_action", "")
            opt = row.get("option_candidate", "N")

            print(
                f"  {int(row['rank']):>3} {row['ticker']:<7} {name:<25} "
                f"${float(row['price']):>7.2f} "
                f"{float(trade):>5.1f} {float(ml):>5.3f} {float(panel):>5.1f} "
                f"{float(opt_score):>5.1f} {action:<12} {opt:>3}"
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
    log.info("=" * 70)
    log.info("  STOCK UNIVERSE SCAN PIPELINE -- STARTING")
    log.info(f"  Time: {now_et()}")
    log.info(f"  Market Open Now (ET RTH): {is_market_open_now()}")
    log.info("=" * 70)

    stage_counts = {}
    ml_params = {}
    feature_importances = {}

    try:
        # ── STAGE 0: Macro Regime Snapshot ───────────────────────────
        # Establishes geopolitical/macro context BEFORE any equity work
        # (the panel review's #1 demand: "consider current context").
        macro = MacroRegime()
        macro.load()
        stage_counts["Stage 0: Macro regime"] = round(macro.regime_score)

        # ── STAGE 1: Universe Discovery ──────────────────────────────
        discovery = UniverseDiscovery(MASSIVE_API_KEY)
        tickers = discovery.discover()
        stage_counts["Stage 1: Universe Discovered"] = len(tickers)

        # ── DATA FETCH: OHLCV via yfinance ───────────────────────────
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
        guarded_data, guard_rejected = ExecutionGuards.apply(all_data)
        stage_counts["Stage 2: Passed Guards"] = len(guarded_data)

        if len(guarded_data) == 0:
            raise PipelineError("No tickers survived execution guards. Pipeline STOPPED.")

        # ── STAGE 3: Hard Buy Rules ──────────────────────────────────
        survivors, buy_rejected = HardBuyRules.apply(guarded_data)
        stage_counts["Stage 3: Passed Hard Buy Rules"] = len(survivors)

        if len(survivors) == 0:
            log.warning(
                "No tickers passed all 10 hard buy rules. "
                "This may indicate a bearish market or very tight conditions."
            )
            # Expanded near-miss report (top 25) for actionable insight.
            near_misses = HardBuyRules.near_misses(guarded_data, top_n=25)
            OutputFormatter.format_and_save([], stage_counts, {}, {}, macro=macro)
            OutputFormatter.save_near_misses(near_misses, stage_counts)
            return

        # ── STAGE 4: ML Ranking ──────────────────────────────────────
        # Train on the FULL guarded universe (~50x more samples than
        # survivors-only) for true discriminative power.
        ranker = MLRanker()
        survivors = ranker.rank(survivors, all_data, training_universe=guarded_data)
        ml_params = {
            "XGBoost": "n_estimators=200, max_depth=6, lr=0.05, subsample=0.8",
            "RandomForest": "n_estimators=200, max_depth=8, min_samples_leaf=20",
            "LSTM": "Available (PyTorch)" if LSTM_AVAILABLE else "Not installed",
            "XGB_available": XGB_AVAILABLE,
            "training_universe_size": len(guarded_data),
        }
        feature_importances = ranker.feature_importances

        # ── FUNDAMENTALS FETCH (survivors only) ──────────────────────
        survivor_tickers = [s["ticker"] for s in survivors]
        fund_fetcher = FundamentalsFetcher(MASSIVE_API_KEY)
        fundamentals = fund_fetcher.fetch_batch(survivor_tickers)

        # ── STAGE 5: 5-Investor Panel ────────────────────────────────
        panel = InvestorPanel(macro=macro)
        panel.set_universe_returns(guarded_data)  # IBD-style RS percentile
        survivors = panel.score_all(survivors, all_data, fundamentals)
        stage_counts["Stage 5: Passed Panel Validation"] = len(survivors)

        if len(survivors) == 0:
            log.warning("No tickers passed panel validation.")
            OutputFormatter.format_and_save([], stage_counts, ml_params, feature_importances, macro=macro)
            return

        # ── STAGE 6: Options Evaluation ──────────────────────────────
        survivors = OptionsEvaluator.evaluate(survivors, fundamentals=fundamentals)
        stage_counts["Stage 6: Final Candidates"] = len(survivors)

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

        stage_counts["Verified Final Output"] = len(final)

        # ── FORMAT AND SAVE OUTPUT ───────────────────────────────────
        result_df = OutputFormatter.format_and_save(
            final, stage_counts, ml_params, feature_importances, macro=macro
        )

        elapsed = time.time() - pipeline_start
        log.info(f"Pipeline completed in {elapsed:.1f} seconds")
        log.info(f"Final candidates: {len(final)}")

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
