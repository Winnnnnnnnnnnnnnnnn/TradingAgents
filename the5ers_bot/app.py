"""
app.py
======
Multi-agent orchestrator + Streamlit dashboard for the The5ers Bootcamp bot.

Agent pipeline (one pass per polling interval)
---------------------------------------------
    MarketDataAgent  -> fetch candles, compute EMA / RSI / ATR
    StrategyAgent    -> turn indicators into a directional Signal
    RiskAgent        -> consult the kill switch, size the position, set SL/TP
    ExecutionAgent   -> send the order to MetaApi / MT5 and book the trade

An optional advisory "Sentinel Council" (``ai_advisor.SentinelCouncil``) is
consulted between signal generation and risk assessment. It is advisory only: it
can never block or delay trading and fails open after a hard timeout.

Threading model
---------------
The trading loop runs in a dedicated background thread with its own asyncio event
loop. Background code **never** touches any ``st.*`` API or ``st.session_state``:
credentials are resolved on the UI thread and cached, the loop communicates only
through ``orchestrator.snapshot()`` (guarded by an ``RLock``), and live dashboard
refresh uses Streamlit's native ``@st.fragment(run_every=5)`` instead of the old
``streamlit-autorefresh`` component (which contended with the ASGI websocket).

Run it:
    streamlit run the5ers_bot/app.py         # dashboard
    python the5ers_bot/app.py --once         # single headless pass (smoke test)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Streamlit Community Cloud runs `streamlit run the5ers_bot/app.py` from the
# repository root, so this file's own directory is not automatically importable.
# Prepending it makes the sibling modules resolve no matter the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker_client import AccountInfo, BrokerClient, OrderSide, PositionInfo, TradeResult
from kill_switch import KillSwitch, KillSwitchDecision

try:  # Optional: the council degrades gracefully when the LLM SDKs are absent.
    from ai_advisor import CouncilVerdict, SentinelCouncil
except Exception:  # pragma: no cover
    SentinelCouncil = None  # type: ignore
    CouncilVerdict = None  # type: ignore

load_dotenv()

LOGGER = logging.getLogger("the5ers.bot")

# Strict minimum reward:risk for any setup (1:3). A smaller REWARD_RISK is clamped
# up to this floor so the bot can never emit a trade with a worse-than-1:3 profile.
MIN_REWARD_RISK = 3.0

# Dashboard chart display parameters (independent of the strategy settings).
CHART_EMA_FAST = 9
CHART_EMA_SLOW = 21
CHART_RSI_PERIOD = 14
CHART_CANDLES = 180
DASHBOARD_REFRESH_SECONDS = 5


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float) -> float:
    try:
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return default if value in (None, "") else int(float(value))
    except (TypeError, ValueError):
        return default


def _mask(secret: str, keep: int = 4) -> str:
    """Mask a secret for display, keeping only the last few characters."""
    if not secret:
        return ""
    if len(secret) <= keep:
        return "*" * len(secret)
    return "*" * (len(secret) - keep) + secret[-keep:]


def _streamlit_secret(key: str) -> str:
    """Read a value from ``st.secrets``. UI thread only.

    On Streamlit Community Cloud credentials live in the Secrets manager, which is
    exposed to the app as ``st.secrets``. Returns ``""`` outside a Streamlit runtime
    or when the key is absent, so local (``.env``) runs are unaffected.
    """
    try:
        import streamlit as st
    except Exception:  # streamlit not installed / not a Streamlit process
        return ""
    try:
        value = st.secrets.get(key, "")
    except Exception:  # no secrets file, malformed TOML, missing key, ...
        return ""
    return str(value).strip() if value not in (None, "") else ""


def _credential_source(dashboard: str, secret: str, env: str) -> str:
    """Label where a credential value came from, in resolution order."""
    if dashboard:
        return "Dashboard"
    if secret:
        return "Streamlit secrets"
    if env:
        return "Environment"
    return "Missing"


@dataclass
class BotConfig:
    """Runtime configuration, sourced entirely from environment variables."""

    symbols: List[str] = field(default_factory=lambda: ["BTCUSD", "ETHUSD"])
    timeframe: str = "15m"
    candle_limit: int = 200
    poll_interval_seconds: int = 60

    ema_fast: int = 21
    ema_slow: int = 55
    rsi_period: int = 14
    rsi_long_min: float = 52.0
    rsi_short_max: float = 48.0

    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    reward_risk: float = MIN_REWARD_RISK
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 2

    dry_run: bool = False
    paper_mode: bool = False

    ai_council_enabled: bool = True
    ai_council_timeout: float = 8.0

    @classmethod
    def from_env(cls) -> "BotConfig":
        symbols = [
            s.strip().upper()
            for s in os.getenv("TRADING_SYMBOLS", "BTCUSD,ETHUSD").split(",")
            if s.strip()
        ]
        return cls(
            symbols=symbols or ["BTCUSD"],
            timeframe=os.getenv("STRATEGY_TIMEFRAME", "15m"),
            candle_limit=_as_int(os.getenv("CANDLE_LIMIT"), 200),
            poll_interval_seconds=_as_int(os.getenv("POLL_INTERVAL_SECONDS"), 60),
            ema_fast=_as_int(os.getenv("EMA_FAST"), 21),
            ema_slow=_as_int(os.getenv("EMA_SLOW"), 55),
            rsi_period=_as_int(os.getenv("RSI_PERIOD"), 14),
            rsi_long_min=_as_float(os.getenv("RSI_LONG_MIN"), 52.0),
            rsi_short_max=_as_float(os.getenv("RSI_SHORT_MAX"), 48.0),
            atr_period=_as_int(os.getenv("ATR_PERIOD"), 14),
            atr_stop_multiplier=_as_float(os.getenv("ATR_STOP_MULTIPLIER"), 2.0),
            # Strict minimum 1:3 - a lower REWARD_RISK is clamped up to the floor.
            reward_risk=max(MIN_REWARD_RISK, _as_float(os.getenv("REWARD_RISK"), MIN_REWARD_RISK)),
            risk_per_trade_pct=_as_float(os.getenv("RISK_PER_TRADE_PCT"), 0.5),
            max_open_positions=_as_int(os.getenv("MAX_OPEN_POSITIONS"), 2),
            dry_run=_as_bool(os.getenv("DRY_RUN"), False),
            paper_mode=_as_bool(os.getenv("PAPER_MODE"), False),
            ai_council_enabled=_as_bool(os.getenv("AI_COUNCIL_ENABLED"), True),
            ai_council_timeout=_as_float(os.getenv("AI_COUNCIL_TIMEOUT"), 8.0),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "Symbols": ", ".join(self.symbols),
            "Timeframe": self.timeframe,
            "Poll interval (s)": self.poll_interval_seconds,
            "EMA fast / slow": f"{self.ema_fast} / {self.ema_slow}",
            "Risk per trade": f"{self.risk_per_trade_pct}% of equity",
            "Stop distance": f"{self.atr_stop_multiplier} x ATR({self.atr_period})",
            "Reward : risk": f"1 : {self.reward_risk:g} (min 1:{MIN_REWARD_RISK:g})",
            "Max open positions": self.max_open_positions,
            "Paper mode": self.paper_mode,
            "Dry run": self.dry_run,
            "AI council": self.ai_council_enabled,
        }


# --------------------------------------------------------------------------- #
# Signal / decision objects
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    symbol: str
    side: OrderSide
    price: float
    atr: float
    strength: float
    reason: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RiskDecision:
    approved: bool
    signal: Optional[Signal]
    symbol: str = ""
    volume: float = 0.0
    entry_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_usd: float = 0.0
    reason: str = ""

    @property
    def side(self) -> OrderSide:
        return self.signal.side if self.signal else OrderSide.BUY


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
class MarketDataAgent:
    """Fetches candles and computes the indicator set used by the strategy."""

    name = "market-data"

    def __init__(self, broker: BrokerClient, config: BotConfig) -> None:
        self.broker = broker
        self.config = config

    async def fetch_frame(self, symbol: str) -> pd.DataFrame:
        candles = await self.broker.get_candles(symbol, self.config.timeframe, self.config.candle_limit)
        return self.build_frame(candles)

    @staticmethod
    def build_frame(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        frame = pd.DataFrame(candles)
        for column in ("open", "high", "low", "close", "volume"):
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["close"]).reset_index(drop=True)

    def add_indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        cfg = self.config
        out = frame.copy()

        out["ema_fast"] = out["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
        out["ema_slow"] = out["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

        delta = out["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / cfg.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / cfg.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        prev_close = out["close"].shift(1)
        true_range = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = true_range.ewm(alpha=1.0 / cfg.atr_period, adjust=False).mean()

        return out


class StrategyAgent:
    """EMA crossover with an RSI confirmation filter."""

    name = "strategy"

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def generate_signal(self, symbol: str, frame: pd.DataFrame) -> Optional[Signal]:
        cfg = self.config
        required = max(cfg.ema_slow, cfg.rsi_period, cfg.atr_period) + 2
        if frame.empty or len(frame) < required:
            return None

        last = frame.iloc[-1]
        prev = frame.iloc[-2]
        atr = float(last.get("atr", 0.0) or 0.0)
        if not np.isfinite(atr) or atr <= 0:
            return None

        fast_now, slow_now = float(last["ema_fast"]), float(last["ema_slow"])
        fast_prev, slow_prev = float(prev["ema_fast"]), float(prev["ema_slow"])
        rsi = float(last["rsi"])
        close = float(last["close"])

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        strength = round(abs(fast_now - slow_now) / atr, 4)

        if crossed_up and rsi >= cfg.rsi_long_min and close > slow_now:
            return Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                price=close,
                atr=atr,
                strength=strength,
                reason=f"EMA{cfg.ema_fast} crossed above EMA{cfg.ema_slow}, RSI {rsi:.1f} >= {cfg.rsi_long_min}",
            )

        if crossed_down and rsi <= cfg.rsi_short_max and close < slow_now:
            return Signal(
                symbol=symbol,
                side=OrderSide.SELL,
                price=close,
                atr=atr,
                strength=strength,
                reason=f"EMA{cfg.ema_fast} crossed below EMA{cfg.ema_slow}, RSI {rsi:.1f} <= {cfg.rsi_short_max}",
            )

        return None


class RiskAgent:
    """Validates every trade against the kill switch and sizes it from ATR."""

    name = "risk"

    def __init__(self, broker: BrokerClient, config: BotConfig, kill_switch: KillSwitch) -> None:
        self.broker = broker
        self.config = config
        self.kill_switch = kill_switch

    async def assess(
        self,
        signal: Signal,
        entry_price: float,
        account: AccountInfo,
        positions: List[PositionInfo],
        decision: KillSwitchDecision,
    ) -> RiskDecision:
        cfg = self.config
        resolved = self.broker.resolve_symbol(signal.symbol)

        if not decision.allowed:
            return RiskDecision(False, signal, symbol=resolved, reason=f"Kill switch: {decision.reason}")
        if len(positions) >= cfg.max_open_positions:
            return RiskDecision(
                False, signal, symbol=resolved,
                reason=f"Max open positions reached ({cfg.max_open_positions}).",
            )
        if any(p.symbol == resolved for p in positions):
            return RiskDecision(False, signal, symbol=resolved, reason="Position already open on this symbol.")

        spec = await self.broker.get_symbol_specification(signal.symbol)
        stop_distance = signal.atr * cfg.atr_stop_multiplier
        if stop_distance <= 0:
            return RiskDecision(False, signal, symbol=resolved, reason="Invalid ATR-based stop distance.")

        risk_budget = account.equity * (cfg.risk_per_trade_pct / 100.0)
        if decision.remaining_risk > 0:
            risk_budget = min(risk_budget, decision.remaining_risk)
        if risk_budget <= 0:
            return RiskDecision(False, signal, symbol=resolved, reason="No risk budget left for today.")

        loss_per_lot = (stop_distance / spec.tick_size) * spec.tick_value
        if loss_per_lot <= 0:
            return RiskDecision(False, signal, symbol=resolved, reason="Symbol specification gives zero tick value.")

        raw_volume = risk_budget / loss_per_lot
        volume = spec.normalize_volume(raw_volume)
        if volume < spec.min_volume:
            return RiskDecision(
                False, signal, symbol=resolved,
                reason=(f"Risk budget {risk_budget:.2f} USD is below the minimum lot risk "
                        f"({spec.min_volume} lot = {spec.min_volume * loss_per_lot:.2f} USD)."),
            )

        actual_risk = round(volume * loss_per_lot, 2)
        take_distance = stop_distance * cfg.reward_risk
        digits = spec.digits

        if signal.side.is_long:
            stop_loss = round(entry_price - stop_distance, digits)
            take_profit = round(entry_price + take_distance, digits)
        else:
            stop_loss = round(entry_price + stop_distance, digits)
            take_profit = round(entry_price - take_distance, digits)

        return RiskDecision(
            approved=True,
            signal=signal,
            symbol=resolved,
            volume=volume,
            entry_price=round(entry_price, digits),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_usd=actual_risk,
            reason=(f"{signal.side.value.upper()} {volume} {resolved} | risk {actual_risk:.2f} USD "
                    f"({cfg.risk_per_trade_pct}% budget, {decision.remaining_risk:.2f} USD left today)"),
        )


class ExecutionAgent:
    """Sends approved decisions to the broker."""

    name = "execution"

    def __init__(self, broker: BrokerClient, kill_switch: KillSwitch, dry_run: bool = False) -> None:
        self.broker = broker
        self.kill_switch = kill_switch
        self.dry_run = dry_run

    async def execute(self, decision: RiskDecision) -> TradeResult:
        if not decision.approved or decision.signal is None:
            return TradeResult(success=False, action="open", message=decision.reason)

        if self.dry_run:
            LOGGER.info("[DRY RUN] would open %s", decision.reason)
            return TradeResult(
                success=True, action="open", symbol=decision.symbol,
                side=decision.side.value, volume=decision.volume, price=decision.entry_price,
                message=f"dry run: {decision.reason}",
            )

        result = await self.broker.open_market_order(
            symbol=decision.symbol,
            side=decision.side,
            volume=decision.volume,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            comment="the5ers-multiagent",
        )
        if result.success:
            self.kill_switch.note_trade_opened()
        return result


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class TradingOrchestrator:
    """Runs the agent pipeline in a background thread for 24/7 operation."""

    def __init__(self, config: Optional[BotConfig] = None, kill_switch: Optional[KillSwitch] = None) -> None:
        self.config = config or BotConfig.from_env()
        self.kill_switch = kill_switch or KillSwitch()

        # Credentials supplied at runtime from the dashboard. Empty values fall back
        # to the environment / st.secrets. ``gemini_api_key`` / ``groq_api_key`` feed
        # the Sentinel Council.
        self._credentials: Dict[str, str] = {
            "token": "",
            "account_id": "",
            "gemini_api_key": "",
            "groq_api_key": "",
        }
        # Effective credentials resolved ON THE UI THREAD and cached. The background
        # loop reads this copy so it never has to touch st.secrets / any st.* API.
        self._resolved_credentials: Dict[str, str] = {
            "token": "",
            "account_id": "",
            "gemini_api_key": "",
            "groq_api_key": "",
        }

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._broker: Optional[BrokerClient] = None
        self._market_agent: Optional[MarketDataAgent] = None
        self._strategy_agent: Optional[StrategyAgent] = None
        self._risk_agent: Optional[RiskAgent] = None
        self._execution_agent: Optional[ExecutionAgent] = None

        self._council: Optional[Any] = self._build_council({})

        self._signals: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._trades: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._events: Deque[str] = deque(maxlen=500)
        self._council_feed: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._frames: Dict[str, pd.DataFrame] = {}

        self._state: Dict[str, Any] = {
            "running": False,
            "connected": False,
            "started_at": None,
            "last_tick": None,
            "last_error": None,
            "tick_count": 0,
            "account": None,
            "decision": None,
            "positions": [],
            "flattened_for_lock": False,
            "ladder": None,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return bool(self._state["connected"])

    def log_event(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}Z] {message}"
        with self._lock:
            self._events.append(line)
        LOGGER.info(message)

    def set_credentials(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ) -> bool:
        """Store dashboard-supplied credentials for the next connection attempt.

        A ``None`` argument leaves the stored value untouched, while an empty
        string explicitly clears it (so the fallback chain to ``st.secrets`` /
        environment kicks in). Returns True when anything actually changed.
        """
        with self._lock:
            current = dict(self._credentials)
            updated = dict(current)
            if token is not None:
                updated["token"] = token.strip()
            if account_id is not None:
                updated["account_id"] = account_id.strip()
            if gemini_api_key is not None:
                updated["gemini_api_key"] = gemini_api_key.strip()
            if groq_api_key is not None:
                updated["groq_api_key"] = groq_api_key.strip()
            changed = updated != current
            self._credentials = updated
        if changed:
            broker_set = bool(updated["token"] or updated["account_id"])
            ai_set = bool(updated["gemini_api_key"] or updated["groq_api_key"])
            source = "dashboard" if (broker_set or ai_set) else "environment"
            self.log_event(
                f"Credentials updated (source: {source}; MetaApi={'set' if broker_set else 'env'}, "
                f"AI={'set' if ai_set else 'env'}). Restart the bot to apply them."
            )
        return changed

    def _effective_credentials(self) -> Dict[str, str]:
        """Resolve credentials: dashboard inputs, then ``st.secrets``, then environment.

        UI thread only - it reads ``st.secrets``.
        """
        with self._lock:
            ui = dict(self._credentials)
        return {
            "token": ui["token"] or _streamlit_secret("METAAPI_TOKEN") or os.getenv("METAAPI_TOKEN", "").strip(),
            "account_id": ui["account_id"] or _streamlit_secret("METAAPI_ACCOUNT_ID") or os.getenv("METAAPI_ACCOUNT_ID", "").strip(),
            "gemini_api_key": (
                ui["gemini_api_key"]
                or _streamlit_secret("GEMINI_API_KEY")
                or _streamlit_secret("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            ),
            "groq_api_key": (
                ui["groq_api_key"]
                or _streamlit_secret("GROQ_API_KEY")
                or os.getenv("GROQ_API_KEY", "").strip()
            ),
        }

    def resolve_credentials(self) -> Dict[str, str]:
        """Resolve credentials on the calling (UI) thread and cache them.

        The background trading thread must never touch ``st.*``; it reads this
        cached copy instead. Called from the dashboard before ``start()`` and from
        the headless single-pass runner. The Sentinel Council is (re)built here so
        its ``status()`` reflects the dashboard-supplied keys before the loop runs.
        """
        resolved = self._effective_credentials()
        with self._lock:
            self._resolved_credentials = resolved
            self._council = self._build_council(resolved)
        return resolved

    def _build_council(self, creds: Dict[str, str]) -> Optional[Any]:
        """Construct a ``SentinelCouncil`` from already-resolved credentials.

        Pure: only uses the passed-in values, so it is safe on any thread (it never
        reads ``st.*``).
        """
        if SentinelCouncil is None or not self.config.ai_council_enabled:
            return None
        return SentinelCouncil(
            timeout=self.config.ai_council_timeout,
            groq_api_key=creds.get("groq_api_key") or None,
            gemini_api_key=creds.get("gemini_api_key") or None,
            enabled=True,
        )

    def credentials_info(self) -> Dict[str, Any]:
        """Describe which credentials the bot would use (secrets are masked)."""
        with self._lock:
            ui = dict(self._credentials)
        secret_token = _streamlit_secret("METAAPI_TOKEN")
        secret_account = _streamlit_secret("METAAPI_ACCOUNT_ID")
        secret_gemini = _streamlit_secret("GEMINI_API_KEY") or _streamlit_secret("GOOGLE_API_KEY")
        secret_groq = _streamlit_secret("GROQ_API_KEY")
        env_token, env_account = os.getenv("METAAPI_TOKEN", "").strip(), os.getenv("METAAPI_ACCOUNT_ID", "").strip()
        env_gemini = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        env_groq = os.getenv("GROQ_API_KEY", "").strip()
        effective = self._effective_credentials()
        return {
            "token_masked": _mask(effective["token"]),
            "token_source": _credential_source(ui["token"], secret_token, env_token),
            "account_id": effective["account_id"] or "(not set)",
            "account_source": _credential_source(ui["account_id"], secret_account, env_account),
            "gemini_set": bool(effective["gemini_api_key"]),
            "gemini_source": _credential_source(ui["gemini_api_key"], secret_gemini, env_gemini),
            "groq_set": bool(effective["groq_api_key"]),
            "groq_source": _credential_source(ui["groq_api_key"], secret_groq, env_groq),
        }

    def start(self) -> bool:
        """Start the background trading loop. Returns False if already running."""
        with self._lock:
            if self._state["running"]:
                return False
        # Resolve credentials on this (UI) thread so the loop never calls st.*.
        self.resolve_credentials()
        with self._lock:
            if self._state["running"]:
                return False
            self._stop_event.clear()
            self._state.update({"running": True, "started_at": datetime.now(timezone.utc).isoformat(),
                                "last_error": None, "tick_count": 0})
            self._thread = threading.Thread(target=self._thread_entry, name="trading-loop", daemon=True)
            self._thread.start()
        self.log_event("Trading loop started.")
        return True

    def stop(self, timeout: float = 15.0) -> None:
        """Signal the loop to stop and wait for the thread to finish."""
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._state["running"] = False
        self.log_event("Trading loop stopped.")

    def snapshot(self) -> Dict[str, Any]:
        """Thread-safe copy of everything the dashboard needs (UI thread only)."""
        with self._lock:
            return {
                "state": dict(self._state),
                "ladder": self._state.get("ladder"),
                "signals": list(self._signals),
                "trades": list(self._trades),
                "events": list(self._events),
                "council": list(self._council_feed),
                "council_status": self.council_status(),
                "frames": {symbol: frame.copy() for symbol, frame in self._frames.items()},
                "kill_switch": self.kill_switch.status(),
                "credentials": self.credentials_info(),
            }

    def council_status(self) -> Dict[str, Any]:
        if self._council is None:
            return {"enabled": False, "scout": False, "auditor": False, "timeout": 0.0}
        return self._council.status()

    def close_all_now(self, timeout: float = 60.0) -> List[TradeResult]:
        """Flatten every position, executed on the trading loop thread."""
        loop = self._loop
        if loop is None or not loop.is_running():
            self.log_event("Cannot close positions: trading loop is not running.")
            return []
        future = asyncio.run_coroutine_threadsafe(self._flatten_all("manual close-all"), loop)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            self.log_event(f"Close-all failed: {exc}")
            return []

    # ------------------------------------------------------------------ #
    # Thread / loop plumbing
    # ------------------------------------------------------------------ #
    def _thread_entry(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        except Exception as exc:  # noqa: BLE001 - top-level guard
            LOGGER.exception("Trading loop crashed: %s", exc)
            with self._lock:
                self._state["last_error"] = str(exc)
            self.log_event(f"Trading loop crashed: {exc}")
        finally:
            try:
                loop.run_until_complete(self._shutdown())
            except Exception:  # noqa: BLE001
                pass
            loop.close()
            self._loop = None
            with self._lock:
                self._state.update({"running": False, "connected": False})
            self.log_event("Trading loop exited.")

    async def _shutdown(self) -> None:
        if self._broker is not None:
            await self._broker.disconnect()
            self._broker = None

    async def _run(self) -> None:
        # Read the credentials cached by the UI thread; never call st.* here.
        with self._lock:
            creds = dict(self._resolved_credentials)

        # (Re)build the Sentinel Council from the cached keys. The helper only uses
        # values passed in here - no st.* / st.secrets access on this thread.
        self._council = self._build_council(creds)

        # Bind the kill-switch state to this account: switching accounts must never
        # inherit another account's daily baseline or lock.
        self.kill_switch.set_account_scope(creds["account_id"] or None)

        self._broker = BrokerClient(
            token=creds["token"] or None,
            account_id=creds["account_id"] or None,
            paper_mode=self.config.paper_mode,
        )
        await self._broker.connect()

        self._market_agent = MarketDataAgent(self._broker, self.config)
        self._strategy_agent = StrategyAgent(self.config)
        self._risk_agent = RiskAgent(self._broker, self.config, self.kill_switch)
        self._execution_agent = ExecutionAgent(self._broker, self.kill_switch, dry_run=self.config.dry_run)

        with self._lock:
            self._state["connected"] = True
        self.log_event(
            f"Broker connected ({'paper' if self.config.paper_mode else 'metaapi'}). "
            f"Watching {', '.join(self.config.symbols)} on {self.config.timeframe}."
        )

        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                await self._tick()
            except Exception as exc:  # noqa: BLE001 - keep the bot alive
                LOGGER.exception("Tick failed: %s", exc)
                with self._lock:
                    self._state["last_error"] = str(exc)
                self.log_event(f"Tick error: {exc}")

            elapsed = time.monotonic() - started
            await self._interruptible_sleep(max(1.0, self.config.poll_interval_seconds - elapsed))

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in short slices so ``stop()`` is honoured promptly.

        A plain ``asyncio.sleep(poll_interval)`` would keep the loop (and the
        broker connection) alive for up to a full polling interval after the Stop
        button was pressed, leaving the dashboard stuck on ``connected``.
        """
        remaining = max(0.0, seconds)
        while remaining > 0 and not self._stop_event.is_set():
            slice_seconds = min(0.25, remaining)
            await asyncio.sleep(slice_seconds)
            remaining -= slice_seconds

    # ------------------------------------------------------------------ #
    # One iteration of the agent pipeline
    # ------------------------------------------------------------------ #
    async def _tick(self) -> None:
        assert self._broker and self._market_agent and self._strategy_agent
        assert self._risk_agent and self._execution_agent

        account = await self._broker.get_account_information()
        decision = self.kill_switch.evaluate(equity=account.equity, balance=account.balance)
        positions = await self._broker.get_positions()

        with self._lock:
            self._state.update(
                {
                    "last_tick": datetime.now(timezone.utc).isoformat(),
                    "tick_count": self._state["tick_count"] + 1,
                    "account": account,
                    "decision": decision,
                    "positions": positions,
                }
            )

        # Refresh the dashboard preview (chart candle frame + indicative ladder) even
        # while the kill switch has trading halted, so the UI never goes stale.
        await self._refresh_market_preview()

        # --- lock handling ------------------------------------------------- #
        if not decision.allowed:
            with self._lock:
                already_flattened = self._state["flattened_for_lock"]
            if self.kill_switch.auto_close_on_lock and positions and not already_flattened:
                self.log_event(f"Flattening {len(positions)} position(s): {decision.reason}")
                await self._flatten_all("kill-switch flatten")
            elif not positions:
                with self._lock:
                    self._state["flattened_for_lock"] = True
            if not already_flattened:
                self.log_event(f"Trading halted - {decision.reason}")
            return

        with self._lock:
            self._state["flattened_for_lock"] = False

        # --- signal generation + execution --------------------------------- #
        # Track open positions tick-locally: the broker snapshot fetched above is
        # stale once we start opening orders in this same iteration, so without a
        # running count several symbols could fill and breach max_open_positions.
        open_symbols = {p.symbol for p in positions}
        open_count = len(positions)
        for symbol in self.config.symbols:
            resolved = self._broker.resolve_symbol(symbol)
            if resolved in open_symbols:
                continue
            if open_count >= self.config.max_open_positions:
                break

            frame = await self._market_agent.fetch_frame(symbol)
            if frame.empty:
                continue
            frame = self._market_agent.add_indicators(frame)
            with self._lock:
                self._frames[symbol] = frame.tail(CHART_CANDLES)

            signal = self._strategy_agent.generate_signal(symbol, frame)
            if signal is None:
                continue

            self._record_signal(signal)

            # Advisory council (fail-open, never blocks the pipeline).
            await self._consult_council(symbol, signal, decision, account)

            quote = await self._broker.get_symbol_price(symbol)
            entry_price = quote["ask"] if signal.side.is_long else quote["bid"]
            if entry_price <= 0:
                self.log_event(f"Skipped {symbol}: no tradable quote.")
                continue

            risk = await self._risk_agent.assess(signal, entry_price, account, positions, decision)
            if not risk.approved:
                self._record_trade(
                    symbol=resolved, side=signal.side.value, volume=0.0, price=entry_price,
                    status="rejected", detail=risk.reason,
                )
                continue

            result = await self._execution_agent.execute(risk)
            self._record_trade(
                symbol=risk.symbol, side=signal.side.value, volume=risk.volume,
                price=risk.entry_price, status="filled" if result.success else "failed",
                detail=result.message if result.success else f"{risk.reason} -> {result.message}",
                stop_loss=risk.stop_loss, take_profit=risk.take_profit, risk_usd=risk.risk_usd,
            )
            if result.success:
                open_symbols.add(risk.symbol)
                open_count += 1
                entry = self.kill_switch.status()
                self.log_event(
                    f"Opened {signal.side.value.upper()} {risk.volume} {risk.symbol} @ {risk.entry_price} "
                    f"(SL {risk.stop_loss}, TP {risk.take_profit}, risk {risk.risk_usd:.2f} USD) | "
                    f"day P&L {decision.daily_pnl:+.2f} USD, trades today {entry['trades_today']}"
                )

    async def _refresh_market_preview(self) -> None:
        """Keep the chart frame + orderbook ladder fresh for the dashboard."""
        if not self._broker or not self._market_agent or not self.config.symbols:
            return
        primary = self.config.symbols[0]
        try:
            frame = await self._market_agent.fetch_frame(primary)
            if not frame.empty:
                frame = self._market_agent.add_indicators(frame)
                with self._lock:
                    self._frames[primary] = frame.tail(CHART_CANDLES)

            quote = await self._broker.get_symbol_price(primary)
            if quote.get("bid", 0) and quote.get("ask", 0):
                try:
                    spec = await self._broker.get_symbol_specification(primary)
                except Exception:  # noqa: BLE001 - ladder is cosmetic
                    spec = None
                ladder = self._build_ladder(primary, quote, spec)
                with self._lock:
                    self._state["ladder"] = ladder
        except Exception as exc:  # noqa: BLE001 - preview must never break the tick
            LOGGER.debug("Market preview refresh failed: %s", exc)

    @staticmethod
    def _build_ladder(symbol: str, quote: Dict[str, float], spec: Optional[Any], levels: int = 8) -> Dict[str, Any]:
        """Indicative orderbook ladder derived from the live bid/ask and tick size.

        MetaApi does not stream a full depth book for every broker, so this is a
        clearly-labelled *indicative* ladder: real quotes at the top of book, with
        deterministic depth rows at the instrument's tick size for display.
        """
        bid = float(quote.get("bid", 0.0) or 0.0)
        ask = float(quote.get("ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            return {}
        digits = int(getattr(spec, "digits", 2) or 2)
        tick = float(getattr(spec, "tick_size", 0.0) or 0.0)
        if tick <= 0:
            tick = max(round(ask - bid, 6), 10 ** -digits)

        bids: List[Dict[str, float]] = []
        asks: List[Dict[str, float]] = []
        for i in range(levels):
            bids.append(
                {"price": round(bid - i * tick, digits), "size": round(0.50 + 0.35 * ((i * 7 + 11) % 9), 2)}
            )
            asks.append(
                {"price": round(ask + i * tick, digits), "size": round(0.40 + 0.30 * ((i * 5 + 3) % 8), 2)}
            )
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "mid": round((ask + bid) / 2.0, digits),
            "spread": round(ask - bid, digits),
            "digits": digits,
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    async def _consult_council(
        self,
        symbol: str,
        signal: Signal,
        decision: KillSwitchDecision,
        account: AccountInfo,
    ) -> None:
        """Ask the Sentinel Council for an advisory read. Never blocks or raises."""
        if self._council is None:
            return
        context = {
            "symbol": symbol,
            "side": signal.side.value.upper(),
            "price": round(signal.price, 4),
            "atr": round(signal.atr, 4),
            "strength": signal.strength,
            "reason": signal.reason,
            "equity": round(account.equity, 2),
            "daily_pnl": decision.daily_pnl,
            "profit_target": decision.profit_target,
            "max_daily_loss": decision.max_daily_loss,
            "remaining_risk": decision.remaining_risk,
            "reward_risk_floor": MIN_REWARD_RISK,
        }
        try:
            verdict = await self._council.deliberate(context)
        except Exception as exc:  # noqa: BLE001 - belt and braces; council is fail-open
            LOGGER.warning("Sentinel council failed open: %s", exc)
            return
        entry = verdict.to_dict()
        entry["symbol"] = symbol
        entry["side"] = signal.side.value.upper()
        with self._lock:
            self._council_feed.append(entry)
        scout_model = entry.get("scout_model_used") or "-"
        auditor_model = entry.get("auditor_model_used") or "-"
        fallback = " [FALLBACK]" if entry.get("fallback_used") else ""
        self.log_event(
            f"Sentinel council: {entry.get('recommendation')} "
            f"({entry.get('confidence', 0.0):.0%}) | scout={scout_model} auditor={auditor_model}{fallback}"
            f" - {str(entry.get('summary', ''))[:120]}"
        )

    async def _flatten_all(self, reason: str) -> List[TradeResult]:
        assert self._broker is not None
        results = await self._broker.close_all_positions()
        for result in results:
            self._record_trade(
                symbol=result.symbol, side="close", volume=result.volume, price=result.price or 0.0,
                status="closed" if result.success else "failed", detail=result.message,
            )
        with self._lock:
            self._state["flattened_for_lock"] = True
        self.log_event(f"Flattened {len(results)} position(s) ({reason}).")
        return results

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def _record_signal(self, signal: Signal) -> None:
        entry = {
            "time": signal.generated_at,
            "symbol": signal.symbol,
            "side": signal.side.value.upper(),
            "price": round(signal.price, 4),
            "atr": round(signal.atr, 4),
            "strength": signal.strength,
            "reason": signal.reason,
        }
        with self._lock:
            self._signals.append(entry)
        LOGGER.info("Signal %s %s: %s", signal.side.value.upper(), signal.symbol, signal.reason)

    def _record_trade(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        status: str,
        detail: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_usd: float = 0.0,
    ) -> None:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side.upper(),
            "volume": volume,
            "price": round(price, 4) if price else None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_usd": risk_usd,
            "status": status,
            "detail": detail,
        }
        with self._lock:
            self._trades.append(entry)
        if status in {"filled", "failed", "closed"}:
            LOGGER.info("Trade %s: %s", status, entry)


# --------------------------------------------------------------------------- #
# Process-wide orchestrator singleton
# --------------------------------------------------------------------------- #
_ORCHESTRATOR: Optional["TradingOrchestrator"] = None
_ORCHESTRATOR_LOCK = threading.Lock()


def get_orchestrator(config: Optional[BotConfig] = None) -> "TradingOrchestrator":
    """Return the single process-wide orchestrator instance.

    Streamlit reruns this script on every interaction and may serve several browser
    sessions from one process. Building a fresh orchestrator each time would spawn
    duplicate background trading threads and split the kill-switch state across them,
    so the instance is created once, race-free, and reused everywhere. Trading state
    that must survive restarts lives in the kill-switch state file, not in memory.
    """
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        with _ORCHESTRATOR_LOCK:
            if _ORCHESTRATOR is None:
                _ORCHESTRATOR = TradingOrchestrator(config or BotConfig.from_env())
    return _ORCHESTRATOR


# --------------------------------------------------------------------------- #
# Headless single pass (smoke test / cron usage)
# --------------------------------------------------------------------------- #
def run_single_pass() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = BotConfig.from_env()
    orchestrator = TradingOrchestrator(config)
    # Resolve credentials on this (main) thread; the async body never touches st.*.
    creds = orchestrator.resolve_credentials()

    async def _one_pass() -> None:
        orchestrator.kill_switch.set_account_scope(creds["account_id"] or None)
        orchestrator._broker = BrokerClient(
            token=creds["token"] or None,
            account_id=creds["account_id"] or None,
            paper_mode=config.paper_mode,
        )
        await orchestrator._broker.connect()
        orchestrator._market_agent = MarketDataAgent(orchestrator._broker, config)
        orchestrator._strategy_agent = StrategyAgent(config)
        orchestrator._risk_agent = RiskAgent(orchestrator._broker, config, orchestrator.kill_switch)
        orchestrator._execution_agent = ExecutionAgent(orchestrator._broker, orchestrator.kill_switch, config.dry_run)
        try:
            for _ in range(3):  # a few passes so indicators warm up in paper mode
                await orchestrator._tick()
        finally:
            await orchestrator._broker.disconnect()

    asyncio.run(_one_pass())

    snapshot = orchestrator.snapshot()
    account = snapshot["state"]["account"]
    if account is not None:
        print(f"balance={account.balance:.2f} equity={account.equity:.2f} currency={account.currency}")
    print(f"signals={len(snapshot['signals'])} trades={len(snapshot['trades'])}")
    for line in snapshot["events"][-8:]:
        print(line)
    return 0


# --------------------------------------------------------------------------- #
# Streamlit dashboard
# --------------------------------------------------------------------------- #
def _kill_switch_state(kill_status: Dict[str, Any], running: bool) -> Tuple[str, str]:
    """Map the raw kill-switch status onto a (label, level) pair for the UI.

    ``level`` is one of ``emergency`` / ``locked`` / ``active`` / ``idle`` and
    drives which coloured banner the dashboard renders.
    """
    if kill_status.get("locked"):
        lock_type = kill_status.get("lock_type")
        if lock_type == "max_daily_loss":
            return "EMERGENCY STOPPED", "emergency"
        if lock_type == "profit_target":
            return "LOCKED · TARGET BANKED", "locked"
        return "LOCKED · MANUAL", "locked"
    if running:
        return "ACTIVE", "active"
    return "ARMED · IDLE", "idle"


def _build_candlestick(frame: pd.DataFrame, symbol: str) -> Optional[Any]:
    """Build the Plotly candlestick + EMA(9/21) + RSI(14) figure.

    ``uirevision`` is pinned to a constant so the user's zoom/pan survive the
    5-second fragment refresh.
    """
    if frame is None or frame.empty:
        return None
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:  # plotly not installed
        return None

    df = frame.copy()
    if "time" in df.columns:
        x = pd.to_datetime(df["time"], errors="coerce")
        if x.isna().all():
            x = df.index
    else:
        x = df.index

    close = pd.to_numeric(df["close"], errors="coerce")
    ema_fast = close.ewm(span=CHART_EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=CHART_EMA_SLOW, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = gain.ewm(alpha=1.0 / CHART_RSI_PERIOD, adjust=False).mean() / (
        loss.ewm(alpha=1.0 / CHART_RSI_PERIOD, adjust=False).mean().replace(0.0, np.nan)
    )
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=x, open=df["open"], high=df["high"], low=df["low"], close=close,
            name=symbol, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=ema_fast, name=f"EMA {CHART_EMA_FAST}",
                   line=dict(color="#42a5f5", width=1.4)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=ema_slow, name=f"EMA {CHART_EMA_SLOW}",
                   line=dict(color="#ffb300", width=1.4)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rsi, name=f"RSI {CHART_RSI_PERIOD}",
                   line=dict(color="#ab47bc", width=1.4)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line=dict(color="#787b86", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#787b86", width=1, dash="dot"), row=2, col=1)

    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        uirevision="constant",  # keep zoom/pan across fragment refreshes
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def _render_ladder(ladder: Optional[Dict[str, Any]]) -> None:
    import streamlit as st

    if not ladder:
        st.info("Orderbook preview unavailable - waiting for the first quote.")
        return
    st.caption(
        f"Indicative ladder · {ladder['symbol']} · mid {ladder['mid']} · spread {ladder['spread']} "
        f"· updated {ladder['updated'][11:19]}Z"
    )
    asks = list(reversed(ladder["asks"]))
    st.dataframe(
        pd.DataFrame([{"Ask price": a["price"], "Size": a["size"]} for a in asks]),
        hide_index=True, use_container_width=True,
    )
    st.markdown(
        f"<div style='text-align:center;font-weight:600'>mid {ladder['mid']}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame([{"Bid price": b["price"], "Size": b["size"]} for b in ladder["bids"]]),
        hide_index=True, use_container_width=True,
    )


def _render_council(feed: List[Dict[str, Any]], status: Dict[str, Any]) -> None:
    import streamlit as st

    scout_state = "on" if status.get("scout") else "off"
    auditor_state = "on" if status.get("auditor") else "off"
    scout_chain = " -> ".join(status.get("scout_models", [status.get("scout_model", "-")]))
    auditor_chain = " -> ".join(status.get("auditor_models", [status.get("auditor_model", "-")]))
    st.caption(
        f"Scout (Groq): {scout_state} · cascade `{scout_chain}` · "
        f"Risk Auditor (Gemini): {auditor_state} · cascade `{auditor_chain}` · "
        f"timeout {status.get('timeout', 0):g}s"
    )
    if not status.get("enabled"):
        st.info("Sentinel Council disabled (AI_COUNCIL_ENABLED=false).")
        return
    lines: List[str] = []
    for entry in reversed(feed[-40:]):
        scout = entry.get("scout") or {}
        auditor = entry.get("auditor") or {}
        scout_model = entry.get("scout_model_used") or "—"
        auditor_model = entry.get("auditor_model_used") or "—"
        flag = " ⚠ FALLBACK" if entry.get("fallback_used") else ""
        lines.append(
            f"[{str(entry.get('timestamp', ''))[11:19]}Z] {entry.get('symbol', '')} {entry.get('side', '')} "
            f"-> {entry.get('recommendation', 'neutral')} ({entry.get('confidence', 0.0):.0%}){flag} "
            f"| scout={scout.get('verdict', '-')}@{scout_model} auditor={auditor.get('verdict', '-')}@{auditor_model} "
            f"| {str(entry.get('summary', ''))[:110]}"
        )
    st.code("\n".join(lines) if lines else "No council activity yet.", language="text")


def _render_dashboard() -> None:
    import streamlit as st

    st.set_page_config(page_title="The5ers Multi-Agent Trading Bot", page_icon="🤖", layout="wide")

    config = BotConfig.from_env()
    orchestrator = get_orchestrator(config)
    snapshot = orchestrator.snapshot()
    state = snapshot["state"]
    kill_status = snapshot["kill_switch"]

    # ---- header ---------------------------------------------------------- #
    st.title("🤖 The5ers Multi-Agent Trading Bot")
    banner = "🧪 PAPER MODE" if config.paper_mode else "🟢 LIVE TRADING"
    if config.dry_run:
        banner += " · DRY RUN (no orders sent)"
    st.caption(f"{banner} · The5ers Bootcamp $5,000 · MetaApi / MT5 (async) · Sentinel Council")

    account: Optional[AccountInfo] = state.get("account")
    positions: List[PositionInfo] = state.get("positions") or []
    ks_state, ks_level = _kill_switch_state(kill_status, orchestrator.is_running)

    # ---- sidebar: live refresh & chart ----------------------------------- #
    autorefresh = st.sidebar.toggle("Auto-refresh (5s)", value=True)
    chart_symbols = config.symbols or []
    chart_symbol = st.sidebar.selectbox("Chart symbol", chart_symbols, index=0) if chart_symbols else None

    # ---- sidebar: MetaApi credentials ------------------------------------ #
    env_account = os.getenv("METAAPI_ACCOUNT_ID", "").strip()
    st.session_state.setdefault("ui_token", "")
    st.session_state.setdefault("ui_account_id", "")

    with st.sidebar.expander("🔐 MetaApi Credentials", expanded=not state["connected"]):
        flash = st.session_state.pop("creds_flash", None)
        if flash:
            st.success(flash)
        if config.paper_mode:
            st.info("PAPER_MODE is on - credentials are ignored.")
        ui_token = st.text_input(
            "MetaApi Token",
            key="ui_token",
            type="password",
            placeholder="leave empty to use .env",
            help="Held in this browser session only - never written to disk.",
        )
        ui_account = st.text_input(
            "MetaApi Account ID",
            key="ui_account_id",
            placeholder=env_account or "leave empty to use .env",
        )
        if st.button("💾 Save credentials", use_container_width=True):
            orchestrator.set_credentials(ui_token, ui_account)
            st.session_state["creds_flash"] = (
                "Saved. Stop and Start the bot to apply them."
                if orchestrator.is_running
                else "Credentials saved."
            )
            st.rerun()

        info = snapshot["credentials"]
        st.caption(
            f"**In effect** · token `{info['token_masked'] or '—'}` ({info['token_source']}) · "
            f"account `{info['account_id']}` ({info['account_source']})"
        )

    # ---- sidebar: Sentinel Council AI credentials ------------------------ #
    st.session_state.setdefault("ui_gemini_key", "")
    st.session_state.setdefault("ui_groq_key", "")

    ai_info = snapshot["credentials"]
    gemini_badge = "🟢 set" if ai_info.get("gemini_set") else "🔴 missing"
    groq_badge = "🟢 set" if ai_info.get("groq_set") else "🔴 missing"
    with st.sidebar.expander("🧠 Sentinel Council AI Credentials", expanded=False):
        ai_flash = st.session_state.pop("ai_creds_flash", None)
        if ai_flash:
            st.success(ai_flash)
        st.caption(
            f"Gemini (Risk Auditor): **{gemini_badge}** · {ai_info.get('gemini_source', 'Missing')}  \n"
            f"Groq (Scout): **{groq_badge}** · {ai_info.get('groq_source', 'Missing')}"
        )
        ui_gemini = st.text_input(
            "Gemini API Key",
            key="ui_gemini_key",
            type="password",
            placeholder="leave empty to use secrets/.env",
            help="Risk Auditor model cascade. Held in this browser session only - never written to disk.",
        )
        ui_groq = st.text_input(
            "Groq API Key",
            key="ui_groq_key",
            type="password",
            placeholder="leave empty to use secrets/.env",
            help="Scout model cascade. Held in this browser session only - never written to disk.",
        )
        if st.button("💾 Save AI credentials", use_container_width=True):
            orchestrator.set_credentials(gemini_api_key=ui_gemini, groq_api_key=ui_groq)
            st.session_state["ai_creds_flash"] = (
                "Saved. Stop and Start the bot to apply them."
                if orchestrator.is_running
                else "AI credentials saved."
            )
            st.rerun()

    # ---- sidebar: controls ----------------------------------------------- #
    st.sidebar.header("🎛️ Controls")
    bot_flash = st.session_state.pop("bot_flash", None)
    if bot_flash:
        st.sidebar.success(bot_flash)

    if orchestrator.is_running:
        if st.sidebar.button("⏹ Stop Bot", use_container_width=True, type="primary"):
            orchestrator.stop()
            st.rerun()
    else:
        if st.sidebar.button("▶ Start Bot", use_container_width=True, type="primary"):
            # Credentials are resolved on this (UI) thread and handed to the loop.
            orchestrator.set_credentials(
                st.session_state.get("ui_token", ""),
                st.session_state.get("ui_account_id", ""),
                st.session_state.get("ui_gemini_key", ""),
                st.session_state.get("ui_groq_key", ""),
            )
            started = orchestrator.start()
            # Wait briefly so the loop can connect before the dashboard re-renders,
            # otherwise the first render still shows STOPPED / DISCONNECTED.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if orchestrator.is_running and orchestrator.is_connected:
                    break
                time.sleep(0.1)
            st.session_state["bot_flash"] = "Bot started." if started else "Bot already running."
            st.rerun()

    if st.sidebar.button("🛑 Emergency Close All Positions", use_container_width=True, disabled=not positions):
        with st.spinner("Closing all positions..."):
            results = orchestrator.close_all_now()
        if results:
            st.sidebar.success(f"Closed {len(results)} position(s).")
        else:
            st.sidebar.info("Nothing to close.")

    # ---- sidebar: kill switch -------------------------------------------- #
    st.sidebar.divider()
    st.sidebar.header("🚨 Kill-Switch")
    if ks_level == "emergency":
        st.sidebar.error(f"**{ks_state}**\n\n{kill_status['lock_reason']}")
    elif ks_level == "locked":
        st.sidebar.warning(f"**{ks_state}**\n\n{kill_status['lock_reason']}")
    else:
        st.sidebar.success(f"**{ks_state}** - trading permitted")

    confirm_reset = st.sidebar.checkbox(
        "I understand this re-arms the day from the current balance", value=False
    )
    if st.sidebar.button("🔓 Reset Kill-Switch", use_container_width=True, disabled=not confirm_reset):
        balance = account.balance if account is not None else None
        decision_reset = orchestrator.kill_switch.reset(
            current_balance=balance, reason="Dashboard operator reset.", force=True
        )
        orchestrator.log_event(f"Kill switch reset from dashboard: {decision_reset.reason}")
        st.rerun()

    # ---- live dashboard (native fragment refresh) ------------------------ #
    interval = DASHBOARD_REFRESH_SECONDS if autorefresh else None

    @st.fragment(run_every=interval)
    def _live_dashboard() -> None:
        snap = orchestrator.snapshot()
        live_state = snap["state"]
        live_kill = snap["kill_switch"]
        live_account: Optional[AccountInfo] = live_state.get("account")
        live_decision: Optional[KillSwitchDecision] = live_state.get("decision")
        live_positions: List[PositionInfo] = live_state.get("positions") or []
        live_ks_state, live_ks_level = _kill_switch_state(live_kill, live_state.get("running", False))

        # KPI bar -------------------------------------------------------- #
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity", f"${live_account.equity:,.2f}" if live_account else "—")
        c2.metric(
            "Daily P&L",
            f"${live_decision.daily_pnl:+,.2f}" if live_decision else "—",
            f"target +$60 · stop -$100",
            delta_color="normal" if (live_decision and live_decision.daily_pnl >= 0) else "inverse",
        )
        c3.metric("RRR floor", f"1 : {MIN_REWARD_RISK:g}", "MIN_REWARD_RISK")
        c4.metric("Kill-Switch", live_ks_state)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Balance", f"${live_account.balance:,.2f}" if live_account else "—")
        r2.metric("Open positions", f"{len(live_positions)} / {config.max_open_positions}")
        r3.metric("Loop", "RUNNING" if live_state.get("running") else "STOPPED")
        r4.metric("Broker", "CONNECTED" if live_state.get("connected") else "DISCONNECTED")

        if live_decision is not None:
            st.progress(
                live_decision.progress_to_target,
                text=(f"Progress to the +${live_decision.profit_target:,.0f} daily cap · "
                      f"current {live_decision.daily_pnl:+.2f} · remaining +{live_decision.remaining_to_target:,.2f} · "
                      f"risk budget left ${live_decision.remaining_risk:,.2f} of -${live_decision.max_daily_loss:,.0f}"),
            )
        else:
            st.caption(
                f"Waiting for the first tick to measure daily P&L against the "
                f"+${live_kill['profit_target']:,.0f} target and -${live_kill['max_daily_loss']:,.0f} stop."
            )

        if live_ks_level == "emergency":
            st.error(f"🚨 EMERGENCY STOP - {live_kill['lock_reason']}")
        elif live_ks_level == "locked":
            st.warning(f"🔒 Kill-switch locked - {live_kill['lock_reason']}")
        if live_state.get("last_error"):
            st.error(f"Last error: {live_state['last_error']}")

        # Price chart ---------------------------------------------------- #
        st.subheader(f"📈 {chart_symbol or 'Market'} · EMA {CHART_EMA_FAST}/{CHART_EMA_SLOW} + RSI {CHART_RSI_PERIOD}")
        frames = snap.get("frames") or {}
        frame = frames.get(chart_symbol) if chart_symbol else None
        if frame is None and frames:
            frame = next(iter(frames.values()), None)
        figure = _build_candlestick(frame, chart_symbol or "market") if frame is not None else None
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True, config={"scrollZoom": True})
        else:
            st.info("Chart data will appear after the first tick (or install `plotly`).")

        # Orderbook ladder + positions ----------------------------------- #
        lcol, pcol = st.columns([1, 2])
        with lcol:
            st.subheader("📚 Orderbook ladder")
            _render_ladder(snap.get("ladder"))
        with pcol:
            st.subheader("📋 Active positions")
            if live_positions:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Symbol": p.symbol,
                                "Side": p.side.upper(),
                                "Volume": p.volume,
                                "Open": p.open_price,
                                "Current": p.current_price,
                                "SL": p.stop_loss,
                                "TP": p.take_profit,
                                "P&L (USD)": round(p.profit, 2),
                                "Opened": (p.time or "")[:19],
                            }
                            for p in live_positions
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No open positions.")

        # Sentinel Council ----------------------------------------------- #
        st.subheader("🛰️ Sentinel Council")
        _render_council(snap.get("council") or [], snap.get("council_status") or {})

        # Signals & trades ----------------------------------------------- #
        left, right = st.columns(2)
        with left:
            st.subheader("Recent signals")
            if snap["signals"]:
                st.dataframe(pd.DataFrame(list(reversed(snap["signals"]))), use_container_width=True, hide_index=True)
            else:
                st.info("No signals yet.")
        with right:
            st.subheader("Trade log")
            if snap["trades"]:
                st.dataframe(pd.DataFrame(list(reversed(snap["trades"]))), use_container_width=True, hide_index=True)
            else:
                st.info("No trades yet.")

        # Config & events ------------------------------------------------ #
        with st.expander("Configuration"):
            credentials = snap["credentials"]
            settings = {
                **config.summary(),
                "credentials.token": credentials["token_masked"] or "(not set)",
                "credentials.token source": credentials["token_source"],
                "credentials.account id": credentials["account_id"],
                "credentials.account source": credentials["account_source"],
                **{f"kill_switch.{k}": v for k, v in live_kill.items()},
            }
            st.table(pd.DataFrame({"Setting": list(settings), "Value": [str(v) for v in settings.values()]}))

        with st.expander("Event log", expanded=False):
            events = snap["events"]
            st.code("\n".join(reversed(events[-200:])) if events else "No events yet.", language="text")

    _live_dashboard()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _render_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The5ers multi-agent trading bot.")
    parser.add_argument("--once", action="store_true", help="run a single headless pass and exit")
    args, _unknown = parser.parse_known_args()

    if args.once:
        raise SystemExit(run_single_pass())
    main()
