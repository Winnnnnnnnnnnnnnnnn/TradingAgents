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

The whole loop runs inside a dedicated background thread with its own asyncio
event loop, so the Streamlit UI can be re-rendered without disturbing it.

Run it:
    streamlit run app.py                     # dashboard
    python app.py --once                     # single headless pass (smoke test)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from broker_client import AccountInfo, BrokerClient, OrderSide, PositionInfo, TradeResult
from kill_switch import KillSwitch, KillSwitchDecision

load_dotenv()

LOGGER = logging.getLogger("the5ers.bot")


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
    reward_risk: float = 1.8
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 2

    dry_run: bool = False
    paper_mode: bool = False

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
            reward_risk=_as_float(os.getenv("REWARD_RISK"), 1.8),
            risk_per_trade_pct=_as_float(os.getenv("RISK_PER_TRADE_PCT"), 0.5),
            max_open_positions=_as_int(os.getenv("MAX_OPEN_POSITIONS"), 2),
            dry_run=_as_bool(os.getenv("DRY_RUN"), False),
            paper_mode=_as_bool(os.getenv("PAPER_MODE"), False),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "Symbols": ", ".join(self.symbols),
            "Timeframe": self.timeframe,
            "Poll interval (s)": self.poll_interval_seconds,
            "EMA fast / slow": f"{self.ema_fast} / {self.ema_slow}",
            "Risk per trade": f"{self.risk_per_trade_pct}% of equity",
            "Stop distance": f"{self.atr_stop_multiplier} x ATR({self.atr_period})",
            "Reward : risk": self.reward_risk,
            "Max open positions": self.max_open_positions,
            "Paper mode": self.paper_mode,
            "Dry run": self.dry_run,
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

        # Credentials can be supplied at runtime from the dashboard. Empty values
        # fall back to METAAPI_TOKEN / METAAPI_ACCOUNT_ID from the environment.
        self._credentials: Dict[str, str] = {"token": "", "account_id": ""}

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._broker: Optional[BrokerClient] = None
        self._market_agent: Optional[MarketDataAgent] = None
        self._strategy_agent: Optional[StrategyAgent] = None
        self._risk_agent: Optional[RiskAgent] = None
        self._execution_agent: Optional[ExecutionAgent] = None

        self._signals: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._trades: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._events: Deque[str] = deque(maxlen=500)

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
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    def log_event(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}Z] {message}"
        with self._lock:
            self._events.append(line)
        LOGGER.info(message)

    def set_credentials(self, token: Optional[str] = None, account_id: Optional[str] = None) -> bool:
        """Store dashboard-supplied credentials for the next connection attempt.

        Blank values are stored as empty strings, which makes ``BrokerClient``
        fall back to ``METAAPI_TOKEN`` / ``METAAPI_ACCOUNT_ID`` from the
        environment. Returns True when the stored values actually changed.
        """
        token = (token or "").strip()
        account_id = (account_id or "").strip()
        with self._lock:
            changed = (token, account_id) != (self._credentials["token"], self._credentials["account_id"])
            self._credentials = {"token": token, "account_id": account_id}
        if changed:
            source = "dashboard" if (token or account_id) else "environment"
            self.log_event(f"Credentials updated (source: {source}). Restart the bot to apply them.")
        return changed

    def _effective_credentials(self) -> Dict[str, str]:
        """Resolve dashboard credentials, falling back to the environment."""
        with self._lock:
            ui_token, ui_account = self._credentials["token"], self._credentials["account_id"]
        return {
            "token": ui_token or os.getenv("METAAPI_TOKEN", "").strip(),
            "account_id": ui_account or os.getenv("METAAPI_ACCOUNT_ID", "").strip(),
        }

    def credentials_info(self) -> Dict[str, Any]:
        """Describe which credentials the bot would use, with the token masked."""
        with self._lock:
            ui_token, ui_account = self._credentials["token"], self._credentials["account_id"]
        env_token, env_account = os.getenv("METAAPI_TOKEN", "").strip(), os.getenv("METAAPI_ACCOUNT_ID", "").strip()
        effective = self._effective_credentials()
        return {
            "token_masked": _mask(effective["token"]),
            "token_source": "Dashboard" if ui_token else ("Environment" if env_token else "Missing"),
            "account_id": effective["account_id"] or "(not set)",
            "account_source": "Dashboard" if ui_account else ("Environment" if env_account else "Missing"),
        }

    def start(self) -> bool:
        """Start the background trading loop. Returns False if already running."""
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
        """Thread-safe copy of everything the dashboard needs."""
        with self._lock:
            return {
                "state": dict(self._state),
                "signals": list(self._signals),
                "trades": list(self._trades),
                "events": list(self._events),
                "kill_switch": self.kill_switch.status(),
                "credentials": self.credentials_info(),
            }

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
        creds = self._effective_credentials()

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
            await asyncio.sleep(max(1.0, self.config.poll_interval_seconds - elapsed))

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
        open_symbols = {p.symbol for p in positions}
        for symbol in self.config.symbols:
            resolved = self._broker.resolve_symbol(symbol)
            if resolved in open_symbols:
                continue

            frame = await self._market_agent.fetch_frame(symbol)
            if frame.empty:
                continue
            frame = self._market_agent.add_indicators(frame)
            signal = self._strategy_agent.generate_signal(symbol, frame)
            if signal is None:
                continue

            self._record_signal(signal)

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
                entry = self.kill_switch.status()
                self.log_event(
                    f"Opened {signal.side.value.upper()} {risk.volume} {risk.symbol} @ {risk.entry_price} "
                    f"(SL {risk.stop_loss}, TP {risk.take_profit}, risk {risk.risk_usd:.2f} USD) | "
                    f"day P&L {decision.daily_pnl:+.2f} USD, trades today {entry['trades_today']}"
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
# Headless single pass (smoke test / cron usage)
# --------------------------------------------------------------------------- #
def run_single_pass() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = BotConfig.from_env()
    orchestrator = TradingOrchestrator(config)

    async def _one_pass() -> None:
        creds = orchestrator._effective_credentials()
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
def _kill_switch_state(kill_status: Dict[str, Any], running: bool) -> tuple:
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


def _render_dashboard() -> None:
    import streamlit as st

    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:  # graceful degradation: manual refresh only

        def st_autorefresh(**_kwargs: Any) -> int:
            return 0

    st.set_page_config(page_title="The5ers Multi-Agent Trading Bot", page_icon="🤖", layout="wide")

    config = BotConfig.from_env()
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = TradingOrchestrator(config)
    orchestrator: TradingOrchestrator = st.session_state.orchestrator
    snapshot = orchestrator.snapshot()
    state = snapshot["state"]
    kill_status = snapshot["kill_switch"]

    # ---- header ---------------------------------------------------------- #
    st.title("🤖 The5ers Multi-Agent Trading Bot")
    banner = "🟢 LIVE TRADING" if not config.paper_mode else "🧪 PAPER MODE"
    if config.dry_run:
        banner += " · DRY RUN (no orders sent)"
    st.caption(f"{banner} · The5ers Bootcamp $5,000 · MetaApi / MT5 (async)")

    account: Optional[AccountInfo] = state.get("account")
    decision: Optional[KillSwitchDecision] = state.get("decision")
    positions: List[PositionInfo] = state.get("positions") or []
    ks_state, ks_level = _kill_switch_state(kill_status, orchestrator.is_running)

    autorefresh = st.sidebar.toggle("Auto-refresh (5s)", value=True)
    if autorefresh:
        st_autorefresh(interval=5000, key="the5ers-autorefresh")

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

    # ---- sidebar: controls ----------------------------------------------- #
    st.sidebar.header("🎛️ Controls")
    if orchestrator.is_running:
        if st.sidebar.button("⏹ Stop Bot", use_container_width=True, type="primary"):
            orchestrator.stop()
            st.rerun()
    else:
        if st.sidebar.button("▶ Start Bot", use_container_width=True, type="primary"):
            # Apply whatever is currently typed in the credential fields.
            orchestrator.set_credentials(st.session_state.get("ui_token", ""), st.session_state.get("ui_account_id", ""))
            orchestrator.start()
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

    # ---- metric cards ---------------------------------------------------- #
    st.subheader("📊 Daily performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Daily P&L",
        f"${decision.daily_pnl:+,.2f}" if decision else "—",
        f"{account.floating_pnl:+,.2f} floating" if account else None,
    )
    c2.metric("Balance", f"${account.balance:,.2f}" if account else "—")
    c3.metric("Equity", f"${account.equity:,.2f}" if account else "—")
    c4.metric("Kill-Switch", ks_state)

    if decision is not None:
        st.progress(
            decision.progress_to_target,
            text=(f"Progress to the +${decision.profit_target:,.0f} daily cap · "
                  f"current {decision.daily_pnl:+.2f} · remaining +{decision.remaining_to_target:,.2f} · "
                  f"risk budget left ${decision.remaining_risk:,.2f} of -${decision.max_daily_loss:,.0f}"),
        )
    else:
        st.caption(
            f"Waiting for the first tick to measure daily P&L against the "
            f"+${kill_status['profit_target']:,.0f} target and -${kill_status['max_daily_loss']:,.0f} stop."
        )

    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Open positions", f"{len(positions)} / {config.max_open_positions}")
    o2.metric("Trades today", kill_status["trades_today"])
    o3.metric("Loop", "RUNNING" if state["running"] else "STOPPED")
    o4.metric("Broker", "CONNECTED" if state["connected"] else "DISCONNECTED")
    o5.metric("Last tick", (state["last_tick"] or "—")[11:19] or "—")

    if ks_level == "emergency":
        st.error(f"🚨 EMERGENCY STOP - {kill_status['lock_reason']}")
    elif ks_level == "locked":
        st.warning(f"🔒 Kill-switch locked - {kill_status['lock_reason']}")
    if state.get("last_error"):
        st.error(f"Last error: {state['last_error']}")

    # ---- positions ------------------------------------------------------- #
    st.subheader("Open positions")
    if positions:
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
                    for p in positions
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No open positions.")

    # ---- signals & trades ------------------------------------------------ #
    left, right = st.columns(2)
    with left:
        st.subheader("Recent signals")
        if snapshot["signals"]:
            st.dataframe(pd.DataFrame(list(reversed(snapshot["signals"]))), use_container_width=True, hide_index=True)
        else:
            st.info("No signals yet.")
    with right:
        st.subheader("Trade log")
        if snapshot["trades"]:
            st.dataframe(pd.DataFrame(list(reversed(snapshot["trades"]))), use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet.")

    # ---- config & events ------------------------------------------------- #
    with st.expander("Configuration"):
        credentials = snapshot["credentials"]
        settings = {
            **config.summary(),
            "credentials.token": credentials["token_masked"] or "(not set)",
            "credentials.token source": credentials["token_source"],
            "credentials.account id": credentials["account_id"],
            "credentials.account source": credentials["account_source"],
            **{f"kill_switch.{k}": v for k, v in kill_status.items()},
        }
        st.table(pd.DataFrame({"Setting": list(settings), "Value": [str(v) for v in settings.values()]}))

    with st.expander("Event log", expanded=False):
        events = snapshot["events"]
        st.code("\n".join(reversed(events[-200:])) if events else "No events yet.", language="text")


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
