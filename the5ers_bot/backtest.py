"""
backtest.py
===========
Standalone historical backtest for the The5ers Bootcamp multi-agent bot.

This script reuses the *exact* production strategy and risk logic:

* indicators ............ ``app.MarketDataAgent.add_indicators``   (EMA / RSI / ATR)
* entry signal .......... ``app.StrategyAgent.generate_signal``    (EMA cross + RSI filter)
* Triple-Lock governor .. ``kill_switch.KillSwitch.evaluate``      (+$60 target / -$100 max loss)

so the simulation cannot drift from what the live bot does. Nothing in
``app.py`` / ``kill_switch.py`` / ``broker_client.py`` is modified.

Data
----
Tries ``yfinance`` for recent intraday candles (default: last 45 days of 15m bars
for XAUUSD / BTCUSD / EURUSD). When the network or the package is unavailable it
falls back to a deterministic, realistic simulated OHLCV generator (geometric
Brownian motion with per-symbol volatility), so the script always runs offline.

Model
-----
Trades are sized so that a full stop risks ``RISK_PER_TRADE_PCT`` of equity (capped
by the remaining daily loss budget), exactly like ``RiskAgent``. P&L is expressed in
R-multiples derived from the ATR stop distance, which preserves the strict 1:3
reward:risk floor: a stop is ``-1R`` and a target is ``+REWARD_RISK*R`` (>= +3R).

Run
---
    python the5ers_bot/backtest.py
    python the5ers_bot/backtest.py --days 60 --interval 1h --symbols BTCUSD,EURUSD
    python the5ers_bot/backtest.py --offline          # force the simulated feed
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Same bootstrap as app.py: keep sibling modules importable from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from app import MIN_REWARD_RISK, BotConfig, MarketDataAgent, StrategyAgent
from broker_client import OrderSide
from kill_switch import KillLockType, KillSwitch

# The kill switch logs every engagement; keep the console output to the report only.
for _noisy in ("kill_switch", "the5ers.bot", "broker_client"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# --------------------------------------------------------------------------- #
# Constants / defaults
# --------------------------------------------------------------------------- #
INITIAL_EQUITY = 5_000.0          # The5ers Bootcamp $5k
PROFIT_TARGET = 60.0              # Lock 1: bank the win at +$60
MAX_DAILY_LOSS = 100.0            # Lock 2: hard stop at -$100
DEFAULT_SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD"]
DEFAULT_INTERVAL = "15m"
DEFAULT_DAYS = 45
DEFAULT_SEED = 7

# yfinance tickers for the requested instruments (fallback: the symbol itself).
YF_TICKERS = {
    "XAUUSD": "GC=F",       # COMEX gold futures proxy for spot XAUUSD
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "EURUSD": "EURUSD=X",
}

# Realistic per-bar volatility (std of log-returns) + anchor price for the
# simulated feed, so the offline run still produces plausible movement.
SIM_PROFILE = {
    "XAUUSD": (0.00060, 2350.0),
    "BTCUSD": (0.00320, 65000.0),
    "ETHUSD": (0.00380, 3200.0),
    "EURUSD": (0.00035, 1.0850),
}
SIM_DEFAULT = (0.00150, 100.0)

VALID_INTERVALS = {"15m": 15, "30m": 30, "1h": 60}


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #
def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw OHLCV frame into lower-case columns with a UTC DatetimeIndex."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns={c: str(c).lower() for c in frame.columns})
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "date" in frame.columns:
            frame = frame.set_index("date")
        elif "datetime" in frame.columns:
            frame = frame.set_index("datetime")
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if getattr(frame.index, "tz", None) is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
    frame = frame[keep]
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return frame


def fetch_yfinance(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch intraday candles from yfinance. Raises on any failure."""
    import yfinance as yf  # imported lazily so the module loads offline

    ticker = YF_TICKERS.get(symbol, symbol)
    raw = yf.download(
        tickers=ticker,
        period=f"{days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv(raw)


def simulate_ohlcv(symbol: str, interval: str, days: int, seed: int) -> pd.DataFrame:
    """Deterministic GBM-based OHLCV generator used when yfinance is unavailable."""
    minutes = VALID_INTERVALS[interval]
    bars = int(days * 24 * 60 / minutes)
    sigma, start_price = SIM_PROFILE.get(symbol, SIM_DEFAULT)

    rng = np.random.default_rng(seed + (zlib.crc32(symbol.encode()) % 10_000))
    end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    index = pd.date_range(end=end, periods=bars, freq=f"{minutes}min", tz="UTC")

    shocks = rng.standard_normal(bars)
    log_returns = sigma * shocks - 0.5 * sigma**2
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty(bars)
    open_[0] = start_price
    open_[1:] = close[:-1]

    wick = np.abs(rng.standard_normal(bars)) * sigma * 0.6
    high = np.maximum(open_, close) * np.exp(wick)
    low = np.minimum(open_, close) * np.exp(-wick)
    volume = np.abs(rng.standard_normal(bars)) * 1_000.0 + 100.0

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def load_frames(symbols: List[str], interval: str, days: int, seed: int, offline: bool) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Load OHLCV for every symbol, returning the frames and the data source label."""
    frames: Dict[str, pd.DataFrame] = {}
    source = "yfinance"

    if not offline:
        for symbol in symbols:
            try:
                frame = fetch_yfinance(symbol, interval, days)
                if len(frame) >= 100:
                    frames[symbol] = frame
                    print(f"  · {symbol:<7} fetched {len(frame):>5} real candles from yfinance")
                else:
                    print(f"  · {symbol:<7} yfinance returned too little data - using simulation")
            except Exception as exc:  # noqa: BLE001 - any network/parse issue falls back
                print(f"  · {symbol:<7} yfinance unavailable ({type(exc).__name__}) - using simulation")

    missing = [s for s in symbols if s not in frames]
    if missing:
        source = "simulated" if not frames else "mixed (yfinance + simulated)"
        for symbol in missing:
            frames[symbol] = simulate_ohlcv(symbol, interval, days, seed)
            print(f"  · {symbol:<7} simulated {len(frames[symbol]):>5} candles")
    return frames, source


# --------------------------------------------------------------------------- #
# Positions & P&L
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    symbol: str
    side: OrderSide
    entry: float
    stop: float
    target: float
    stop_distance: float
    risk_usd: float
    opened_at: Any


@dataclass
class Trade:
    symbol: str
    side: OrderSide
    entry: float
    exit: float
    pnl: float
    reason: str
    opened_at: Any
    closed_at: Any


def pnl_usd(position: Position, exit_price: float) -> float:
    """USD P&L for a given exit, expressed in R-multiples of the ATR stop."""
    direction = 1.0 if position.side.is_long else -1.0
    move = (exit_price - position.entry) * direction
    return move / position.stop_distance * position.risk_usd


def check_exit(position: Position, high: float, low: float) -> Tuple[Optional[float], Optional[str]]:
    """Return (exit_price, reason) if the bar hit the stop or target, else (None, None)."""
    if position.side.is_long:
        if low <= position.stop:            # conservative: stop assumed first
            return position.stop, "stop"
        if high >= position.target:
            return position.target, "target"
    else:
        if high >= position.stop:
            return position.stop, "stop"
        if low <= position.target:
            return position.target, "target"
    return None, None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    closed: List[Trade]
    equity_curve: List[float]
    final_equity: float
    days_locked_target: int
    days_stopped_loss: int
    bars_by_symbol: Dict[str, int]
    period_start: Any
    period_end: Any


def run_backtest(
    cfg: BotConfig,
    frames: Dict[str, pd.DataFrame],
    initial_equity: float = INITIAL_EQUITY,
    profit_target: float = PROFIT_TARGET,
    max_daily_loss: float = MAX_DAILY_LOSS,
) -> BacktestResult:
    strategy = StrategyAgent(cfg)
    data_agent = MarketDataAgent(broker=None, config=cfg)  # add_indicators needs config only
    required = max(cfg.ema_slow, cfg.rsi_period, cfg.atr_period) + 2

    # ---- indicators + exact signal logic, per symbol --------------------- #
    signals: Dict[str, Dict[Any, Any]] = {}
    tagged: Dict[str, pd.DataFrame] = {}          # frame with indicator columns
    events: List[Tuple[Any, str, int]] = []

    for symbol, frame in frames.items():
        enriched = data_agent.add_indicators(frame)
        tagged[symbol] = enriched
        series = enriched.set_index(enriched.index)
        found: Dict[Any, Any] = {}
        for i in range(required - 1, len(series)):
            window = series.iloc[i - required + 1 : i + 1]
            signal = strategy.generate_signal(symbol, window)
            if signal is not None:
                found[series.index[i]] = signal
        signals[symbol] = found
        for i in range(len(series)):
            events.append((series.index[i], symbol, i))

    events.sort(key=lambda item: item[0])

    # ---- kill switch (fresh, isolated state) ------------------------------ #
    tmpdir = tempfile.mkdtemp(prefix="the5ers_backtest_")
    state_path = os.path.join(tmpdir, "kill_switch_state.json")
    kill_switch = KillSwitch(
        state_file=state_path,
        profit_target=profit_target,
        max_daily_loss=max_daily_loss,
        auto_close_on_lock=True,
    )
    kill_switch.set_account_scope("backtest")

    balance = initial_equity
    positions: Dict[str, Position] = {}
    last_price: Dict[str, float] = {}
    closed: List[Trade] = []
    equity_curve: List[float] = [initial_equity]
    locked_days_target: set = set()
    locked_days_loss: set = set()
    flattened_days: set = set()
    reward_risk = max(MIN_REWARD_RISK, cfg.reward_risk)

    def close_position(position: Position, exit_price: float, reason: str, when: Any) -> None:
        nonlocal balance
        realized = pnl_usd(position, exit_price)
        balance += realized
        closed.append(
            Trade(position.symbol, position.side, position.entry, exit_price,
                  realized, reason, position.opened_at, when)
        )

    try:
        for ts, symbol, idx in events:
            row = tagged[symbol].iloc[idx]
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
            last_price[symbol] = close

            # 1) manage an existing position on this symbol
            position = positions.get(symbol)
            if position is not None:
                exit_price, reason = check_exit(position, high, low)
                if exit_price is not None:
                    close_position(position, exit_price, reason, ts)
                    del positions[symbol]

            # 2) mark-to-market equity (realised + floating), as the kill switch demands
            floating = sum(pnl_usd(p, last_price[p.symbol]) for p in positions.values())
            equity = balance + floating

            # 3) Triple-Lock governor against the live equity snapshot
            decision = kill_switch.evaluate(equity=equity, balance=balance, now=ts)
            day = ts.date()

            if not decision.allowed:
                if decision.lock_type is KillLockType.PROFIT_TARGET:
                    locked_days_target.add(day)
                elif decision.lock_type is KillLockType.MAX_DAILY_LOSS:
                    locked_days_loss.add(day)
                # flatten once per locked day, then stand down until rollover
                if day not in flattened_days:
                    for held in list(positions):
                        close_position(positions[held], last_price[held], "kill-switch flatten", ts)
                        del positions[held]
                    flattened_days.add(day)
                equity_curve.append(balance)
                continue

            # 4) open a new position on a fresh signal (respecting capacity)
            signal = signals.get(symbol, {}).get(ts)
            if (
                signal is not None
                and symbol not in positions
                and len(positions) < cfg.max_open_positions
            ):
                stop_distance = signal.atr * cfg.atr_stop_multiplier
                if stop_distance > 0:
                    risk_budget = equity * (cfg.risk_per_trade_pct / 100.0)
                    if decision.remaining_risk > 0:
                        risk_budget = min(risk_budget, decision.remaining_risk)
                    if risk_budget > 0:
                        direction = 1.0 if signal.side.is_long else -1.0
                        positions[symbol] = Position(
                            symbol=symbol,
                            side=signal.side,
                            entry=close,
                            stop=close - direction * stop_distance,
                            target=close + direction * stop_distance * reward_risk,
                            stop_distance=stop_distance,
                            risk_usd=round(risk_budget, 2),
                            opened_at=ts,
                        )

            equity_curve.append(balance + sum(pnl_usd(p, last_price[p.symbol]) for p in positions.values()))

        # close whatever is still open at the final available price
        if positions:
            final_ts = events[-1][0]
            for held in list(positions):
                close_position(positions[held], last_price[held], "end of backtest", final_ts)
                del positions[held]
            equity_curve.append(balance)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return BacktestResult(
        closed=closed,
        equity_curve=equity_curve,
        final_equity=balance,
        days_locked_target=len(locked_days_target),
        days_stopped_loss=len(locked_days_loss),
        bars_by_symbol={s: len(f) for s, f in frames.items()},
        period_start=events[0][0] if events else None,
        period_end=events[-1][0] if events else None,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def max_drawdown(equity_curve: List[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def summarize(result: BacktestResult, initial_equity: float, cfg: BotConfig, source: str) -> None:
    trades = result.closed
    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    win_rate = (len(wins) / total * 100.0) if total else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    net = result.final_equity - initial_equity
    net_pct = net / initial_equity * 100.0
    dd = max_drawdown(result.equity_curve)
    period = f"{result.period_start:%Y-%m-%d} -> {result.period_end:%Y-%m-%d}" if result.period_start else "n/a"
    bars = ", ".join(f"{s}:{n}" for s, n in result.bars_by_symbol.items())

    def money(value: float) -> str:
        return f"${value:,.2f}"

    print()
    print("=" * 74)
    print(f"  The5ers Bootcamp $5k — strategy & Triple-Lock backtest")
    print("=" * 74)
    print(f"  Data source          : {source}")
    print(f"  Symbols              : {', '.join(cfg.symbols)}")
    print(f"  Timeframe            : {cfg.timeframe}   (bars: {bars})")
    print(f"  Period               : {period}")
    print(f"  Initial equity       : {money(initial_equity)}")
    print(f"  Reward:risk floor    : 1:{max(MIN_REWARD_RISK, cfg.reward_risk):g} (MIN_REWARD_RISK={MIN_REWARD_RISK:g})")
    print(f"  Profit-target lock   : +{money(PROFIT_TARGET)}")
    print(f"  Max-drawdown lock    : -{money(MAX_DAILY_LOSS)}")
    print("-" * 74)
    print(f"  Total trades         : {total}")
    print(f"  Win rate             : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit factor        : {'inf' if math.isinf(profit_factor) else f'{profit_factor:.2f}'}")
    print(f"  Max drawdown         : {money(dd)}")
    print(f"  Days locked (+$60)   : {result.days_locked_target}")
    print(f"  Days stopped (-$100) : {result.days_stopped_loss}")
    print(f"  Final equity         : {money(result.final_equity)}")
    print("-" * 74)
    print(f"  Net P&L              : {net:+,.2f} USD  ({net_pct:+.2f}%)")
    if wins:
        print(f"  Avg win / avg loss   : {money(gross_profit / len(wins))} / {money(gross_loss / len(losses)) if losses else '$0.00'}")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace, symbols: List[str]) -> BotConfig:
    cfg = BotConfig.from_env()
    cfg.symbols = symbols
    cfg.timeframe = args.interval
    cfg.reward_risk = max(MIN_REWARD_RISK, cfg.reward_risk)   # enforce the 1:3 floor
    cfg.max_open_positions = max(1, cfg.max_open_positions)
    return cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical backtest for the The5ers bot.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help=f"comma-separated instruments (default: {','.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, choices=sorted(VALID_INTERVALS),
                        help="candle timeframe (default: 15m)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help="look-back window in days (default: 45; yfinance intraday max ~60)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed for the simulated feed (default: 7)")
    parser.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY,
                        help="starting equity (default: 5000)")
    parser.add_argument("--profit-target", type=float, default=PROFIT_TARGET,
                        help="daily profit-target lock in USD (default: 60)")
    parser.add_argument("--max-loss", type=float, default=MAX_DAILY_LOSS,
                        help="daily max-loss lock in USD (default: 100)")
    parser.add_argument("--offline", action="store_true",
                        help="skip yfinance and use the simulated feed")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(DEFAULT_SYMBOLS)
    cfg = build_config(args, symbols)

    print(f"Loading {args.interval} candles for {', '.join(symbols)} "
          f"(last {args.days} days, {'offline' if args.offline else 'yfinance'} mode)...")
    frames, source = load_frames(symbols, args.interval, args.days, args.seed, args.offline)
    if not frames or all(frame.empty for frame in frames.values()):
        print("No data available - aborting.")
        return 1

    result = run_backtest(
        cfg, frames,
        initial_equity=args.initial_equity,
        profit_target=args.profit_target,
        max_daily_loss=args.max_loss,
    )
    summarize(result, args.initial_equity, cfg, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
