"""
broker_client.py
================
Async broker abstraction layer for the The5ers Bootcamp multi-agent trading bot.

Two execution back-ends are provided behind a single interface:

1. ``MetaApi Cloud SDK`` (production default) - talks to a MetaTrader 4/5 account
   deployed on MetaApi's cloud infrastructure. This is the recommended path for a
   The5ers account because the bot keeps working when your local machine is off.
2. ``Paper mode`` (``PAPER_MODE=true``) - a fully offline simulator used for smoke
   tests, CI and dashboard demos. It requires no credentials and no network.

Every credential is read from environment variables - nothing is hard-coded.

Environment variables
---------------------
METAAPI_TOKEN          MetaApi authentication token (required outside paper mode)
METAAPI_ACCOUNT_ID     MetaApi account id, e.g. "1f2ab3cd-..." (or leave empty to
                       create/reuse an account from raw MT5 credentials below)
METAAPI_PLATFORM       "mt5" (default) or "mt4"
METAAPI_REGION         optional MetaApi region, e.g. "new-york"
MT5_LOGIN              raw MT5 login, used to provision a MetaApi account
MT5_PASSWORD           raw MT5 investor/main password
MT5_SERVER             broker server name, e.g. "The5ers-Live"
MT5_MAGIC              magic number stamped on every order (default 520052)
SYMBOL_SUFFIX          appended to every symbol, e.g. ".a" for "BTCUSD.a"
PAPER_MODE             "true" to run the offline simulator
PAPER_START_BALANCE    starting balance for paper mode (default 5000)
PAPER_BASE_PRICES      "BTCUSD:65000,ETHUSD:3200"

Usage
-----
    async with BrokerClient() as broker:
        account = await broker.get_account_information()
        await broker.open_market_order("BTCUSD", OrderSide.BUY, 0.01,
                                       stop_loss=account.equity * 0.99)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)

try:  # The SDK is optional so the module can be imported in paper mode / CI.
    from metaapi_cloud_sdk import MetaApi  # type: ignore

    METAAPI_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when the SDK is absent
    MetaApi = None  # type: ignore
    METAAPI_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among camelCase/snake_case keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _with_retry(
    attempts: int = 3,
    base_delay: float = 1.0,
    description: str = "",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Exponential-backoff retry decorator for flaky network calls."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - retried then re-raised
                    last_exc = exc
                    if attempt == attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    LOGGER.warning(
                        "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                        description or func.__name__, attempt, attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
class OrderSide(str, Enum):
    """Direction of a market order."""

    BUY = "buy"
    SELL = "sell"

    @property
    def is_long(self) -> bool:
        return self is OrderSide.BUY

    @property
    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self.is_long else OrderSide.BUY


@dataclass
class AccountInfo:
    """A normalised snapshot of the trading account."""

    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    currency: str = "USD"
    leverage: float = 0.0
    login: Optional[int] = None
    broker: str = ""
    server: str = ""
    platform: str = "mt5"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def floating_pnl(self) -> float:
        return round(self.equity - self.balance, 8)

    @classmethod
    def from_raw(cls, data: Dict[str, Any], platform: str = "mt5") -> "AccountInfo":
        data = data or {}
        return cls(
            balance=_as_float(_pick(data, "balance")),
            equity=_as_float(_pick(data, "equity")),
            margin=_as_float(_pick(data, "margin")),
            free_margin=_as_float(_pick(data, "freeMargin", "free_margin")),
            margin_level=_as_float(_pick(data, "marginLevel", "margin_level")),
            currency=str(_pick(data, "currency", default="USD") or "USD"),
            leverage=_as_float(_pick(data, "leverage")),
            login=_pick(data, "login"),
            broker=str(_pick(data, "broker", "brokerName", default="") or ""),
            server=str(_pick(data, "server", "serverName", default="") or ""),
            platform=platform,
            raw=dict(data),
        )


@dataclass
class SymbolSpecification:
    """Contract details required for correct position sizing."""

    symbol: str
    contract_size: float = 1.0
    tick_size: float = 0.01
    tick_value: float = 0.01
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    digits: int = 2
    currency_profit: str = "USD"
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, symbol: str, data: Dict[str, Any]) -> "SymbolSpecification":
        data = data or {}
        tick_size = _as_float(_pick(data, "tickSize", "tick_size"), 0.01) or 0.01
        tick_value = _as_float(_pick(data, "tickValue", "tick_value"), 0.01) or 0.01
        return cls(
            symbol=symbol,
            contract_size=_as_float(_pick(data, "contractSize", "contract_size"), 1.0) or 1.0,
            tick_size=tick_size,
            tick_value=tick_value,
            min_volume=_as_float(_pick(data, "minVolume", "min_volume"), 0.01) or 0.01,
            max_volume=_as_float(_pick(data, "maxVolume", "max_volume"), 100.0) or 100.0,
            volume_step=_as_float(_pick(data, "volumeStep", "volume_step"), 0.01) or 0.01,
            digits=int(_as_float(_pick(data, "digits"), 2)),
            currency_profit=str(_pick(data, "profitCurrency", "currencyProfit", default="USD") or "USD"),
            raw=dict(data),
        )

    def normalize_volume(self, volume: float) -> float:
        """Clamp to [minVolume, maxVolume] and round down to the volume step."""
        step = self.volume_step if self.volume_step > 0 else 0.01
        steps = int(volume / step)
        normalised = round(steps * step, 8)
        normalised = max(self.min_volume, min(self.max_volume, normalised))
        return round(normalised, 8)


@dataclass
class PositionInfo:
    """An open position."""

    id: str
    symbol: str
    side: str  # "buy" or "sell"
    volume: float
    open_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    profit: float
    comment: str = ""
    time: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.side == OrderSide.BUY.value

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "PositionInfo":
        raw_type = str(_pick(data, "type", default="") or "").upper()
        side = OrderSide.BUY.value if "BUY" in raw_type else OrderSide.SELL.value
        raw_time = _pick(data, "time")
        return cls(
            id=str(_pick(data, "id", "positionId", default="") or ""),
            symbol=str(_pick(data, "symbol", default="") or ""),
            side=side,
            volume=_as_float(_pick(data, "volume")),
            open_price=_as_float(_pick(data, "openPrice", "open_price")),
            current_price=_as_float(_pick(data, "currentPrice", "current_price")),
            stop_loss=_pick(data, "stopLoss", "stop_loss"),
            take_profit=_pick(data, "takeProfit", "take_profit"),
            profit=_as_float(_pick(data, "profit")),
            comment=str(_pick(data, "comment", default="") or ""),
            time=str(raw_time) if raw_time else None,
            raw=dict(data),
        )


@dataclass
class TradeResult:
    """Outcome of an order / close request."""

    success: bool
    action: str
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    price: Optional[float] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __bool__(self) -> bool:  # allows `if result:`
        return self.success


# --------------------------------------------------------------------------- #
# Broker client
# --------------------------------------------------------------------------- #
class BrokerClient:
    """Unified async broker interface for MetaApi / MT5 and paper trading."""

    def __init__(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        platform: Optional[str] = None,
        region: Optional[str] = None,
        symbol_suffix: Optional[str] = None,
        paper_mode: Optional[bool] = None,
        application: Optional[str] = None,
    ) -> None:
        self.token = token if token is not None else os.getenv("METAAPI_TOKEN", "")
        self.account_id = account_id if account_id is not None else os.getenv("METAAPI_ACCOUNT_ID", "")
        self.platform = (platform or os.getenv("METAAPI_PLATFORM", "mt5")).lower()
        self.region = region if region is not None else (os.getenv("METAAPI_REGION") or None)
        self.application = application or os.getenv("METAAPI_APPLICATION", "The5ersMultiAgentBot")
        self.symbol_suffix = symbol_suffix if symbol_suffix is not None else os.getenv("SYMBOL_SUFFIX", "")
        self.paper_mode = _as_bool(os.getenv("PAPER_MODE"), False) if paper_mode is None else paper_mode
        self.magic = int(_as_float(os.getenv("MT5_MAGIC"), 520052))
        self.retry_attempts = int(_as_float(os.getenv("BROKER_RETRY_ATTEMPTS"), 3))

        self._api: Any = None
        self._account: Any = None
        self._connection: Any = None
        self._connected = False

        # ---- paper-mode state -------------------------------------------- #
        self._paper_rng = random.Random(20251112)
        self._paper_balance = _as_float(os.getenv("PAPER_START_BALANCE"), 5000.0)
        self._paper_prices: Dict[str, float] = {}
        self._paper_positions: Dict[str, Dict[str, Any]] = {}
        self._paper_closed: List[Dict[str, Any]] = []
        self._paper_order_seq = 0
        self._paper_base_prices = self._parse_base_prices(os.getenv("PAPER_BASE_PRICES", ""))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> "BrokerClient":
        """Connect to the broker (or initialise the paper engine)."""
        if self._connected:
            return self

        if self.paper_mode:
            self._connected = True
            LOGGER.warning("BrokerClient running in PAPER mode - no real orders will be sent.")
            return self

        if not METAAPI_AVAILABLE:
            raise RuntimeError(
                "metaapi_cloud_sdk is not installed. Run `pip install metaapi-cloud-sdk` "
                "or set PAPER_MODE=true."
            )
        if not self.token:
            raise RuntimeError("METAAPI_TOKEN is missing. Set it in your .env file or enable PAPER_MODE.")

        opts: Dict[str, Any] = {"application": self.application}
        if self.region:
            opts["region"] = self.region

        self._api = MetaApi(self.token, opts)
        self._account = await self._resolve_account()
        await self._account.wait_connected()

        self._connection = self._account.get_rpc_connection()
        await self._connection.connect()
        await self._connection.wait_synchronized({"timeoutInSeconds": 300})
        self._connected = True
        LOGGER.info("Connected to MetaApi account %s (%s)", self.account_id or "provisioned", self.platform)
        return self

    async def _resolve_account(self) -> Any:
        """Fetch the configured account, provisioning one from MT5 creds if needed."""
        accounts_api = self._api.metatrader_account_api

        if self.account_id:
            LOGGER.info("Fetching existing MetaApi account %s", self.account_id)
            return await accounts_api.get_account(self.account_id)

        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        if not (login and password and server):
            raise RuntimeError(
                "Provide METAAPI_ACCOUNT_ID, or MT5_LOGIN + MT5_PASSWORD + MT5_SERVER "
                "so the account can be provisioned on MetaApi."
            )

        LOGGER.info("Creating MetaApi account for login %s on %s", login, server)
        account = await accounts_api.create_account(
            {
                "login": str(login),
                "password": password,
                "name": f"The5ers-{login}",
                "server": server,
                "platform": self.platform,
                "magic": self.magic,
                "type": "cloud",
                "application": self.application,
            }
        )
        await account.deploy()
        self.account_id = getattr(account, "id", self.account_id)
        return account

    async def disconnect(self) -> None:
        """Release the connection / streaming subscription."""
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Error closing RPC connection: %s", exc)
        self._connection = None
        self._account = None
        self._api = None
        self._connected = False

    async def __aenter__(self) -> "BrokerClient":
        return await self.connect()

    async def __aexit__(self, *_exc: Any) -> None:
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    # Symbol helpers
    # ------------------------------------------------------------------ #
    def resolve_symbol(self, symbol: str) -> str:
        """Apply the configured broker suffix (e.g. ``BTCUSD`` -> ``BTCUSD.a``)."""
        if not self.symbol_suffix:
            return symbol
        if symbol.endswith(self.symbol_suffix):
            return symbol
        return f"{symbol}{self.symbol_suffix}"

    # ------------------------------------------------------------------ #
    # Account / market data
    # ------------------------------------------------------------------ #
    @_with_retry(attempts=3, description="get_account_information")
    async def get_account_information(self) -> AccountInfo:
        """Return balance, equity, margin and related account metrics."""
        if self.paper_mode:
            return self._paper_account_info()

        if self._connection is None:
            raise RuntimeError("BrokerClient is not connected. Call `await connect()` first.")

        raw = await self._connection.get_account_information()
        return AccountInfo.from_raw(raw, platform=self.platform)

    @_with_retry(attempts=3, description="get_symbol_price")
    async def get_symbol_price(self, symbol: str) -> Dict[str, float]:
        """Return ``{"bid": .., "ask": .., "mid": ..}`` for a symbol."""
        resolved = self.resolve_symbol(symbol)
        if self.paper_mode:
            mid = self._paper_next_price(resolved)
            spread = max(mid * 0.0002, 0.01)
            return {"bid": round(mid - spread / 2, 4), "ask": round(mid + spread / 2, 4), "mid": round(mid, 4)}

        raw = await self._connection.get_symbol_price(resolved)
        bid = _as_float(_pick(raw, "bid"))
        ask = _as_float(_pick(raw, "ask"))
        mid = (bid + ask) / 2 if bid and ask else (bid or ask)
        return {"bid": bid, "ask": ask, "mid": mid}

    @_with_retry(attempts=3, description="get_symbol_specification")
    async def get_symbol_specification(self, symbol: str) -> SymbolSpecification:
        """Return contract details (contract size, tick value, volume limits)."""
        resolved = self.resolve_symbol(symbol)
        if self.paper_mode:
            return SymbolSpecification(symbol=resolved, raw={"paper": True})

        raw = await self._connection.get_symbol_specification(resolved)
        return SymbolSpecification.from_raw(resolved, raw)

    @_with_retry(attempts=2, description="get_candles")
    async def get_candles(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` OHLC candles, oldest first.

        Each candle is ``{"time", "open", "high", "low", "close", "volume"}``.
        """
        resolved = self.resolve_symbol(symbol)
        if self.paper_mode:
            return self._paper_candles(resolved, timeframe, limit)

        minutes = self._timeframe_to_minutes(timeframe)
        start_time = datetime.now(timezone.utc) - timedelta(minutes=minutes * limit * 2)
        candles = await self._connection.get_historical_candles(resolved, timeframe, start_time, limit)

        normalised: List[Dict[str, Any]] = []
        for candle in candles or []:
            normalised.append(
                {
                    "time": _pick(candle, "time", "brokerTime"),
                    "open": _as_float(_pick(candle, "open")),
                    "high": _as_float(_pick(candle, "high")),
                    "low": _as_float(_pick(candle, "low")),
                    "close": _as_float(_pick(candle, "close")),
                    "volume": _as_float(_pick(candle, "tickVolume", "volume")),
                }
            )
        return normalised[-limit:]

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        return mapping.get(timeframe, 15)

    # ------------------------------------------------------------------ #
    # Positions & trading
    # ------------------------------------------------------------------ #
    @_with_retry(attempts=3, description="get_positions")
    async def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """Return open positions, optionally filtered by symbol."""
        if self.paper_mode:
            positions = [self._paper_position_view(p) for p in self._paper_positions.values()]
        else:
            if self._connection is None:
                raise RuntimeError("BrokerClient is not connected. Call `await connect()` first.")
            positions = await self._connection.get_positions() or []

        result = [PositionInfo.from_raw(p) for p in positions]

        if symbol:
            resolved = self.resolve_symbol(symbol)
            result = [p for p in result if p.symbol == resolved or p.symbol == symbol]
        return result

    async def open_market_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> TradeResult:
        """Send a market order in either direction (long/short)."""
        resolved = self.resolve_symbol(symbol)
        side = OrderSide(side)
        volume = round(float(volume), 8)

        if volume <= 0:
            return TradeResult(
                success=False, action="open", symbol=resolved, side=side.value, volume=volume,
                message="Volume must be greater than zero.",
            )

        if self.paper_mode:
            return self._paper_open(resolved, side, volume, stop_loss, take_profit, comment)

        if self._connection is None:
            raise RuntimeError("BrokerClient is not connected. Call `await connect()` first.")

        options: Dict[str, Any] = {}
        if comment:
            options["comment"] = comment[:31]
        options["clientId"] = f"ta-{int(datetime.now(timezone.utc).timestamp())}"

        try:
            if side.is_long:
                raw = await self._connection.create_market_buy_order(
                    resolved, volume, stop_loss, take_profit, options
                )
            else:
                raw = await self._connection.create_market_sell_order(
                    resolved, volume, stop_loss, take_profit, options
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a TradeResult
            LOGGER.error("Market %s order on %s failed: %s", side.value, resolved, exc)
            return TradeResult(
                success=False, action="open", symbol=resolved, side=side.value, volume=volume,
                message=str(exc),
            )

        position_id = _pick(raw, "positionId", "position_id")
        return TradeResult(
            success=True,
            action="open",
            symbol=resolved,
            side=side.value,
            volume=volume,
            price=_as_float(_pick(raw, "price", "openPrice")) or None,
            order_id=str(_pick(raw, "orderId", "order_id", default="") or "") or None,
            position_id=str(position_id) if position_id else None,
            message=str(_pick(raw, "message", "stringCode", default="order accepted") or "order accepted"),
            raw=dict(raw or {}),
        )

    async def close_position(self, position_id: str, comment: str = "bot-close") -> TradeResult:
        """Fully close a single position by id."""
        if not position_id:
            return TradeResult(success=False, action="close", message="position_id is required.")

        if self.paper_mode:
            return self._paper_close(position_id, comment)

        if self._connection is None:
            raise RuntimeError("BrokerClient is not connected. Call `await connect()` first.")

        options = {"comment": comment[:31]} if comment else None
        try:
            raw = await self._connection.close_position(position_id, options)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to close position %s: %s", position_id, exc)
            return TradeResult(success=False, action="close", position_id=position_id, message=str(exc))

        return TradeResult(
            success=True,
            action="close",
            symbol=str(_pick(raw or {}, "symbol", default="") or ""),
            position_id=position_id,
            order_id=str(_pick(raw or {}, "orderId", default="") or "") or None,
            message=str(_pick(raw or {}, "message", default="position closed") or "position closed"),
            raw=dict(raw or {}),
        )

    async def close_all_positions(self, symbol: Optional[str] = None) -> List[TradeResult]:
        """Close every open position (optionally limited to one symbol)."""
        positions = await self.get_positions(symbol=symbol)
        results: List[TradeResult] = []
        for position in positions:
            results.append(await self.close_position(position.id))
        return results

    # ------------------------------------------------------------------ #
    # Paper-mode engine
    # ------------------------------------------------------------------ #
    def _parse_base_prices(self, raw: str) -> Dict[str, float]:
        prices: Dict[str, float] = {"BTCUSD": 65000.0, "ETHUSD": 3200.0, "XAUUSD": 2350.0}
        for chunk in (raw or "").split(","):
            if ":" not in chunk:
                continue
            key, _, value = chunk.partition(":")
            prices[key.strip().upper()] = _as_float(value, prices.get(key.strip().upper(), 100.0))
        return prices

    def _paper_next_price(self, symbol: str) -> float:
        current = self._paper_prices.get(symbol)
        if current is None:
            base = self._paper_base_prices.get(symbol.upper().rstrip("."), None)
            if base is None:
                base = self._paper_base_prices.get(symbol.upper(), 1000.0)
            current = float(base)
        current = max(0.0001, current * (1.0 + self._paper_rng.gauss(0.0, 0.0006)))
        self._paper_prices[symbol] = current
        return current

    def _paper_account_info(self) -> AccountInfo:
        floating = sum(self._paper_position_view(p)["profit"] for p in self._paper_positions.values())
        equity = self._paper_balance + floating
        return AccountInfo(
            balance=round(self._paper_balance, 2),
            equity=round(equity, 2),
            margin=round(sum(p["volume"] * p["open_price"] * 0.01 for p in self._paper_positions.values()), 2),
            free_margin=round(equity, 2),
            margin_level=0.0,
            currency="USD",
            leverage=100.0,
            login=99999,
            broker="PaperBroker",
            server="paper",
            platform="paper",
            raw={"paper": True},
        )

    def _paper_open(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        comment: str,
    ) -> TradeResult:
        price = self._paper_next_price(symbol)
        self._paper_order_seq += 1
        position_id = f"paper-{self._paper_order_seq}"
        self._paper_positions[position_id] = {
            "id": position_id,
            "symbol": symbol,
            "side": side.value,
            "volume": volume,
            "open_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "comment": comment,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        LOGGER.info("[PAPER] %s %s %s @ %.2f", side.value.upper(), volume, symbol, price)
        return TradeResult(
            success=True, action="open", symbol=symbol, side=side.value, volume=volume,
            price=price, order_id=position_id, position_id=position_id, message="paper order filled",
        )

    def _paper_close(self, position_id: str, comment: str = "") -> TradeResult:
        position = self._paper_positions.pop(position_id, None)
        if position is None:
            return TradeResult(success=False, action="close", position_id=position_id, message="position not found")

        view = self._paper_position_view(position)
        pnl = view["profit"]
        self._paper_balance += pnl
        self._paper_closed.append({**position, "profit": pnl, "closed_at": datetime.now(timezone.utc).isoformat()})
        LOGGER.info("[PAPER] closed %s for %.2f USD (balance %.2f)", position_id, pnl, self._paper_balance)
        return TradeResult(
            success=True, action="close", symbol=position["symbol"], position_id=position_id,
            volume=position["volume"], price=view["currentPrice"],
            message=comment or f"paper position closed (PnL {pnl:.2f})",
        )

    def _paper_position_view(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Return a position in MT5 raw format so it parses through ``PositionInfo``."""
        current = self._paper_prices.get(position["symbol"], position["open_price"])
        direction = 1.0 if position["side"] == OrderSide.BUY.value else -1.0
        profit = (current - position["open_price"]) * direction * position["volume"]
        return {
            "id": position["id"],
            "symbol": position["symbol"],
            "type": "POSITION_TYPE_BUY" if direction > 0 else "POSITION_TYPE_SELL",
            "volume": position["volume"],
            "openPrice": position["open_price"],
            "currentPrice": round(current, 4),
            "stopLoss": position.get("stop_loss"),
            "takeProfit": position.get("take_profit"),
            "profit": round(profit, 2),
            "comment": position.get("comment", ""),
            "time": position.get("time"),
        }

    def _paper_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        end_price = self._paper_next_price(symbol)
        minutes = self._timeframe_to_minutes(timeframe)
        rng = random.Random(f"{symbol}-{timeframe}-{limit}")

        closes = [end_price]
        price = end_price
        for _ in range(max(limit - 1, 0)):
            price = price / (1.0 + rng.gauss(0.0, 0.0025))
            closes.append(price)
        closes.reverse()

        volatility = end_price * 0.003
        start = datetime.now(timezone.utc) - timedelta(minutes=minutes * limit)
        candles: List[Dict[str, Any]] = []
        for index, close in enumerate(closes):
            open_price = closes[index - 1] if index else close
            high = max(open_price, close) + abs(rng.gauss(0.0, volatility))
            low = min(open_price, close) - abs(rng.gauss(0.0, volatility))
            candles.append(
                {
                    "time": (start + timedelta(minutes=minutes * index)).isoformat(),
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(max(low, 0.0001), 4),
                    "close": round(close, 4),
                    "volume": round(abs(rng.gauss(100.0, 30.0)), 2),
                }
            )
        return candles


# --------------------------------------------------------------------------- #
# Smoke test: python broker_client.py
# --------------------------------------------------------------------------- #
async def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    async with BrokerClient() as broker:
        account = await broker.get_account_information()
        print(f"balance={account.balance} equity={account.equity} currency={account.currency}")
        price = await broker.get_symbol_price("BTCUSD")
        print(f"BTCUSD bid={price['bid']} ask={price['ask']}")
        candles = await broker.get_candles("BTCUSD", "15m", limit=5)
        print(f"received {len(candles)} candles, last close={candles[-1]['close'] if candles else 'n/a'}")
        result = await broker.open_market_order("BTCUSD", OrderSide.BUY, 0.01, comment="smoke-test")
        print(f"open: success={result.success} position_id={result.position_id}")
        if result.position_id:
            closed = await broker.close_position(result.position_id)
            print(f"close: success={closed.success} message={closed.message}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
