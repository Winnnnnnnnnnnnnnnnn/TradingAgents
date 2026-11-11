"""
kill_switch.py
==============
"Triple-Lock" daily risk governor for a The5ers Bootcamp $5,000 account.

The three locks
---------------
Lock 1 - PROFIT CAP  : once the day's P&L reaches ``DAILY_PROFIT_TARGET`` (+60 USD)
                       the bot stops opening new trades and banks the win.
Lock 2 - MAX LOSS   : if the day's P&L falls to ``DAILY_MAX_LOSS`` (-100 USD) the
                       bot hard-stops for the rest of the day.
Lock 3 - MANUAL     : the operator can trip the kill switch from the dashboard.
                       This lock survives restarts and only clears on an explicit
                       operator reset.

Why equity and not just balance?
--------------------------------
Prop-firm daily drawdown limits are measured on *equity* (realised + floating P&L).
This module therefore tracks ``day_start_balance`` and compares the live equity
against it, which matches how The5ers evaluates the daily loss rule.

Environment variables
---------------------
KILL_SWITCH_STATE_FILE   persistence path (default: .state/kill_switch_state.json)
DAILY_PROFIT_TARGET      +60.0 USD by default
DAILY_MAX_LOSS           -100.0 USD by default
DAILY_RESET_TZ           IANA timezone used for the daily rollover (default UTC)
KILL_SWITCH_AUTO_CLOSE   "true" to flatten positions the moment a lock trips
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class KillLockType(str, Enum):
    """Which of the three locks is currently engaged."""

    NONE = "none"
    PROFIT_TARGET = "profit_target"
    MAX_DAILY_LOSS = "max_daily_loss"
    MANUAL = "manual"

    @property
    def is_engaged(self) -> bool:
        return self is not KillLockType.NONE


@dataclass
class KillSwitchDecision:
    """Result of evaluating the kill switch against a live account snapshot."""

    allowed: bool
    lock_type: KillLockType
    reason: str
    daily_pnl: float
    day_start_balance: float
    profit_target: float
    max_daily_loss: float
    trades_today: int = 0
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def remaining_to_target(self) -> float:
        """USD still needed to hit the profit cap (never negative)."""
        return round(max(0.0, self.profit_target - self.daily_pnl), 2)

    @property
    def remaining_risk(self) -> float:
        """USD the bot may still lose today before lock 2 trips (never negative)."""
        return round(max(0.0, self.max_daily_loss + self.daily_pnl), 2)

    @property
    def progress_to_target(self) -> float:
        """0..1 progress towards the daily profit cap."""
        if self.profit_target <= 0:
            return 0.0
        return max(0.0, min(1.0, self.daily_pnl / self.profit_target))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lock_type"] = self.lock_type.value
        data["remaining_to_target"] = self.remaining_to_target
        data["remaining_risk"] = self.remaining_risk
        data["progress_to_target"] = round(self.progress_to_target, 4)
        return data


class KillSwitch:
    """Thread-safe, file-backed daily profit/loss governor."""

    def __init__(
        self,
        state_file: Optional[str] = None,
        profit_target: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
        reset_timezone: Optional[str] = None,
        auto_close_on_lock: Optional[bool] = None,
    ) -> None:
        self.state_file = Path(
            state_file
            or os.getenv("KILL_SWITCH_STATE_FILE")
            or Path(".state") / "kill_switch_state.json"
        ).expanduser()
        self.profit_target = abs(
            _as_float(profit_target if profit_target is not None else os.getenv("DAILY_PROFIT_TARGET"), 60.0)
        )
        self.max_daily_loss = abs(
            _as_float(max_daily_loss if max_daily_loss is not None else os.getenv("DAILY_MAX_LOSS"), 100.0)
        )
        self.reset_timezone_name = reset_timezone or os.getenv("DAILY_RESET_TZ", "UTC")
        self.auto_close_on_lock = (
            _as_bool(os.getenv("KILL_SWITCH_AUTO_CLOSE"), False)
            if auto_close_on_lock is None
            else auto_close_on_lock
        )

        self._lock = threading.RLock()
        self._state: Dict[str, Any] = self._default_state()
        self._load_state()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            "trading_day": None,
            "day_start_balance": None,
            "locked": False,
            "lock_type": KillLockType.NONE.value,
            "lock_reason": "",
            "locked_at": None,
            "trades_today": 0,
            "realized_pnl_today": 0.0,
            "last_equity": 0.0,
            "updated_at": None,
        }

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with self.state_file.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                self._state.update(stored)
            LOGGER.info("Kill switch state restored from %s", self.state_file)
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.error("Could not read kill switch state (%s) - starting fresh.", exc)
            self._state = self._default_state()

    def _persist(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self._state, handle, indent=2)
            tmp.replace(self.state_file)
        except OSError as exc:  # pragma: no cover - disk issues
            LOGGER.error("Could not persist kill switch state: %s", exc)

    # ------------------------------------------------------------------ #
    # Time helpers
    # ------------------------------------------------------------------ #
    def _tz(self):
        if ZoneInfo is None:
            return timezone.utc
        try:
            return ZoneInfo(self.reset_timezone_name)
        except Exception:  # noqa: BLE001 - unknown tz name
            return timezone.utc

    def _current_trading_day(self, now: Optional[datetime] = None) -> date:
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(self._tz()).date()

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #
    def _rollover_if_needed(self, balance: float, now: Optional[datetime] = None) -> bool:
        """Reset the daily baseline when the trading day changes. Returns True if rolled."""
        today = self._current_trading_day(now)
        stored_day = self._state.get("trading_day")
        if isinstance(stored_day, str):
            try:
                stored_day = date.fromisoformat(stored_day)
            except ValueError:
                stored_day = None

        if stored_day == today and self._state.get("day_start_balance") is not None:
            return False

        LOGGER.info(
            "New trading day %s - resetting baseline to balance %.2f (previous day %s)",
            today, balance, stored_day,
        )
        self._state.update(
            {
                "trading_day": today.isoformat(),
                "day_start_balance": round(balance, 2),
                "locked": False,
                "lock_type": KillLockType.NONE.value,
                "lock_reason": "",
                "locked_at": None,
                "trades_today": 0,
                "realized_pnl_today": 0.0,
                "last_equity": round(balance, 2),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._persist()
        return True

    def _engage_lock(self, lock_type: KillLockType, reason: str) -> None:
        self._state.update(
            {
                "locked": True,
                "lock_type": lock_type.value,
                "lock_reason": reason,
                "locked_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._persist()
        LOGGER.warning("KILL SWITCH ENGAGED [%s]: %s", lock_type.value, reason)

    def evaluate(
        self,
        equity: float,
        balance: float,
        now: Optional[datetime] = None,
    ) -> KillSwitchDecision:
        """Evaluate all three locks against the current account snapshot.

        ``balance`` seeds the daily baseline on the first call of a new day;
        ``equity`` drives the actual P&L measurement (realised + floating).
        """
        with self._lock:
            self._rollover_if_needed(balance, now)

            day_start = _as_float(self._state.get("day_start_balance"), balance)
            daily_pnl = round(equity - day_start, 2)
            self._state["last_equity"] = round(equity, 2)
            self._state["updated_at"] = datetime.now(timezone.utc).isoformat()

            current_lock = KillLockType(self._state.get("lock_type", KillLockType.NONE.value))

            # Lock 3 (manual) is sticky and only cleared by an operator reset.
            if self._state.get("locked") and current_lock is KillLockType.MANUAL:
                return self._decision(False, current_lock, self._state.get("lock_reason", "manual lock"), daily_pnl, day_start)

            if not self._state.get("locked"):
                # Lock 1 - bank the win.
                if daily_pnl >= self.profit_target:
                    self._engage_lock(
                        KillLockType.PROFIT_TARGET,
                        f"Daily profit target reached (+{daily_pnl:.2f} USD >= +{self.profit_target:.2f} USD).",
                    )
                    current_lock = KillLockType.PROFIT_TARGET
                # Lock 2 - protect the daily drawdown.
                elif daily_pnl <= -self.max_daily_loss:
                    self._engage_lock(
                        KillLockType.MAX_DAILY_LOSS,
                        f"Daily max loss breached ({daily_pnl:.2f} USD <= -{self.max_daily_loss:.2f} USD).",
                    )
                    current_lock = KillLockType.MAX_DAILY_LOSS
            else:
                self._persist()

            if self._state.get("locked"):
                reason = self._state.get("lock_reason") or "kill switch engaged"
                return self._decision(False, current_lock, reason, daily_pnl, day_start)

            return self._decision(
                True, KillLockType.NONE,
                "Trading allowed: within daily profit target and max loss limits.",
                daily_pnl, day_start,
            )

    def _decision(
        self,
        allowed: bool,
        lock_type: KillLockType,
        reason: str,
        daily_pnl: float,
        day_start: float,
    ) -> KillSwitchDecision:
        return KillSwitchDecision(
            allowed=allowed,
            lock_type=lock_type,
            reason=reason,
            daily_pnl=daily_pnl,
            day_start_balance=round(day_start, 2),
            profit_target=self.profit_target,
            max_daily_loss=self.max_daily_loss,
            trades_today=int(self._state.get("trades_today", 0)),
        )

    # ------------------------------------------------------------------ #
    # Operator actions
    # ------------------------------------------------------------------ #
    def engage_manual(self, reason: str = "Manual kill switch engaged by operator.") -> KillSwitchDecision:
        """Lock 3 - trip the kill switch by hand."""
        with self._lock:
            self._engage_lock(KillLockType.MANUAL, reason)
            day_start = _as_float(self._state.get("day_start_balance"), 0.0)
            daily_pnl = round(_as_float(self._state.get("last_equity"), day_start) - day_start, 2)
            return self._decision(False, KillLockType.MANUAL, reason, daily_pnl, day_start)

    def reset(
        self,
        current_balance: Optional[float] = None,
        reason: str = "Operator reset.",
        force: bool = False,
    ) -> KillSwitchDecision:
        """Clear the lock and re-arm the day from the current balance.

        Clearing a Lock 3 (manual) trip requires ``force=True`` so it can never
        happen by accident. This does **not** bypass the daily loss rule - the new
        baseline simply becomes the current balance.
        """
        with self._lock:
            lock_type = KillLockType(self._state.get("lock_type", KillLockType.NONE.value))
            if lock_type is KillLockType.MANUAL and not force:
                LOGGER.warning("Refusing to clear a MANUAL lock without force=True.")
                day_start = _as_float(self._state.get("day_start_balance"), 0.0)
                daily_pnl = round(_as_float(self._state.get("last_equity"), day_start) - day_start, 2)
                return self._decision(
                    False, KillLockType.MANUAL,
                    "Manual lock requires an explicit force reset.", daily_pnl, day_start,
                )

            new_baseline = _as_float(
                current_balance if current_balance is not None else self._state.get("last_equity"),
                0.0,
            )
            self._state.update(
                {
                    "trading_day": self._current_trading_day().isoformat(),
                    "day_start_balance": round(new_baseline, 2),
                    "locked": False,
                    "lock_type": KillLockType.NONE.value,
                    "lock_reason": "",
                    "locked_at": None,
                    "trades_today": 0,
                    "realized_pnl_today": 0.0,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._persist()
            LOGGER.info("Kill switch reset (%s) - new baseline %.2f", reason, new_baseline)
            return self._decision(
                True, KillLockType.NONE,
                f"{reason} Baseline re-armed at {new_baseline:.2f} USD.",
                0.0, new_baseline,
            )

    def note_trade_opened(self) -> None:
        """Increment the daily trade counter (used for reporting limits)."""
        with self._lock:
            self._state["trades_today"] = int(self._state.get("trades_today", 0)) + 1
            self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._persist()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_locked(self) -> bool:
        with self._lock:
            return bool(self._state.get("locked"))

    @property
    def lock_type(self) -> KillLockType:
        with self._lock:
            return KillLockType(self._state.get("lock_type", KillLockType.NONE.value))

    def status(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for the dashboard."""
        with self._lock:
            return {
                "locked": bool(self._state.get("locked")),
                "lock_type": self._state.get("lock_type", KillLockType.NONE.value),
                "lock_reason": self._state.get("lock_reason", ""),
                "locked_at": self._state.get("locked_at"),
                "trading_day": self._state.get("trading_day"),
                "day_start_balance": self._state.get("day_start_balance"),
                "last_equity": self._state.get("last_equity"),
                "trades_today": self._state.get("trades_today", 0),
                "profit_target": self.profit_target,
                "max_daily_loss": self.max_daily_loss,
                "auto_close_on_lock": self.auto_close_on_lock,
                "reset_timezone": self.reset_timezone_name,
                "state_file": str(self.state_file),
            }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ks = KillSwitch(state_file=".state/demo_kill_switch.json")
    ks.reset(current_balance=5000.0, reason="demo start", force=True)

    for equity in (5010.0, 5040.0, 5065.0):
        decision = ks.evaluate(equity=equity, balance=5000.0)
        print(f"equity={equity:>8.2f} allowed={decision.allowed!s:<5} "
              f"lock={decision.lock_type.value:<15} pnl={decision.daily_pnl:+.2f} "
              f"reason={decision.reason}")

    print("status:", json.dumps(ks.status(), indent=2))
