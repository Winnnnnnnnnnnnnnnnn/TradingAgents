# The5ers Multi-Agent Trading Bot — Architecture & Code Inspection Report

Scope inspected: `app.py`, `broker_client.py`, `kill_switch.py` (the strategy/indicator
logic lives inside `app.py`; there is no separate strategy module).
Line references are to the versions in this repo.

---

## 1. System overview

The bot is a 4-agent pipeline wrapped by a single orchestrator, with a separate,
file-backed "Triple-Lock" risk governor. It targets a **The5ers Bootcamp $5,000**
account and talks to MetaTrader 4/5 through the **MetaApi Cloud SDK** (with a fully
offline **paper-mode** simulator as a fallback).

| File | Responsibility |
|------|----------------|
| `app.py` | Config (`BotConfig`), the 4 agents, the `TradingOrchestrator`, the process-wide singleton, the headless single-pass runner, and the Streamlit dashboard. |
| `broker_client.py` | Async broker abstraction (`BrokerClient`) with a single interface over MetaApi/MT5 and the paper simulator; value objects (`AccountInfo`, `SymbolSpecification`, `PositionInfo`, `TradeResult`, `OrderSide`). |
| `kill_switch.py` | `KillSwitch` daily governor + `KillLockType` / `KillSwitchDecision`. |

**Entry points**

- `streamlit run the5ers_bot/app.py` → `main()` → `_render_dashboard()` (interactive, 24/7).
- `python the5ers_bot/app.py --once` → `run_single_pass()` (3 warm-up ticks, headless smoke test).
- `python the5ers_bot/broker_client.py` → `_smoke_test()` (connects, quotes, opens + closes one order).
- `python the5ers_bot/kill_switch.py` → manual kill-switch demo.

---

## 2. Agent decomposition & roles

```
                         ┌───────────────────────────────────────────────┐
                         │              TradingOrchestrator              │
                         │   (background thread + private asyncio loop)  │
                         └───────────────┬───────────────────────────────┘
                                         │  _tick()  one pass per poll interval
        ┌────────────────┬───────────────┼────────────────┬─────────────────┐
        ▼                ▼               ▼                ▼                 ▼
 MarketDataAgent   StrategyAgent     RiskAgent      ExecutionAgent     KillSwitch
 fetch + EMA/RSI/ATR  cross → Signal  size + SL/TP   send order        daily governor
        │                │               │                │                 ▲
        └────────────────┴──── BrokerClient (MetaApi / paper) ──────┴─────────┘
```

### 2.1 `MarketDataAgent` — market data & indicators
`app.py` (class starts at the `MarketDataAgent` definition)

- **Role:** fetch recent candles for a symbol, normalise them into a DataFrame, and
  append the indicator columns the strategy needs.
- **Holds:** `broker: BrokerClient`, `config: BotConfig`.
- **Key methods**
  - `async fetch_frame(symbol) -> pd.DataFrame` — calls `broker.get_candles(symbol, timeframe, candle_limit)` then `build_frame`.
  - `@staticmethod build_frame(candles: List[Dict]) -> pd.DataFrame` — builds the frame, coerces `open/high/low/close/volume` to numeric, drops rows with NaN `close`.
  - `add_indicators(frame) -> pd.DataFrame` — appends:
    - `ema_fast`, `ema_slow` — EMA via `ewm(span=…, adjust=False)`.
    - `rsi` — Wilder smoothing: `avg_gain/avg_loss` with `ewm(alpha=1/period)`, `NaN`-filled to 50.
    - `atr` — true range (max of `high-low`, `|high-prev_close|`, `|low-prev_close|`) smoothed with `ewm(alpha=1/period)`.

### 2.2 `StrategyAgent` — signal generation
`app.py`

- **Role:** convert indicators into a directional `Signal` (BUY / SELL / none).
- **Holds:** `config: BotConfig`.
- **Key method:** `generate_signal(symbol, frame) -> Optional[Signal]`
  - Guard: needs `len(frame) >= max(ema_slow, rsi_period, atr_period) + 2` (default 57 bars) and finite positive ATR.
  - Crossover: `crossed_up = fast_prev <= slow_prev and fast_now > slow_now` (mirror for down).
  - Confirmation: BUY requires `rsi >= rsi_long_min (52)` **and** `close > slow_now`; SELL requires `rsi <= rsi_short_max (48)` **and** `close < slow_now`.
  - `strength = |ema_fast - ema_slow| / atr`.

### 2.3 `RiskAgent` — sizing, SL/TP, kill-switch gate
`app.py`

- **Role:** the only place a trade can be approved. Enforces the kill-switch verdict, position limits, the ATR stop, the risk budget and the **1:3 RRR floor**.
- **Holds:** `broker`, `config`, `kill_switch`.
- **Key method:** `async assess(signal, entry_price, account, positions, decision) -> RiskDecision`
  - Rejects if `not decision.allowed` (kill switch), if `len(positions) >= max_open_positions`, or if the symbol already has a position.
  - `stop_distance = signal.atr * atr_stop_multiplier` (default **2×ATR**).
  - `risk_budget = equity * risk_per_trade_pct/100` (default **0.5%**), capped by `decision.remaining_risk` (the day's remaining loss budget).
  - `loss_per_lot = (stop_distance / tick_size) * tick_value`; `volume = normalize_volume(risk_budget / loss_per_lot)`; rejects if below `min_volume`.
  - `take_distance = stop_distance * reward_risk`; SL/TP placed symmetrically for long/short.

### 2.4 `ExecutionAgent` — order placement
`app.py`

- **Role:** send an approved decision to the broker; record the trade against the kill switch.
- **Holds:** `broker`, `kill_switch`, `dry_run`.
- **Key method:** `async execute(decision) -> TradeResult`
  - `dry_run` short-circuits to a simulated success (no order sent).
  - Otherwise calls `broker.open_market_order(...)`; on success calls `kill_switch.note_trade_opened()`.

### 2.5 `KillSwitch` — Triple-Lock governor (the "risk-management agent")
`kill_switch.py`

- **Role:** the daily profit/loss governor. Not an agent in `app.py` — a standalone,
  thread-safe, file-backed state machine consulted by `RiskAgent`, `ExecutionAgent`
  and the orchestrator.
- Three locks: **PROFIT_TARGET (+$60)**, **MAX_DAILY_LOSS (−$100)**, **MANUAL**.
- **Key methods:** `evaluate`, `set_account_scope`, `reset`, `engage_manual`, `note_trade_opened`, `status`.

### 2.6 `TradingOrchestrator` — the conductor
`app.py`

- **Role:** own the agents, run the loop in a background thread with its own asyncio
  event loop, expose a thread-safe `snapshot()` for the UI, and handle lock-driven flattening.
- **Key methods:** `start`, `stop`, `snapshot`, `set_credentials`, `close_all_now`, `_run`, `_tick`, `_flatten_all`.

---

## 3. Signal generation & data flow

### 3.1 End-to-end flow (one symbol, one tick)

```
broker.get_candles(symbol, "15m", 200)      # MetaApi get_historical_candles  | paper _paper_candles
        │  List[{time,open,high,low,close,volume}]
        ▼
MarketDataAgent.build_frame()               # -> DataFrame, numeric, drop NaN close
        ▼
MarketDataAgent.add_indicators()            # + ema_fast, ema_slow, rsi, atr
        ▼
StrategyAgent.generate_signal(symbol, df)   # EMA cross + RSI filter -> Signal | None
        ▼
broker.get_symbol_price(symbol)             # ask for longs, bid for shorts -> entry_price
        ▼
KillSwitch.evaluate(equity, balance)        # -> KillSwitchDecision (allowed?)
        ▼
RiskAgent.assess(signal, entry, account, positions, decision)
        │   reject: kill switch / max positions / duplicate symbol
        │   size  : 0.5% equity ∩ remaining_risk  →  volume
        │   levels: SL = entry ∓ 2·ATR ,  TP = entry ± 3·(2·ATR)
        ▼
ExecutionAgent.execute(decision)            # broker.open_market_order(...)  (+note_trade_opened)
        ▼
_record_trade(...) ; log "Opened …"         # dashboard "Trade log" / "Event log"
```

### 3.2 Where the pieces live

| Stage | Function | Source |
|-------|----------|--------|
| Feed | `BrokerClient.get_candles` | `broker_client.py` (MetaApi `get_historical_candles`, or `_paper_candles`) |
| Build frame | `MarketDataAgent.build_frame` | `app.py` |
| Indicators | `MarketDataAgent.add_indicators` | `app.py` |
| Signal | `StrategyAgent.generate_signal` | `app.py` |
| Quote | `BrokerClient.get_symbol_price` | `broker_client.py` |
| Risk verdict | `KillSwitch.evaluate` | `kill_switch.py` |
| Sizing / SL-TP | `RiskAgent.assess` | `app.py` |
| Order | `BrokerClient.open_market_order` | `broker_client.py` |
| Bookkeeping | `TradingOrchestrator._record_signal` / `_record_trade` | `app.py` |

**Note:** the entry signal is evaluated on the **last two rows only** each tick
(`frame.iloc[-1]`, `frame.iloc[-2]`), so the crossover must occur on the most recent
closed candle; a signal is not "carried" across ticks.

---

## 4. Risk guardrails & Triple-Lock enforcement

### 4.1 The 1:3 RRR floor — `MIN_REWARD_RISK = 3.0`

| Location | Enforcement |
|----------|-------------|
| `app.py` module constant `MIN_REWARD_RISK = 3.0` | single source of truth |
| `BotConfig.reward_risk` default = `MIN_REWARD_RISK` | dataclass default |
| `BotConfig.from_env()` | `reward_risk = max(MIN_REWARD_RISK, _as_float(os.getenv("REWARD_RISK"), MIN_REWARD_RISK))` — a smaller `REWARD_RISK` is **clamped up**, so a misconfigured env can never weaken the floor |
| `BotConfig.summary()` | surfaces `1 : {reward_risk} (min 1:3)` in the dashboard |
| `RiskAgent.assess()` | `take_distance = stop_distance * cfg.reward_risk`; `TP = entry ± take_distance` |

Because the stop is derived from ATR (`stop_distance = atr * 2.0`) and the target is
`reward_risk × stop_distance`, every emitted trade has reward:risk ≥ 1:3 by construction.

### 4.2 Daily profit lock (+$60) and max-drawdown stop (−$100)

Both live entirely inside `KillSwitch.evaluate(equity, balance, now)`
(`kill_switch.py`):

```
_rollover_if_needed(balance, now)          # new trading day → reset baseline to current balance
daily_pnl = round(equity − day_start_balance, 2)
if not locked:
    if daily_pnl >= profit_target:   engage PROFIT_TARGET (+$60)   # Lock 1
    elif daily_pnl <= -max_daily_loss: engage MAX_DAILY_LOSS (−$100) # Lock 2
→ returns KillSwitchDecision(allowed=False, lock_type=…)
```

- P&L is measured on **equity** (realised + floating), matching prop-firm daily-loss rules.
- Defaults: `profit_target = 60.0`, `max_daily_loss = 100.0` (`DAILY_PROFIT_TARGET` / `DAILY_MAX_LOSS`).
- `remaining_risk = max(0, max_daily_loss + daily_pnl)` — the "can still lose" figure that `RiskAgent` uses to cap size.
- `progress_to_target = clamp(daily_pnl / profit_target, 0, 1)` — drives the dashboard progress bar.

### 4.3 Where the decision is consumed

| Consumer | Code | Behaviour |
|----------|------|-----------|
| Orchestrator | `_tick()` calls `kill_switch.evaluate(equity, balance)` | refreshes state for UI every tick |
| Orchestrator | if `not decision.allowed` and `auto_close_on_lock` and positions exist | `_flatten_all("kill-switch flatten")` — flatten **once** per lock, then halt for the day |
| RiskAgent | `if not decision.allowed: reject` | no new entries while locked |
| RiskAgent | `risk_budget = min(risk_budget, decision.remaining_risk)` | never risks more than the day's remaining budget |
| Dashboard | `_kill_switch_state()` | maps lock → banner (`EMERGENCY STOPPED` / `LOCKED · TARGET BANKED` / `LOCKED · MANUAL`) |

`auto_close_on_lock` defaults to **True** (`KILL_SWITCH_AUTO_CLOSE`), so tripping either
lock flattens open positions. The `flattened_for_lock` flag prevents repeated flatten calls.

### 4.4 Manual lock (Lock 3)
`engage_manual()` sets `MANUAL`; `reset(..., force=True)` is required to clear it (the
dashboard gates this behind a confirmation checkbox). **See finding F-2** — the manual
lock is *not* preserved across the daily rollover.

---

## 5. Asynchronous execution & state management

### 5.1 Threading / event-loop model

```
Streamlit main thread                      Background "trading-loop" thread
────────────────────────                   ────────────────────────────────
main() → _render_dashboard()               _thread_entry():
  get_orchestrator()  ──singleton──►         loop = asyncio.new_event_loop()
  snapshot()                                  set_event_loop(loop)
  start() ──────────────────────────►         loop.run_until_complete(_run())
  stop()  ── _stop_event.set() ─────►         │
  close_all_now()                            │  while not _stop_event.is_set():
    asyncio.run_coroutine_threadsafe ───────►│      await _tick()
              future.result(timeout)          │      await asyncio.sleep(poll − elapsed)
  snapshot()  (RLock copy)                    └─ finally: _shutdown(); loop.close()
```

- **One loop, one thread.** `_thread_entry` creates a private event loop so the blocking
  Streamlit render loop is never blocked.
- **UI → loop hand-off** uses `asyncio.run_coroutine_threadsafe(self._flatten_all(...), self._loop)`
  with a timeout, so "Emergency Close All" runs on the trading thread (not a second thread).
- **Cooperative stop** via `threading.Event` (`_stop_event`); the poll sleep is bounded by
  `max(1.0, poll_interval_seconds − elapsed)`.

### 5.2 Singleton orchestrator (multi-session safety)

```python
_ORCHESTRATOR: Optional["TradingOrchestrator"] = None
_ORCHESTRATOR_LOCK = threading.Lock()

def get_orchestrator(config=None):
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:                 # fast path
        with _ORCHESTRATOR_LOCK:             # double-checked locking
            if _ORCHESTRATOR is None:
                _ORCHESTRATOR = TradingOrchestrator(config or BotConfig.from_env())
    return _ORCHESTRATOR
```

Streamlit re-runs the whole script on every interaction and can serve several browser
sessions from **one process**. The module-level singleton guarantees a single trading
thread and one shared kill-switch instance; without it, each rerun would spawn a new
thread and split the locks. Concurrency inside the object is guarded by
`threading.RLock` (`self._lock`) around `_state`, `_signals`, `_trades`, `_events`.

### 5.3 Persistent state — `.state/`

- `KillSwitch` persists to `.state/kill_switch_state.json` (override via `KILL_SWITCH_STATE_FILE`).
- Writes are **atomic**: dump to `*.tmp` then `Path.replace()`, so a crash never leaves a torn file.
- State is **account-scoped**: `set_account_scope(account_id)` clears the daily baseline/lock when the account changes (a $5k baseline carried onto a bigger account would instantly trip the +$60 cap).
- Daily rollover keys off `trading_day` (timezone from `DAILY_RESET_TZ`, default UTC) — when the day changes, the baseline resets to the current balance.
- The file survives process restarts, so a restart mid-day does **not** re-arm the account.

### 5.4 Credential resolution (no secrets on disk)

`_effective_credentials()` resolves in order: **dashboard input → `st.secrets` → environment
(`.env`)**. The dashboard stores the token in `st.session_state` (browser session only,
never written to disk); `credentials_info()` returns a **masked** token (`_mask`) plus the
source label.

---

## 6. Core function signature reference

```python
# app.py ────────────────────────────────────────────────────────────────
class BotConfig:
    @classmethod
    def from_env(cls) -> "BotConfig"
    def summary(self) -> Dict[str, Any]

class MarketDataAgent:
    async def fetch_frame(self, symbol: str) -> pd.DataFrame
    @staticmethod
    def build_frame(candles: List[Dict[str, Any]]) -> pd.DataFrame
    def add_indicators(self, frame: pd.DataFrame) -> pd.DataFrame

class StrategyAgent:
    def generate_signal(self, symbol: str, frame: pd.DataFrame) -> Optional[Signal]

class RiskAgent:
    async def assess(self, signal: Signal, entry_price: float, account: AccountInfo,
                     positions: List[PositionInfo], decision: KillSwitchDecision) -> RiskDecision

class ExecutionAgent:
    async def execute(self, decision: RiskDecision) -> TradeResult

class TradingOrchestrator:
    def set_credentials(self, token: Optional[str] = None, account_id: Optional[str] = None) -> bool
    def credentials_info(self) -> Dict[str, Any]
    def start(self) -> bool
    def stop(self, timeout: float = 15.0) -> None
    def snapshot(self) -> Dict[str, Any]
    def close_all_now(self, timeout: float = 60.0) -> List[TradeResult]
    async def _run(self) -> None
    async def _tick(self) -> None
    async def _flatten_all(self, reason: str) -> List[TradeResult]

def get_orchestrator(config: Optional[BotConfig] = None) -> "TradingOrchestrator"
def run_single_pass() -> int

# broker_client.py ──────────────────────────────────────────────────────
class BrokerClient:
    async def connect(self) -> "BrokerClient"
    async def disconnect(self) -> None
    def resolve_symbol(self, symbol: str) -> str
    async def get_account_information(self) -> AccountInfo
    async def get_symbol_price(self, symbol: str) -> Dict[str, float]
    async def get_symbol_specification(self, symbol: str) -> SymbolSpecification
    async def get_candles(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> List[Dict[str, Any]]
    async def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]
    async def open_market_order(self, symbol: str, side: OrderSide, volume: float,
                                stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                comment: str = "") -> TradeResult
    async def close_position(self, position_id: str, comment: str = "bot-close") -> TradeResult
    async def close_all_positions(self, symbol: Optional[str] = None) -> List[TradeResult]

# kill_switch.py ────────────────────────────────────────────────────────
class KillSwitch:
    def evaluate(self, equity: float, balance: float, now: Optional[datetime] = None) -> KillSwitchDecision
    def set_account_scope(self, scope: Optional[str]) -> bool
    def engage_manual(self, reason: str = "…") -> KillSwitchDecision
    def reset(self, current_balance: Optional[float] = None, reason: str = "…", force: bool = False) -> KillSwitchDecision
    def note_trade_opened(self) -> None
    def status(self) -> Dict[str, Any]
```

---

## 7. Potential areas of improvement for live deployment

### High priority

- **F-1 ✅ RESOLVED — `max_open_positions` enforced within a tick.** `_tick()` now keeps a
  tick-local `open_count` (seeded from the broker snapshot, incremented on each fill) and
  stops iterating symbols once the cap is reached, so a single tick can no longer open one
  position per symbol.

- **F-2 ✅ RESOLVED — manual lock persists across the daily rollover.** `_rollover_if_needed()`
  now captures and re-applies a `MANUAL` lock after resetting the daily baseline; only the
  daily profit/loss locks reset at the day boundary. Clearing a manual lock still requires an
  explicit `reset(force=True)`.

- **F-3 ✅ RESOLVED — broker timezone alignment.** `DAILY_RESET_TZ` drives the daily baseline
  rollover (via `zoneinfo`); set it to the MetaTrader server timezone so the reset matches the
  broker's trading day instead of UTC. Documented in `.env.example`.

- **F-4 Ephemeral filesystem under Streamlit Cloud.** `.state/` lives on the container's
  local disk, which is wiped on restart/redeploy, so the day's baseline and any lock are
  lost. For production, persist kill-switch state to durable storage (e.g. a small
  DB/KV/object store) rather than a local file.

### Medium priority

- **F-5 Paper simulator ignores contract size and never enforces SL/TP.**
  `_paper_position_view()` computes `profit = (current − open) × direction × volume`,
  which is USD-per-unit and unrealistic for XAUUSD/BTCUSD lots; paper positions also never
  auto-close on SL/TP. So paper mode validates plumbing, not P&L or exits.

- **F-6 No intra-trade management.** SL/TP are placed once at entry; there is no
  break-even move, trailing stop, or time-based exit. All exits depend on the broker
  hitting a resting order.

- **F-7 Singleton freezes the first config.** `get_orchestrator(config)` captures `config`
  only on first creation; later env changes (or a different caller's config) are ignored
  until process restart. Document this, or add an explicit reconfigure path.

- **F-8 `open_market_order` has no retry decorator** (unlike the read methods). This is
  deliberate — retrying a write can double-fill — but it means a transient network error
  during entry is surfaced as a failed trade with no reconciliation against broker state.

### Low priority / hardening

- **F-9 No cap on trades per day** — only the +$60/−$100 P&L locks bound activity; a choppy
  range can churn many small losers before the loss lock trips.
- **F-10 Signal is edge-triggered on the latest bar only.** A missed tick (poll longer than
  the bar) can skip a crossover entirely; consider evaluating the last *closed* bar
  explicitly.
- **F-11 `flattened_for_lock` is per-orchestrator in-memory state**, so a process restart
  while locked will re-attempt a flatten on the next tick (harmless if flat, but worth noting).
- **F-12 `SymbolSpecification.from_raw` defaults** (`tick_size=0.01`, `tick_value=0.01`) can
  produce wrong sizing if MetaApi omits fields for an exotic symbol; a sanity check that
  `loss_per_lot` is plausible would help.

---

## 8. Verified invariants

- Every entry flows through `RiskAgent.assess`; there is no path to `ExecutionAgent.execute`
  that bypasses the kill-switch verdict.
- `reward_risk` can never be configured below 3.0 (clamped in `BotConfig.from_env`).
- Kill-switch P&L uses **equity**, so open floating losses count against the −$100 stop.
- Kill-switch state is account-scoped and written atomically.
- One trading thread per process (singleton), and UI actions marshal onto that thread's loop.
- Credentials are never persisted to disk by the app; the dashboard token is session-only and masked in all displays.
