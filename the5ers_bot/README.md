# The5ers Multi-Agent Crypto Trading Bot

An automated, prop-firm-compliant trading bot for a **The5ers Bootcamp $5,000** account.
It connects to MetaTrader 5 through the **MetaApi Cloud SDK**, runs a four-stage
multi-agent pipeline, and enforces a **Triple-Lock** daily risk governor
(`+$60` profit cap / `-$100` emergency stop / manual kill switch).

```
MarketDataAgent ──▶ StrategyAgent ──▶ RiskAgent ──▶ ExecutionAgent
   (EMA/RSI/ATR)      (signal)        (kill switch,    (MetaApi / MT5
                                       sizing, SL/TP)    order routing)
                                            ▲
                                     KillSwitch (Triple-Lock)
```

## Files

| File | Purpose |
| --- | --- |
| `broker_client.py` | Async MetaApi / MT5 client: account equity & balance, prices, candles, symbol specs, open/close orders (long **and** short), plus an offline paper-trading engine. |
| `kill_switch.py` | The Triple-Lock daily kill switch: `+$60` profit target cap, `-$100` emergency daily stop loss, persistent manual lock. File-backed and thread-safe. |
| `app.py` | Multi-agent orchestrator + Streamlit dashboard for 24/7 monitoring and operator controls. |
| `requirements.txt` | Python dependencies. |

## Prerequisites

* Python **3.9+** (3.11+ recommended)
* A MetaApi account ([app.metaapi.cloud](https://app.metaapi.cloud)) with your
  The5ers MT5 account attached, **or** your raw MT5 login credentials so the bot can
  provision the MetaApi account for you
* Your The5ers MT5 **server name**, login and password (investor password is enough
  for read-only, but order execution requires the main password)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file (see below)
cp .env.example .env 2>/dev/null || true
```

### `.env` reference

Sensitive values live **only** in `.env` (never in source). Create a `.env` next to
`app.py`:

```dotenv
# ---------------------------------------------------------------------------
# MetaApi credentials
# ---------------------------------------------------------------------------
METAAPI_TOKEN=your-metaapi-token
METAAPI_ACCOUNT_ID=                      # leave empty to provision from MT5_* below
METAAPI_PLATFORM=mt5
# METAAPI_REGION=new-york                # optional

# ---------------------------------------------------------------------------
# Raw MT5 credentials (only needed if METAAPI_ACCOUNT_ID is empty)
# ---------------------------------------------------------------------------
MT5_LOGIN=12345678
MT5_PASSWORD=your-mt5-password
MT5_SERVER=The5ers-Live
MT5_MAGIC=520052

# ---------------------------------------------------------------------------
# Instruments / strategy
# ---------------------------------------------------------------------------
TRADING_SYMBOLS=BTCUSD,ETHUSD
SYMBOL_SUFFIX=                           # e.g. ".a" if your broker uses suffixes
STRATEGY_TIMEFRAME=15m
CANDLE_LIMIT=200
POLL_INTERVAL_SECONDS=60

EMA_FAST=21
EMA_SLOW=55
RSI_PERIOD=14
RSI_LONG_MIN=52
RSI_SHORT_MAX=48
ATR_PERIOD=14
ATR_STOP_MULTIPLIER=2.0
REWARD_RISK=1.8
RISK_PER_TRADE_PCT=0.5
MAX_OPEN_POSITIONS=2

# ---------------------------------------------------------------------------
# Triple-Lock kill switch
# ---------------------------------------------------------------------------
DAILY_PROFIT_TARGET=60                   # Lock 1: bank the win at +$60
DAILY_MAX_LOSS=100                       # Lock 2: hard stop at -$100
DAILY_RESET_TZ=UTC                       # daily rollover timezone
KILL_SWITCH_AUTO_CLOSE=true              # flatten positions when a lock trips
KILL_SWITCH_STATE_FILE=.state/kill_switch_state.json

# ---------------------------------------------------------------------------
# Safety switches
# ---------------------------------------------------------------------------
DRY_RUN=true                             # log orders, send nothing
PAPER_MODE=false                         # true = fully offline simulator
LOG_LEVEL=INFO

# Paper-mode only
PAPER_START_BALANCE=5000
PAPER_BASE_PRICES=BTCUSD:65000,ETHUSD:3200
```

## Running

```bash
# Dashboard (24/7 monitoring)
streamlit run app.py

# Headless single pass - useful as a smoke test or a cron entry point
python app.py --once

# Standalone broker connectivity check
python broker_client.py

# Kill switch logic demo
python kill_switch.py
```

### Safe first run

Set `PAPER_MODE=true` first to exercise the full pipeline with synthetic prices and
no broker connection. Then set `PAPER_MODE=false` and `DRY_RUN=true` against the real
account to confirm signals, sizing and the kill switch behave as expected. Only then
set `DRY_RUN=false`.

## How the Triple-Lock works

| Lock | Trigger | Behaviour | Clears when |
| --- | --- | --- | --- |
| 1 — Profit cap | Day P&L ≥ **+$60** | Stops opening new trades, banks the win | New trading day (or operator reset) |
| 2 — Max loss | Day P&L ≤ **−$100** | Hard stop for the rest of the day | New trading day (or operator reset) |
| 3 — Manual | Operator presses the kill switch | Stops trading and (optionally) flattens | Only an explicit `force` reset |

* P&L is measured as `equity − day_start_balance`, i.e. realised **plus** floating,
  which is how prop firms evaluate daily drawdown.
* State is persisted to `KILL_SWITCH_STATE_FILE`, so a restart never forgets a lock.
* The daily baseline rolls over automatically at `DAILY_RESET_TZ` midnight.
* Position sizing never risks more than `RISK_PER_TRADE_PCT` of equity **and** never
  more than the remaining daily loss budget. If the minimum lot size would exceed the
  remaining budget, the trade is rejected rather than downgraded.

## Dashboard controls

* **Start / Stop** — starts or stops the background trading thread.
* **Emergency close all** — flattens every open position immediately.
* **Reset kill switch** — re-arms the day from the current balance (requires the
  confirmation checkbox).
* Live metrics: equity, balance, day P&L, progress to the `+$60` target, remaining
  risk budget, open positions, signals and trade log.

## Security notes

* No credentials are hard-coded; everything is read from environment variables.
* `.env` and `.state/` should be git-ignored — add them to your `.gitignore`.
* The kill switch reset intentionally requires `force=True`, so it cannot be cleared
  accidentally by a stray code path.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `METAAPI_TOKEN is missing` | Populate `.env` or set `PAPER_MODE=true`. |
| `metaapi_cloud_sdk is not installed` | `pip install -r requirements.txt`. |
| Symbols never trade | Check `SYMBOL_SUFFIX` matches your broker (e.g. `BTCUSD.a`). |
| `Risk budget ... below the minimum lot risk` | Lower `RISK_PER_TRADE_PCT` won't help; use a smaller-volume instrument or a larger account. |
| Kill switch stuck locked | Use the dashboard reset with the confirmation checkbox, or delete the state file before starting. |

## Disclaimer

Trading leveraged instruments carries substantial risk. This software is provided as
engineering tooling, not financial advice. Always validate behaviour in paper mode and
on a demo account before risking capital, and confirm that automated trading is
permitted by your prop firm's terms of service.
