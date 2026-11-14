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

Sensitive values live in `.env` (never in source). The MetaApi token and account id can
alternatively be entered at runtime in the dashboard sidebar — see
[Dashboard controls](#dashboard-controls). Create a `.env` next to `app.py`:

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
REWARD_RISK=3.0                          # strict minimum 1:3 (smaller values are clamped up)
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

Community Cloud always runs from the repository root, so run it the same way locally
(`streamlit run` interprets most paths relative to the working directory):

```bash
# From the repository root (matches Streamlit Community Cloud)
streamlit run the5ers_bot/app.py

# Headless single pass - useful as a smoke test or a cron entry point
python the5ers_bot/app.py --once

# Standalone broker connectivity check / kill switch demo
python the5ers_bot/broker_client.py
python the5ers_bot/kill_switch.py
```

If you `cd the5ers_bot` first, drop the folder prefix (`streamlit run app.py`).

### Cloud deployment (Streamlit Community Cloud)

The dashboard is ready to deploy on [Community Cloud](https://share.streamlit.io) from
GitHub with **no code changes**:

1. Push this repository to GitHub (public, or private with the app granted access).
2. On <https://share.streamlit.io> choose **Create app → Deploy a public app from
   GitHub** and select your repository and branch.
3. **Main file path:** `the5ers_bot/app.py`
4. Under **Advanced settings → Secrets**, add either your MetaApi credentials …

   ```toml
   METAAPI_TOKEN = "your-metaapi-token"
   METAAPI_ACCOUNT_ID = "your-account-id"
   ```

   … or set `PAPER_MODE = "true"` to demo the dashboard against the offline simulator.
   You can also leave Secrets empty and type the credentials into the sidebar at
   runtime — they are kept in the session only.
5. Click **Deploy**. Community Cloud installs `the5ers_bot/requirements.txt` and reads
   the configuration from the repository-root `.streamlit/config.toml`.

**Why the nested layout works on Cloud:**

* Community Cloud always runs from the **repository root**. `app.py` prepends its own
  folder to `sys.path`, so `broker_client` and `kill_switch` import cleanly regardless
  of the working directory.
* The dependency file in the **entrypoint's directory** (`the5ers_bot/requirements.txt`)
  takes precedence over the repository-root `requirements.txt`, so the TradingAgents
  project's dependencies are ignored.
* The configuration file **must be at the repository root** (`.streamlit/config.toml`);
  an app-specific copy next to the entrypoint would override it. No port/host or
  CORS/XSRF overrides are set, so nothing conflicts with the platform proxy.
* Credentials resolve in this order: **sidebar inputs → `st.secrets` → environment
  variables (`.env`)**. The "in effect" caption always shows which source won.

> **State caveat:** Community Cloud containers are ephemeral and sleep when idle, so the
> kill-switch state file (`.state/`) and the running background thread do not survive a
> reboot or redeploy. Use Community Cloud for monitoring and supervised sessions; run on
> a persistent VM if you need truly always-on trading and hard state guarantees.

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
* The state file is bound to a single MetaApi **account id**. Switching accounts clears
  the daily state, so a `$5,000` baseline can never be carried onto a `$10,000` account
  and instantly trip the profit cap.
* Position sizing never risks more than `RISK_PER_TRADE_PCT` of equity **and** never
  more than the remaining daily loss budget. If the minimum lot size would exceed the
  remaining budget, the trade is rejected rather than downgraded.
* Every setup must offer a strict **minimum 1:3** reward:risk. `REWARD_RISK` is clamped
  up to `1:3` at runtime, so the bot can never emit a worse-than-1:3 trade.
* On a lock trip the bot **auto-flattens all open positions by default**
  (`KILL_SWITCH_AUTO_CLOSE=true`) and stops opening new trades for the rest of the day.

## Dashboard controls

**🔐 MetaApi Credentials** (sidebar)

* Enter your **MetaApi Token** (masked, password-style) and **MetaApi Account ID**
  directly in the dashboard, then press **Save credentials**.
* Pressing **Start Bot** also picks up whatever is currently typed in those fields.
* If a field is left **empty**, the bot falls back to `st.secrets` (Streamlit Community
  Cloud Secrets) and then to `METAAPI_TOKEN` / `METAAPI_ACCOUNT_ID` from `.env` — so all
  workflows are supported.
* The caption under the inputs always shows what is **in effect** (token masked, account
  id, and the winning source: Dashboard, Streamlit secrets, or Environment).
* Credentials are held in the browser session's memory only and are **never written to
  disk**. They are applied on the next **Start Bot**; changing them while the bot is
  running shows a reminder to restart.

**🎛️ Controls**

* **Start Bot / Stop Bot** — a single switcher that toggles the background trading thread.
* **Emergency Close All Positions** — flattens every open position immediately.
* **🔓 Reset Kill-Switch** — re-arms the day from the current balance and is disabled
  until the safety confirmation checkbox is ticked.

**📊 Metric cards**

* **Daily P&L** with a progress bar towards the `+$60` daily cap, plus the remaining
  amount to target and the remaining daily risk budget.
* **Balance** and **Equity** (equity shows floating P&L).
* **Kill-Switch** status — `ACTIVE`, `ARMED · IDLE`, `LOCKED · MANUAL`,
  `LOCKED · TARGET BANKED`, or `EMERGENCY STOPPED` — mirrored by a coloured banner.
* Operational row: open positions, trades today, loop state, broker connection, last tick.

## Security notes

* No credentials are hard-coded; they come from the dashboard inputs, `st.secrets`
  (Cloud Secrets manager) or environment variables.
* `app.py` keeps a single process-wide orchestrator, so Streamlit reruns and extra
  browser sessions can never spawn duplicate background trading threads.
* Dashboard-entered tokens live in Streamlit session memory only — they are never
  persisted to disk and never rendered back in full (always masked).
* `.env` and `.state/` should be git-ignored — add them to your `.gitignore`.
* The kill switch reset intentionally requires `force=True`, so it cannot be cleared
  accidentally by a stray code path.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `METAAPI_TOKEN is missing` | Enter it in the dashboard's **MetaApi Credentials** panel, populate `.env`, or set `PAPER_MODE=true`. |
| `metaapi_cloud_sdk is not installed` | `pip install -r requirements.txt`. |
| Symbols never trade | Check `SYMBOL_SUFFIX` matches your broker (e.g. `BTCUSD.a`). |
| `Risk budget ... below the minimum lot risk` | Lower `RISK_PER_TRADE_PCT` won't help; use a smaller-volume instrument or a larger account. |
| Kill switch stuck locked | Use the dashboard reset with the confirmation checkbox, or delete the state file before starting. |

## Disclaimer

Trading leveraged instruments carries substantial risk. This software is provided as
engineering tooling, not financial advice. Always validate behaviour in paper mode and
on a demo account before risking capital, and confirm that automated trading is
permitted by your prop firm's terms of service.
