# GitHub Actions Setup Guide

Your ETH trading system runs on GitHub's cloud for FREE. No laptop. No phone. No card.

## What This Does

Every 5 minutes during London session (07:00-16:00 UTC / 12:30-21:30 IST, Mon-Fri), GitHub:
1. Fetches the latest 5-min candles from Binance
2. Scores them with your 9-agent model (identical logic)
3. Checks filters (score >= 0.5, agree >= 4, London session)
4. If signal fires:
   - Sends Telegram message with entry/stop/target
   - Places order on Binance USDM testnet
5. If in a trade:
   - Checks stop/target/breakeven
   - Closes if hit, sends Telegram alert
6. Saves state back to the repo

## Step 1: Create GitHub Account (2 min)

1. Go to **github.com**
2. Click **Sign up**
3. Enter email, password, username
4. Verify email
5. Free account - no card needed

## Step 2: Train and Save Model (ONCE, on your laptop)

Open Command Prompt and run:

```
cd "C:\Users\Mohammed Maaz\OneDrive\Desktop\Maaz trading system\backend"
python save_model.py
```

This creates `model_weights.pkl` (about 50 KB).

Wait for it to finish (2-3 minutes).

## Step 3: Create GitHub Repository (5 min)

1. Go to **github.com/new**
2. Repository name: `maaz-trading`
3. Select **Public** (required for free unlimited Actions)
4. Check **"Add a README"**
5. Click **Create repository**
6. On the repo page, click **Upload files**
7. Upload these items from your Desktop:

```
github_signal/
  signal_engine.py
  state.json          (create empty: {})
  journal.csv         (create empty file)
  requirements.txt
  .github/workflows/signal.yml
```

AND from `backend/`:
```
backend/
  agents/             (folder)
  features.py
  regime.py
  orchestrator.py
  data_loader.py
  volume_proxy.py
  config.py
  __init__.py
  model_weights.pkl   (from Step 2)
```

## Step 4: Set Secrets (3 min)

In your repo, go to **Settings** -> **Secrets and variables** -> **Actions**

Click **New repository secret** for each:

| Name | Value | Where to get it |
|------|-------|-----------------|
| `BINANCE_KEY` | Your API key | Binance Demo Futures -> API Management |
| `BINANCE_SECRET` | Your API secret | Same screen (shown once when created) |
| `TELEGRAM_TOKEN` | Your bot token | @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Your chat ID | @userinfobot on Telegram |

## Step 5: Set Variables (2 min)

Still in **Settings** -> **Secrets and variables** -> **Actions**, click **Variables** tab:

| Name | Value | Notes |
|------|-------|-------|
| `DRY_RUN` | `true` | Start with true (simulated fills). Change to `false` for real testnet orders |
| `RISK_USDT` | `10` | Risk per trade in USDT |
| `MAX_NOTIONAL` | `100` | Max notional per trade |

## Step 6: Enable Actions (1 min)

1. In your repo, click **Actions** tab (top menu)
2. If you see a green "Enable" button, click it
3. You should see "ETH Signal Engine" workflow listed
4. Click **Run workflow** -> **Run workflow** (manual test)

Wait ~2 minutes. Check the run log.

## Step 7: Verify (2 min)

### Test 1: Manual run
- Go to **Actions** tab -> click the latest run
- Check the log: should show `[engine] ... hour=... DRY_RUN=...`
- If London is open: `[engine] side=... score=...`
- If signal: `SIGNAL SENT` and Telegram message

### Test 2: Check Telegram
- Send `hi` to your bot
- Should reply: `Maaz Trading Bot alive. Commands: /status ...`

### Test 3: Check state
- Repo should show updated `state.json` after each run

## Step 8: Go Live (when ready)

1. Go to **Settings** -> **Variables**
2. Change `DRY_RUN` from `true` to `false`
3. Next run during London session will place real testnet orders

## Schedule

The workflow runs:
- **When:** Every 5 minutes, 07:00-15:55 UTC, Mon-Fri
- **London session:** 07:00-16:00 UTC = 12:30-21:30 IST
- **Outside London:** Workflow skips (idle)

## How to See Results

### On Telegram (instant)
- Signal alerts
- Entry/stop/target prices
- Exit alerts (target hit / stop hit / timeout)
- Breakeven alerts

### In the repo (after each run)
- `state.json` - open positions
- `journal.csv` - all closed trades with R

### On your laptop (optional)
- Open `Maaz trading system` -> double-click `START_server.py`
- The platform shows all trades including GitHub-traded ones
- Historical charts, monthly reports, etc.

## Costs

| Item | Cost |
|------|------|
| GitHub account | Free |
| GitHub Actions (public repo) | **Free, unlimited minutes** |
| Binance USDM testnet | Free (demo money) |
| Telegram bot | Free |
| **Total** | **$0/month** |

## Important Notes

1. **Model is frozen** - the saved weights are from your 2021-2023 training. Same signals as backtest.
2. **No retraining on GitHub** - it loads the pickle, scores live data, exits. Fast (< 60 seconds per run).
3. **Position state persists** - state.json tracks open trades across runs.
4. **Testnet only** - Binance USDM demo money ($0 real risk).
5. **One position per panel** - if panel 1:2 is in a trade, it won't open another until that closes.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Actions not running | Check repo is PUBLIC |
| `ModuleNotFoundError: talib` | The workflow installs TA-Lib; check the "Install TA-Lib" step log |
| No signals during London | Normal - not every bar fires. Average ~1 trade/day for 1:2 panel |
| State not saving | Check the "Commit state" step in Actions log |
| Telegram not sending | Verify secrets: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID |
