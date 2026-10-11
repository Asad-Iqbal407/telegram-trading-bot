# Telegram trading bot

Automated crypto trading (via [ccxt](https://github.com/ccxt/ccxt)) controlled from Telegram:
5-minute candles, 7 strategies, optional leverage with long **and** short positions,
volatility-based stops, risk-based position sizing and a backtester that uses the same rules as
the live engine. **Starts in paper mode (simulated money).**

> **Risk warning.** Leverage multiplies losses as well as gains: at 10x a 10% adverse move wipes out the
> margin. Short-timeframe strategies pay fees on every trade and most lose money after costs. Nothing
> here is financial advice or a promise of profit. Paper trade for weeks, then risk only what you can lose.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # fill in TELEGRAM_BOT_TOKEN (from @BotFather)
python main.py              # send /myid to the bot, put the id in ALLOWED_CHAT_IDS, restart
```
Then in Telegram: `/backtest 3000 all` → `/strategy ...` → `/startbot`.

## Commands
| | |
|---|---|
| `/status` `/balance` `/trades` | state, equity, last trades |
| `/startbot` `/stopbot` | automatic trading on/off |
| `/long [margin]` `/short [margin]` `/close` | manual trades (no margin = auto-sized) |
| `/strategies` `/strategy <name>` | list / switch strategy |
| `/backtest [candles] [name\|all]` | e.g. `/backtest 3000 all` (up to 5000 candles) |
| `/leverage <x>` | leverage for new trades (capped by `MAX_LEVERAGE`) |
| `/set <size\|risk\|sl\|tp\|atrsl\|atrtp\|fast\|slow> <v>` | tweak parameters (reset on restart) |

## Strategies (all evaluated on closed candles, no look-ahead – covered by tests)
| name | idea |
|---|---|
| `ensemble` (default) | trade only when 2 of 3 trend models agree and ADX > 20 |
| `ema_trend` | EMA 9/21 cross in the direction of the EMA 200 trend |
| `supertrend` | ATR trailing trend follower |
| `macd_trend` | MACD histogram zero-cross with EMA 200 filter |
| `bb_reversion` | fade Bollinger-band extremes with RSI, only in ranging markets (ADX < 25) |
| `breakout` | Donchian 20 breakout confirmed by volume |
| `sma_rsi` | SMA cross with RSI filter |

Trend followers do badly in sideways markets and the reverter does badly in strong trends – use
`/backtest N all` on the market and period you care about, and expect results to change.

## How leverage and risk work
- `MARKET_TYPE=futures` enables leverage and shorting; `spot` is long-only, 1x.
- **Position size:** with `RISK_PER_TRADE_PCT=1` the size is chosen so that hitting the stop loses ≈1% of
  equity, capped by `MAX_MARGIN_PCT`.
- **Stops:** `SL_ATR_MULT` × ATR below/above entry (bounded by `MIN_STOP_PCT` and `STOP_LOSS_PCT`), target
  `TP_ATR_MULT` × ATR. Stops and liquidation are checked against candle wicks, worst case first.
- **Liquidation guard:** the bot refuses any leverage whose liquidation distance is not comfortably beyond
  the stop (e.g. 25x with a 5% stop), and `MAX_LEVERAGE` can never be exceeded from Telegram.
- **Daily loss limit** closes the position and halts trading until the next UTC day; `MAX_DAILY_TRADES` limits churn.
- **Live futures:** isolated margin, leverage set per symbol, and an exchange-side reduce-only stop-loss is
  placed right after entry. If that stop can't be placed the position is closed again immediately.
  Live leverage needs `LIVE_CONFIRM=I_ACCEPT_THE_RISK` **and** `LIVE_LEVERAGE_CONFIRM=I_UNDERSTAND_LIQUIDATION`.
- Only chat ids in `ALLOWED_CHAT_IDS` can control the bot. Use API keys **without withdrawal rights**.

## Running 24/7
The bot is a long-running process (polls every `POLL_SECONDS`, keeps state in a file), so it needs an
always-on host. **Vercel / Netlify / other serverless platforms are not suitable**: functions are killed after seconds,
there is no persistent disk and no background loop.

Use a small VPS or a container host (Railway, Fly.io, Render background worker, Hetzner/DigitalOcean/Oracle VM…):
1. Deploy this repo with the included `Dockerfile` (or `pip install -r requirements.txt && python main.py` under systemd/pm2).
2. Set the variables from `.env.example` in the host's **environment/secrets** settings – never commit `.env`.
3. Attach a **persistent volume** mounted at `/data` (the Dockerfile sets `STATE_FILE=/data/state.json`);
   without it the open position and paper balance are lost on every redeploy.
4. Set `AUTOSTART=true` so trading resumes after a restart, and run exactly **one** instance
   (two instances with the same token fight over Telegram updates and could double-trade).
5. Pick a region where your exchange is available (Binance blocks e.g. US IPs). For live trading, whitelist the
   server's static IP on the API key if the host offers one.

## Honest limitations
- The backtester ignores funding fees, partial fills, outages and the daily loss limit, and assumes fills at
  candle open ± slippage. Real results will be worse.
- Live trading code is tested against a fake exchange only. Run paper mode first, then go live with tiny size.
- One symbol and one position at a time. The bot needs to stay running (use a VPS for 24/7).

## Layout
`tgtrader/` – `strategy.py`, `indicators.py`, `risk.py`, `broker.py` (paper/live), `engine.py` (loop),
`backtest.py`, `bot.py` (Telegram), `config.py`. Tests: `python -m pytest`.
