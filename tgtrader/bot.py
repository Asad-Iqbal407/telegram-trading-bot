import asyncio
import functools
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from .backtest import backtest
from .engine import Engine
from .strategy import STRATEGIES, build_strategy

HELP = """Trading
/status – mode, price, equity, position
/balance – account balance
/startbot, /stopbot – automatic trading on/off
/long [margin], /short [margin] – open manually (no margin = auto size)
/close – close the open position
/trades – last 10 trades

Strategy & research
/strategies – list strategies
/strategy <name> – switch strategy
/backtest [candles] [name|all] – e.g. /backtest 3000 all
/leverage <x> – leverage for NEW trades
/set <size|risk|sl|tp|atrsl|atrtp|fast|slow> <value>

/myid, /help"""

SETTINGS = dict(size="trade_size_quote", risk="risk_per_trade_pct", sl="stop_loss_pct", tp="take_profit_pct",
                atrsl="sl_atr_mult", atrtp="tp_atr_mult", fast="fast_sma", slow="slow_sma")


def authorized(fn):
    @functools.wraps(fn)
    async def wrapper(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat.id not in self.cfg.allowed_chat_ids:
            await update.message.reply_text("Not authorized.")
            return
        try:
            await fn(self, update, ctx)
        except Exception as e:
            await update.message.reply_text(f"⚠️ {type(e).__name__}: {e}")
    return wrapper


def fmt_backtest(name, r) -> str:
    pf = "inf" if r["profit_factor"] == float("inf") else f"{r['profit_factor']:.2f}"
    return (f"{name}: {r['return_pct']:+.1f}% | DD {r['max_drawdown_pct']:.1f}% | "
            f"{r['trades']} trades, win {r['win_rate']:.0f}%, PF {pf}"
            + (f" | 💥{r['liquidations']} liquidated" if r["liquidations"] else ""))


class TradingBot:
    def __init__(self, cfg, market, broker, strategy):
        self.cfg, self.market, self.broker = cfg, market, broker
        self.app = Application.builder().token(cfg.telegram_token).post_init(self._post_init).build()
        self.engine = Engine(cfg, market, broker, strategy, self.notify)
        handlers = dict(start=self.help, help=self.help, status=self.status, balance=self.balance,
                        startbot=self.startbot, stopbot=self.stopbot, long=self.long, short=self.short,
                        buy=self.long, sell=self.close, close=self.close, trades=self.trades,
                        strategies=self.strategies, strategy=self.strategy, backtest=self.backtest,
                        leverage=self.leverage, set=self.set_param)
        for name, fn in handlers.items():
            self.app.add_handler(CommandHandler(name, fn))
        self.app.add_handler(CommandHandler("myid", self.myid))

    async def _post_init(self, app):
        if self.cfg.autostart:
            self.engine.start()
        await self.notify(f"🤖 Bot online: {self.broker.name.upper()} {self.cfg.market_type}, "
                          f"{self.cfg.timeframe}, {self.cfg.leverage:g}x, {self.engine.strategy.name}, "
                          f"auto-trading {'ON' if self.engine.running else 'OFF (/startbot)'}. /help")

    async def notify(self, text: str):
        for chat in self.cfg.allowed_chat_ids:
            try:
                await self.app.bot.send_message(chat, text)
            except Exception:
                pass

    def run(self):
        self.app.run_polling()

    async def myid(self, update, ctx):
        await update.message.reply_text(f"Your chat id: {update.effective_chat.id}")

    @authorized
    async def help(self, update, ctx):
        await update.message.reply_text(HELP)

    @authorized
    async def status(self, update, ctx):
        await update.message.reply_text(await self.engine.status())

    @authorized
    async def balance(self, update, ctx):
        e = self.engine
        price = await asyncio.to_thread(self.market.price, self.cfg.symbol)
        wallet = await asyncio.to_thread(self.broker.wallet)
        equity = await e.equity(price)
        await update.message.reply_text(f"[{self.broker.name}] wallet {wallet:.2f} {self.cfg.quote}\n"
                                        f"equity (incl. open PnL) {equity:.2f} {self.cfg.quote}")

    @authorized
    async def startbot(self, update, ctx):
        await update.message.reply_text("▶️ Bot started." if self.engine.start() else "Already running.")

    @authorized
    async def stopbot(self, update, ctx):
        await update.message.reply_text("⏹ Bot stopped (open position stays open)."
                                        if self.engine.stop() else "Not running.")

    async def _open(self, update, ctx, side):
        margin = float(ctx.args[0]) if ctx.args else None
        if margin is not None and margin <= 0:
            raise ValueError("margin must be positive")
        msg = await self.engine.open_position(side, margin, "manual")
        if not msg.startswith(("🟢", "🟣")):   # success is already announced via notify()
            await update.message.reply_text(msg)

    @authorized
    async def long(self, update, ctx):
        await self._open(update, ctx, "long")

    @authorized
    async def short(self, update, ctx):
        await self._open(update, ctx, "short")

    @authorized
    async def close(self, update, ctx):
        msg = await self.engine.close_position("manual")
        if msg == "No open position.":
            await update.message.reply_text(msg)

    @authorized
    async def trades(self, update, ctx):
        rows = self.engine.trades[-10:]
        if not rows:
            await update.message.reply_text("No trades yet.")
            return
        lines = []
        for t in rows:
            when = datetime.fromtimestamp(t["ts"], timezone.utc).strftime("%m-%d %H:%M")
            pnl = f" pnl {t['pnl']:+.2f}" if t.get("pnl") is not None else ""
            lines.append(f"{when} {t['action'].upper()} {t['side']} {t['qty']:.5f} @ {t['price']:.2f}{pnl} ({t['reason'][:40]})")
        await update.message.reply_text("\n".join(lines))

    @authorized
    async def strategies(self, update, ctx):
        cur = self.engine.strategy.name
        await update.message.reply_text("\n".join(
            f"{'👉 ' if n == cur else '• '}{n}: {c.description}" for n, c in STRATEGIES.items()))

    @authorized
    async def strategy(self, update, ctx):
        if not ctx.args:
            await update.message.reply_text(f"Current: {self.engine.strategy.name}. Options: {', '.join(STRATEGIES)}")
            return
        self.engine.set_strategy(ctx.args[0].lower())
        await update.message.reply_text(f"✅ Strategy: {self.engine.strategy.name} (acts from the next closed candle)")

    @authorized
    async def backtest(self, update, ctx):
        args = list(ctx.args)
        n = int(args.pop(0)) if args and args[0].isdigit() else 2000
        which = args[0].lower() if args else self.engine.strategy.name
        n = max(400, min(n, 5000))
        await update.message.reply_text(f"Fetching {n} × {self.cfg.timeframe} candles…")
        df = await asyncio.to_thread(self.market.history, self.cfg.symbol, self.cfg.timeframe, n)
        names = list(STRATEGIES) if which == "all" else [which]
        out = [f"Backtest {self.cfg.symbol} {self.cfg.timeframe}, {len(df)} candles, "
               f"{self.cfg.leverage:g}x, fees {self.cfg.fee_rate * 100:.3f}% + slippage {self.cfg.slippage_pct:g}%"]
        for name in names:
            strat = self.engine.strategy if name == self.engine.strategy.name else build_strategy(name, self.cfg)
            r = await asyncio.to_thread(backtest, df, strat, self.cfg)
            out.append(fmt_backtest(name, r))
        out.append(f"Buy&hold: {r['buy_hold_pct']:+.1f}%\n"
                   "Optimistic by design (no funding fees, ideal fills). Past results don't predict the future.")
        await update.message.reply_text("\n".join(out))

    @authorized
    async def leverage(self, update, ctx):
        if len(ctx.args) != 1:
            await update.message.reply_text(f"Leverage now {self.cfg.leverage:g}x (max {self.cfg.max_leverage:g}x). "
                                            "Usage: /leverage <x>")
            return
        x = float(ctx.args[0])
        self.cfg.validate_leverage(x)
        self.cfg.leverage = x
        await update.message.reply_text(f"✅ Leverage {x:g}x for new trades (resets on restart; edit .env to persist)")

    @authorized
    async def set_param(self, update, ctx):
        if len(ctx.args) != 2 or ctx.args[0].lower() not in SETTINGS:
            await update.message.reply_text("Usage: /set <" + "|".join(SETTINGS) + "> <value>")
            return
        key, val = ctx.args[0].lower(), float(ctx.args[1])
        attr = SETTINGS[key]
        if val < 0 or (val == 0 and key in ("size", "sl", "tp", "fast", "slow")):
            raise ValueError("invalid value")
        old = getattr(self.cfg, attr)
        setattr(self.cfg, attr, int(val) if key in ("fast", "slow") else val)
        try:
            if self.cfg.fast_sma >= self.cfg.slow_sma:
                raise ValueError("fast must stay below slow")
            if key == "risk" and val > 5:
                raise ValueError("risk is capped at 5% per trade")
            self.cfg.validate_leverage()
        except ValueError:
            setattr(self.cfg, attr, old)
            raise
        if key in ("fast", "slow") and self.engine.strategy.name == "sma_rsi":
            self.engine.set_strategy("sma_rsi")
        await update.message.reply_text(f"✅ {key} = {val:g} (resets on restart; edit .env to persist)")
