import asyncio
import json
import os
import time
from datetime import datetime, timezone

from .indicators import atr
from .risk import direction, entry_margin, liq_price, stop_fractions, stop_prices
from .strategy import build_strategy


class Engine:
    """Polls the market, applies the strategy and risk rules, places orders.
    One position at a time (long or short)."""

    def __init__(self, cfg, market, broker, strategy, notify):
        self.cfg, self.market, self.broker = cfg, market, broker
        self.strategy, self.notify = strategy, notify
        self.running = False
        self.halted_reason = ""
        self._task = None
        self.position = None  # side, qty, entry, margin, leverage, fee, sl, tp, liq, ts
        self.trades = []
        self.last_candle = None
        self.day = None
        self.day_start_equity = None
        self.day_trades = 0
        self._load()

    # ---------- persistence ----------
    def _identity(self) -> dict:
        return dict(mode=self.cfg.mode, market_type=self.cfg.market_type, symbol=self.cfg.symbol)

    def _load(self):
        if not os.path.exists(self.cfg.state_file):
            return
        with open(self.cfg.state_file) as f:
            st = json.load(f)
        if st.get("identity") != self._identity():  # never mix paper/live or symbols
            return
        self.position = st.get("position")
        self.trades = st.get("trades", [])
        self.last_candle = st.get("last_candle")
        self.day = st.get("day")
        self.day_start_equity = st.get("day_start_equity")
        self.day_trades = st.get("day_trades", 0)
        if st.get("broker") and hasattr(self.broker, "set_state"):
            self.broker.set_state(st["broker"])

    def _save(self):
        st = dict(identity=self._identity(), position=self.position, trades=self.trades[-200:],
                  last_candle=self.last_candle, day=self.day, day_start_equity=self.day_start_equity,
                  day_trades=self.day_trades)
        if hasattr(self.broker, "get_state"):
            st["broker"] = self.broker.get_state()
        tmp = self.cfg.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, self.cfg.state_file)

    # ---------- helpers ----------
    @staticmethod
    def unrealized(pos, price: float) -> float:
        return direction(pos["side"]) * (price - pos["entry"]) * pos["qty"]

    async def equity(self, price: float) -> float:
        w = await asyncio.to_thread(self.broker.wallet)
        if self.position:
            if self.broker.margin_out_of_wallet:
                w += self.position["margin"]
            w += self.unrealized(self.position, price)
        return w

    def _record(self, action, side, qty, price, fee, reason, pnl=None):
        self.trades.append(dict(ts=time.time(), action=action, side=side, qty=qty, price=price,
                                fee=fee, reason=reason, pnl=pnl))

    async def _fetch(self):
        limit = min(1000, max(300, self.strategy.min_candles + 60))
        return await asyncio.to_thread(self.market.ohlcv, self.cfg.symbol, self.cfg.timeframe, limit)

    # ---------- trading actions ----------
    async def open_position(self, side: str, margin: float = None, reason: str = "manual",
                            atr_value: float = None) -> str:
        cfg = self.cfg
        if self.position:
            return "A position is already open."
        if self.halted_reason:
            return f"Trading halted: {self.halted_reason}"
        if side == "short" and not cfg.allow_short:
            return "Shorting needs MARKET_TYPE=futures."
        if self.day_trades >= cfg.max_daily_trades:
            return f"Daily trade limit reached ({cfg.max_daily_trades})."
        if atr_value is None:
            df = await self._fetch()
            atr_value = float(atr(df, 14).iloc[-1])
            price = float(df["close"].iloc[-1])
        else:
            price = await asyncio.to_thread(self.market.price, cfg.symbol)
        sl_frac, tp_frac = stop_fractions(price, atr_value, cfg)
        equity = await self.equity(price)
        cap = equity * cfg.max_margin_pct / 100
        if margin is None:
            margin = entry_margin(equity, sl_frac, cfg)
        elif margin > cap:
            return f"Refused: margin {margin:.2f} exceeds MAX_MARGIN_PCT ({cfg.max_margin_pct:g}% = {cap:.2f})."
        if margin < 1:
            return "Refused: position size too small."
        sl0, _ = stop_prices(side, price, sl_frac, tp_frac)
        fill = await asyncio.to_thread(self.broker.open, cfg.symbol, side, margin, cfg.leverage, sl0)
        sl, tp = stop_prices(side, fill.price, sl_frac, tp_frac)
        self.position = dict(side=side, qty=fill.qty, entry=fill.price, margin=margin,
                             leverage=cfg.leverage, fee=fill.fee, sl=sl, tp=tp, ts=fill.ts,
                             liq=liq_price(side, fill.price, cfg.leverage) if cfg.leverage > 1 else None)
        self.day_trades += 1
        self._record("open", side, fill.qty, fill.price, fill.fee, reason)
        self._save()
        liq = f" liq {self.position['liq']:.2f}" if self.position["liq"] else ""
        msg = (f"{'🟢' if side == 'long' else '🟣'} {side.upper()} {fill.qty:.6f} {cfg.base} @ {fill.price:.2f} "
               f"| {cfg.leverage:g}x margin {margin:.2f} | SL {sl:.2f} TP {tp:.2f}{liq} ({reason}) [{self.broker.name}]")
        await self.notify(msg)
        return msg

    async def close_position(self, reason: str, price_hint: float = None, liquidated: bool = False) -> str:
        if not self.position:
            return "No open position."
        pos, d = self.position, direction(self.position["side"])
        if liquidated:
            await asyncio.to_thread(self.broker.liquidate, self.cfg.symbol)
            price, qty, fee, pnl = pos["liq"], pos["qty"], 0.0, -pos["margin"] - pos["fee"]
        else:
            fill = await asyncio.to_thread(self.broker.close, self.cfg.symbol, pos["qty"], price_hint)
            if fill is None:   # the exchange already closed it (stop-loss or liquidation)
                price = await asyncio.to_thread(self.market.price, self.cfg.symbol)
                qty, fee = pos["qty"], 0.0
                pnl = d * (price - pos["entry"]) * qty - pos["fee"]
                reason = f"closed on exchange (stop/liquidation), pnl approximate; {reason}"
            else:
                price, qty, fee = fill.price, fill.qty, fill.fee
                pnl = d * (price - pos["entry"]) * qty - pos["fee"] - fee
        self.position = None
        self._record("close", pos["side"], qty, price, fee, reason, pnl)
        self._save()
        msg = (f"{'💥' if liquidated else '🔴'} CLOSE {pos['side'].upper()} {qty:.6f} {self.cfg.base} @ {price:.2f} "
               f"({reason}) PnL {pnl:+.2f} {self.cfg.quote} [{self.broker.name}]")
        await self.notify(msg)
        return msg

    # ---------- main logic ----------
    async def _check_exits(self, df, price: float) -> bool:
        """Stop / target / liquidation, using candle wicks since entry. Worst case first."""
        p = self.position
        lo = hi = price
        entry_ms = int(p["ts"] * 1000)
        for ts, l, h in zip(df["ts"].tail(4), df["low"].tail(4), df["high"].tail(4)):
            if ts > entry_ms:   # candle opened after entry -> whole range is post-entry
                lo, hi = min(lo, l), max(hi, h)
        long = p["side"] == "long"
        worst, best = (lo, hi) if long else (hi, lo)
        beyond = (lambda a, b: a <= b) if long else (lambda a, b: a >= b)
        if p["liq"] and beyond(worst, p["liq"]) and self.broker.name == "paper":
            await self.close_position("LIQUIDATED", liquidated=True)
        elif beyond(worst, p["sl"]):
            await self.close_position("stop-loss", price_hint=p["sl"])
        elif beyond(p["tp"], best):
            await self.close_position("take-profit", price_hint=p["tp"])
        else:
            return False
        return True

    async def tick(self):
        cfg = self.cfg
        df = await self._fetch()
        price = float(df["close"].iloc[-1])

        if self.position and self.broker.name == "live" and cfg.futures:
            size = await asyncio.to_thread(self.broker.position_qty, cfg.symbol)
            if size == 0:
                await self.close_position("reconcile")
                return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        equity = await self.equity(price)
        if self.day != today:
            self.day, self.day_start_equity, self.halted_reason, self.day_trades = today, equity, "", 0
            self._save()

        if self.day_start_equity and not self.halted_reason:
            drop = (self.day_start_equity - equity) / self.day_start_equity * 100
            if drop >= cfg.daily_loss_limit_pct:
                self.halted_reason = f"daily loss limit hit (-{drop:.1f}%)"
                await self.notify(f"⛔ Trading halted: {self.halted_reason}")
                await self.close_position("daily loss limit")
                return

        if self.position and await self._check_exits(df, price):
            return

        closed = df.iloc[:-1]  # the last candle is still forming
        if len(closed) < self.strategy.min_candles:
            return
        candle_ts = int(closed["ts"].iloc[-1])
        if candle_ts == self.last_candle:
            return
        self.last_candle = candle_ts
        self._save()

        sig = self.strategy.signal(closed)
        if not sig:
            return
        if sig == "short" and not cfg.allow_short:
            sig = "exit_long"
        pos = self.position
        if sig in ("exit_long", "exit_short"):
            if pos and pos["side"] == sig.split("_")[1]:
                await self.close_position(f"signal:{self.strategy.name}")
        elif not self.halted_reason:
            if pos and pos["side"] != sig:
                await self.close_position(f"signal flip:{self.strategy.name}")
            if not self.position:
                await self.open_position(sig, None, f"signal:{self.strategy.name}",
                                         float(atr(closed, 14).iloc[-1]))

    async def _loop(self):
        while self.running:
            try:
                await self.tick()
            except Exception as e:  # keep the bot alive on network/exchange errors
                await self.notify(f"⚠️ Error: {type(e).__name__}: {e}")
                await asyncio.sleep(max(self.cfg.poll_seconds, 60))
                continue
            await asyncio.sleep(self.cfg.poll_seconds)

    def start(self):
        if self.running:
            return False
        self.running = True
        self._task = asyncio.create_task(self._loop())
        return True

    def stop(self):
        if not self.running:
            return False
        self.running = False
        if self._task:
            self._task.cancel()
        return True

    def set_strategy(self, name: str):
        self.strategy = build_strategy(name, self.cfg)
        self.cfg.strategy = name

    async def status(self) -> str:
        c = self.cfg
        price = await asyncio.to_thread(self.market.price, c.symbol)
        equity = await self.equity(price)
        lines = [f"Mode: {self.broker.name.upper()} {c.market_type} | Bot: {'RUNNING' if self.running else 'stopped'}",
                 f"{c.symbol} {c.timeframe} | price {price:.2f}",
                 f"Strategy: {self.strategy.name} | leverage {c.leverage:g}x",
                 f"Equity: {equity:.2f} {c.quote} | trades today: {self.day_trades}/{c.max_daily_trades}"]
        p = self.position
        if p:
            pnl = self.unrealized(p, price)
            liq = f" | liq {p['liq']:.2f}" if p["liq"] else ""
            lines.append(f"Position: {p['side'].upper()} {p['qty']:.6f} @ {p['entry']:.2f} | "
                         f"PnL {pnl:+.2f} ({pnl / p['margin'] * 100:+.1f}% of margin)")
            lines.append(f"SL {p['sl']:.2f} | TP {p['tp']:.2f}{liq}")
        else:
            lines.append("Position: none")
        if self.halted_reason:
            lines.append(f"⛔ Halted: {self.halted_reason}")
        return "\n".join(lines)
