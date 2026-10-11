import time
from dataclasses import dataclass

import pandas as pd

from .risk import direction


@dataclass
class Fill:
    side: str       # "long" / "short" (the position side)
    symbol: str
    qty: float
    price: float
    fee: float      # quote currency
    ts: float


class Market:
    """Market data (and order access when keys are given) via ccxt."""

    def __init__(self, exchange_id: str, api_key: str = "", secret: str = "", futures: bool = True):
        import ccxt

        params = {"enableRateLimit": True}
        if api_key:
            params.update(apiKey=api_key, secret=secret)
        if futures and not exchange_id.startswith("binance"):
            params["options"] = {"defaultType": "swap"}
        self.ex = getattr(ccxt, exchange_id)(params)

    def ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        return self._frame(self.ex.fetch_ohlcv(symbol, timeframe, limit=limit))

    def history(self, symbol: str, timeframe: str, candles: int) -> pd.DataFrame:
        """Paginated fetch of the last `candles` candles (exchanges cap each call)."""
        step = self.ex.parse_timeframe(timeframe) * 1000
        since = int(time.time() * 1000) - candles * step
        rows = []
        while len(rows) < candles:
            batch = self.ex.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
            if not batch:
                break
            rows += batch
            nxt = batch[-1][0] + step
            if nxt <= since:
                break
            since = nxt
            if batch[-1][0] >= time.time() * 1000 - step:
                break
        df = self._frame(rows).loc[lambda d: ~d.index.duplicated()]
        return df.tail(candles)

    @staticmethod
    def _frame(rows) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def price(self, symbol: str) -> float:
        return float(self.ex.fetch_ticker(symbol)["last"])


class PaperBroker:
    """Simulated account on real prices. Models fees, slippage, leverage,
    shorts and liquidation. No real orders are ever sent."""

    name = "paper"
    margin_out_of_wallet = False   # wallet still includes the margin in use

    def __init__(self, market, quote: str, start_balance: float, fee_rate: float = 0.0005,
                 slippage_pct: float = 0.02):
        self.market, self.quote = market, quote
        self.fee_rate, self.slip = fee_rate, slippage_pct / 100
        self.wallet_balance = start_balance
        self.pos = None

    # -- persistence (so a restart doesn't reset the paper account) --
    def get_state(self) -> dict:
        return {"wallet": self.wallet_balance, "pos": self.pos}

    def set_state(self, st: dict):
        self.wallet_balance, self.pos = st["wallet"], st.get("pos")

    def wallet(self) -> float:
        return self.wallet_balance

    def position_qty(self, symbol: str):
        return self.pos["qty"] if self.pos else 0.0

    def open(self, symbol, side, margin, leverage, sl_price=None) -> Fill:
        if self.pos:
            raise ValueError("A position is already open")
        notional = margin * leverage
        fee = notional * self.fee_rate
        if margin + fee > self.wallet_balance:
            raise ValueError(f"Insufficient {self.quote}: have {self.wallet_balance:.2f}")
        d = direction(side)
        price = self.market.price(symbol) * (1 + d * self.slip)
        qty = notional / price
        self.wallet_balance -= fee
        self.pos = dict(side=side, qty=qty, entry=price, margin=margin, leverage=leverage)
        return Fill(side, symbol, qty, price, fee, time.time())

    def close(self, symbol, qty=None, price_hint=None) -> Fill:
        if not self.pos:
            raise ValueError("No open position")
        p, d = self.pos, direction(self.pos["side"])
        base = price_hint if price_hint else self.market.price(symbol)
        price = base * (1 - d * self.slip)
        fee = p["qty"] * price * self.fee_rate
        self.wallet_balance += d * (price - p["entry"]) * p["qty"] - fee
        self.pos = None
        return Fill(p["side"], symbol, p["qty"], price, fee, time.time())

    def liquidate(self, symbol) -> None:
        if self.pos:
            self.wallet_balance -= self.pos["margin"]
            self.pos = None


class LiveBroker:
    """Real orders through ccxt. USE WITH CARE.

    Futures: isolated margin, leverage set per symbol, and an exchange-side
    reduce-only stop-loss is placed right after entry. If that stop cannot be
    placed the position is closed again immediately."""

    name = "live"
    margin_out_of_wallet = False

    def __init__(self, market: Market, quote: str, futures: bool):
        self.market, self.quote, self.futures = market, quote, futures
        self.margin_out_of_wallet = not futures   # spot: cost leaves the free balance

    def wallet(self) -> float:
        bal = self.market.ex.fetch_balance()
        key = "total" if self.futures else "free"
        return float(bal.get(key, {}).get(self.quote, 0) or 0)

    def position_qty(self, symbol: str):
        """Open size on the exchange (futures only; None when unknown/spot)."""
        if not self.futures:
            return None
        for p in self.market.ex.fetch_positions([symbol]):
            amt = abs(float(p.get("contracts") or 0) * float(p.get("contractSize") or 1))
            if amt > 0:
                return amt
        return 0.0

    def _fill(self, order, side, symbol, qty, price) -> Fill:
        qty = float(order.get("filled") or qty)
        price = float(order.get("average") or order.get("price") or price)
        fee = 0.0
        base = symbol.split("/")[0]
        f = order.get("fee") or {}
        if f.get("cost"):
            if f.get("currency") == self.quote:
                fee = float(f["cost"])
            elif f.get("currency") == base:
                fee = float(f["cost"]) * price
                if not self.futures and side == "long":
                    qty -= float(f["cost"])
        return Fill(side, symbol, qty, price, fee, time.time())

    def open(self, symbol, side, margin, leverage, sl_price=None) -> Fill:
        ex = self.market.ex
        if self.futures:
            try:
                ex.set_margin_mode("isolated", symbol)
            except Exception:
                pass  # already isolated / not changeable with an open position
            ex.set_leverage(int(leverage), symbol)
        price = self.market.price(symbol)
        qty = float(ex.amount_to_precision(symbol, margin * leverage / price))
        order = ex.create_order(symbol, "market", "buy" if side == "long" else "sell", qty)
        fill = self._fill(order, side, symbol, qty, price)
        if self.futures and sl_price:
            try:
                ex.create_order(symbol, "market", "sell" if side == "long" else "buy", fill.qty, None,
                                {"stopLossPrice": float(ex.price_to_precision(symbol, sl_price)),
                                 "reduceOnly": True})
            except Exception as e:
                self.close(symbol, fill.qty)
                raise RuntimeError(f"Protective stop could not be placed ({e}); position closed again.")
        return fill

    def close(self, symbol, qty=None, price_hint=None):
        ex = self.market.ex
        if self.futures:
            size = self.position_qty(symbol)
            if not size:
                return None          # already closed on the exchange (stop-loss / liquidation)
            qty = size
            side_str = self._open_side(symbol)
        else:
            side_str = "long"
        qty = float(ex.amount_to_precision(symbol, qty))
        price = self.market.price(symbol)
        params = {"reduceOnly": True} if self.futures else {}
        order = ex.create_order(symbol, "market", "sell" if side_str == "long" else "buy", qty, None, params)
        if self.futures:
            try:
                ex.cancel_all_orders(symbol)
            except Exception:
                pass
        return self._fill(order, side_str, symbol, qty, price)

    def _open_side(self, symbol) -> str:
        for p in self.market.ex.fetch_positions([symbol]):
            if float(p.get("contracts") or 0) > 0:
                return p.get("side") or "long"
        return "long"
