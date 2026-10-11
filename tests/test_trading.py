import json

import numpy as np
import pandas as pd
import pytest

from tests.helpers import FakeMarket, make_df, synthetic
from tgtrader.backtest import backtest
from tgtrader.broker import Fill, LiveBroker, PaperBroker
from tgtrader.config import Config, normalize_market
from tgtrader.engine import Engine
from tgtrader.indicators import adx, atr, supertrend
from tgtrader.risk import entry_margin, liq_price, stop_fractions
from tgtrader.strategy import STRATEGIES, Strategy, build_strategy

VALID = {"", "long", "short", "exit_long", "exit_short"}


# ---------------- strategies ----------------
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategy_signals_valid_and_present(name):
    df = synthetic()
    sig = build_strategy(name).signals(df)
    assert set(sig) <= VALID
    assert (sig == "long").sum() + (sig == "short").sum() > 0, "strategy never trades on 3000 candles"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategy_has_no_lookahead(name):
    """Signals on a truncated history must equal the same candles' signals on the full history."""
    df, strat = synthetic(1500, seed=7), build_strategy(name)
    full = strat.signals(df)
    for k in (700, 1000, 1400):
        assert strat.signals(df.iloc[:k]).iloc[-1] == full.iloc[k - 1], f"{name} looks ahead at {k}"


def test_indicators_basic():
    df = synthetic(500)
    assert (atr(df, 14).dropna() > 0).all()
    assert set(supertrend(df)) <= {-1.0, 0.0, 1.0}
    assert adx(df).dropna().between(0, 100).all()


# ---------------- risk / config ----------------
def test_liq_price():
    assert liq_price("long", 100, 10) == pytest.approx(100 * (1 - 0.1 + 0.005))
    assert liq_price("short", 100, 10) == pytest.approx(100 * (1 + 0.1 - 0.005))


def test_leverage_validation_blocks_stop_beyond_liquidation():
    cfg = Config(stop_loss_pct=1.5, max_leverage=25)
    cfg.validate_leverage(20)                       # liquidation ~4.5% away, stop 1.5% -> fine
    with pytest.raises(ValueError, match="MAX_LEVERAGE"):
        cfg.validate_leverage(30)
    wide = Config(stop_loss_pct=5, max_leverage=25)
    with pytest.raises(ValueError, match="liquidated"):
        wide.validate_leverage(25)                  # liquidation ~3.5% away, stop is 5%


def test_spot_rejects_leverage():
    with pytest.raises(ValueError):
        Config(market_type="spot").validate_leverage(2)


def test_normalize_market():
    assert normalize_market("binance", "BTC/USDT", "futures") == ("binanceusdm", "BTC/USDT:USDT")
    assert normalize_market("binanceusdm", "BTC/USDT:USDT", "spot") == ("binance", "BTC/USDT")


def test_stop_fractions_and_sizing():
    cfg = Config(sl_atr_mult=1.5, tp_atr_mult=3, min_stop_pct=0.2, stop_loss_pct=1.5, take_profit_pct=3,
                 leverage=5, risk_per_trade_pct=1, max_margin_pct=50)
    sl, tp = stop_fractions(30000, 60, cfg)      # ATR 0.2% -> SL 0.3%, TP 0.6%
    assert sl == pytest.approx(0.003) and tp == pytest.approx(0.006)
    sl2, _ = stop_fractions(30000, 3000, cfg)    # huge ATR -> capped at 1.5%
    assert sl2 == pytest.approx(0.015)
    # losing the stop on the sized position costs ~1% of equity
    margin = entry_margin(1000, sl, cfg)
    assert margin * cfg.leverage * sl == pytest.approx(10, rel=1e-6) or margin == 500
    assert entry_margin(1000, 0.0001, cfg) == 500  # capped by MAX_MARGIN_PCT


# ---------------- paper broker ----------------
def test_paper_long_pnl():
    m = FakeMarket([100.0] * 5)
    b = PaperBroker(m, "USDT", 1000, fee_rate=0.0, slippage_pct=0.0)
    f = b.open("X", "long", 100, 5)
    assert f.qty == pytest.approx(5.0)
    m.p = 110.0
    b.close("X")
    assert b.wallet() == pytest.approx(1050)       # +10% * 5x on 100 margin


def test_paper_short_profit_direction():
    m = FakeMarket([100.0] * 5)
    b = PaperBroker(m, "USDT", 1000, fee_rate=0.0, slippage_pct=0.0)
    b.open("X", "short", 100, 5)
    m.p = 90.0
    b.close("X")
    assert b.wallet() == pytest.approx(1050)


def test_paper_fees_slippage_and_limits():
    m = FakeMarket([100.0] * 5)
    b = PaperBroker(m, "USDT", 1000, fee_rate=0.001, slippage_pct=0.1)
    b.open("X", "long", 100, 2)
    b.close("X")
    assert b.wallet() < 1000 - 0.4 and b.pos is None
    with pytest.raises(ValueError):
        b.open("X", "long", 5000, 2)
    with pytest.raises(ValueError):
        b.close("X")


def test_paper_liquidate_loses_margin():
    m = FakeMarket([100.0] * 5)
    b = PaperBroker(m, "USDT", 1000, fee_rate=0.0, slippage_pct=0.0)
    b.open("X", "long", 200, 10)
    b.liquidate("X")
    assert b.wallet() == pytest.approx(800)


# ---------------- engine ----------------
class Stub(Strategy):
    name, min_candles = "stub", 5

    def __init__(self, sig=""):
        self.sig = sig

    def signals(self, df):
        out = pd.Series("", index=df.index)
        out.iloc[-1] = self.sig
        return out


def flat(price=100.0, n=30):
    return [price] * n


async def make_engine(tmp_path, closes=None, strat=None, **cfg_kw):
    kw = dict(symbol="BTC/USDT:USDT", leverage=5, trade_size_quote=100, risk_per_trade_pct=0,
              sl_atr_mult=0, tp_atr_mult=0, stop_loss_pct=2, take_profit_pct=4, fee_rate=0.0,
              slippage_pct=0.0, state_file=str(tmp_path / "s.json"))
    cfg = Config(**{**kw, **cfg_kw})
    market = FakeMarket(closes or flat())
    broker = PaperBroker(market, "USDT", 1000, cfg.fee_rate, cfg.slippage_pct)
    msgs = []

    async def notify(t):
        msgs.append(t)
    return Engine(cfg, market, broker, strat or Stub(), notify), market, msgs


async def test_open_long_sets_stops_and_liquidation(tmp_path):
    eng, _, _ = await make_engine(tmp_path)
    await eng.open_position("long", 100)
    p = eng.position
    assert p["sl"] == pytest.approx(98) and p["tp"] == pytest.approx(104)
    assert p["liq"] == pytest.approx(100 * (1 - 0.2 + 0.005))


async def test_short_stop_loss_triggered_by_wick(tmp_path):
    eng, market, msgs = await make_engine(tmp_path)
    await eng.open_position("short", 100)
    eng.position["ts"] = 0                           # opened before every candle shown
    closes = flat() + [100.0]
    market.set(closes)
    market.df.loc[market.df.index[-1], "high"] = 103  # wick above the 102 stop, close back at 100
    await eng.tick()
    assert eng.position is None and any("stop-loss" in m for m in msgs)
    assert eng.trades[-1]["pnl"] == pytest.approx(-100 * 5 * 0.02)   # lost the stop distance exactly


async def test_long_take_profit_by_wick(tmp_path):
    eng, market, msgs = await make_engine(tmp_path)
    await eng.open_position("long", 100)
    eng.position["ts"] = 0
    market.set(flat())
    market.df.loc[market.df.index[-1], "high"] = 105
    await eng.tick()
    assert eng.position is None and eng.trades[-1]["pnl"] == pytest.approx(100 * 5 * 0.04)


async def test_paper_liquidation_when_stop_is_skipped(tmp_path):
    eng, market, msgs = await make_engine(tmp_path, stop_loss_pct=50, leverage=10, max_leverage=25)
    await eng.open_position("long", 100)
    eng.position["ts"] = 0
    market.set(flat())
    market.df.loc[market.df.index[-1], "low"] = 85   # below the ~90.5 liquidation price
    await eng.tick()
    assert eng.position is None and "LIQUIDATED" in msgs[-1]
    assert eng.broker.wallet() == pytest.approx(900)


async def test_candle_of_entry_ignores_pre_entry_wick(tmp_path):
    eng, market, _ = await make_engine(tmp_path)
    await eng.open_position("long", 100)             # ts = now, newer than every candle
    market.set(flat())
    market.df.loc[market.df.index[-1], "low"] = 90   # happened before we entered
    await eng.tick()
    assert eng.position is not None


async def test_spot_refuses_short_and_signal_maps_to_exit(tmp_path):
    eng, _, _ = await make_engine(tmp_path, strat=Stub("short"), market_type="spot", leverage=1,
                                  symbol="BTC/USDT")
    assert "futures" in await eng.open_position("short", 50)
    await eng.open_position("long", 50)
    await eng.tick()                                  # 'short' signal on spot closes the long
    assert eng.position is None


async def test_signal_flip_long_to_short(tmp_path):
    eng, market, _ = await make_engine(tmp_path, strat=Stub("short"))
    await eng.open_position("long", 100)
    await eng.tick()
    assert eng.position["side"] == "short"


async def test_risk_sized_entry(tmp_path):
    eng, _, _ = await make_engine(tmp_path, strat=Stub("long"), risk_per_trade_pct=1, leverage=5)
    await eng.tick()
    p = eng.position
    assert p["margin"] * p["leverage"] * 0.02 == pytest.approx(10)   # stop = 2% -> loses 1% of 1000


async def test_margin_cap_and_daily_trade_limit(tmp_path):
    eng, _, _ = await make_engine(tmp_path, max_daily_trades=1)
    assert "Refused" in await eng.open_position("long", 900)
    await eng.open_position("long", 100)
    await eng.close_position("x")
    assert "Daily trade limit" in await eng.open_position("long", 100)


async def test_daily_loss_limit_halts(tmp_path):
    eng, market, _ = await make_engine(tmp_path, stop_loss_pct=50, leverage=2, daily_loss_limit_pct=5,
                                       max_leverage=25)
    await eng.tick()                                  # sets day start equity
    await eng.open_position("long", 500)
    market.set(flat()[:-1] + [94.0])                  # -6% * 2x on 500 margin = -60 (6%)
    market.df.loc[market.df.index[-1], "low"] = 94.0
    await eng.tick()
    assert eng.halted_reason and eng.position is None
    assert "halted" in (await eng.open_position("long", 10)).lower()


async def test_state_persists_including_paper_wallet(tmp_path):
    eng, _, _ = await make_engine(tmp_path)
    await eng.open_position("short", 100)
    eng2, _, _ = await make_engine(tmp_path)
    assert eng2.position["side"] == "short" and eng2.broker.pos is not None
    assert eng2.broker.wallet() == pytest.approx(1000)
    await eng2.close_position("test")                 # must not raise "no position" in the broker


async def test_state_ignored_for_other_symbol(tmp_path):
    eng, _, _ = await make_engine(tmp_path)
    await eng.open_position("long", 100)
    eng2, _, _ = await make_engine(tmp_path, symbol="ETH/USDT:USDT")
    assert eng2.position is None


# ---------------- backtest ----------------
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_backtest_runs_for_every_strategy(name):
    cfg = Config(leverage=3)
    r = backtest(synthetic(2500), build_strategy(name), cfg)
    assert r["trades"] >= 0 and 0 <= r["max_drawdown_pct"] <= 100 and r["candles"] > 0


def test_backtest_high_leverage_liquidates_on_crash():
    down = 30000 * 0.9995 ** np.arange(80)
    closes = list(down) + list(down[-1] * 1.001 ** np.arange(220))               # dip, then a trend up -> supertrend flips long
    closes += [closes[-1] * 0.7] + [closes[-1] * 0.7] * 50       # -30% in one candle
    df = make_df(closes, spread=0.0005)
    kw = dict(stop_loss_pct=50, take_profit_pct=50, sl_atr_mult=0, tp_atr_mult=0, max_leverage=25,
              risk_per_trade_pct=0, trade_size_quote=100)
    low = backtest(df, build_strategy("supertrend"), Config(leverage=2, **kw))
    high = backtest(df, build_strategy("supertrend"), Config(leverage=25, **kw))
    assert high["liquidations"] >= 1 and low["liquidations"] == 0
    assert high["return_pct"] < low["return_pct"]


def test_backtest_costs_reduce_returns():
    df = synthetic(2500, seed=5)
    s = build_strategy("ema_trend")
    free = backtest(df, s, Config(fee_rate=0, slippage_pct=0))
    paid = backtest(df, s, Config(fee_rate=0.001, slippage_pct=0.05))
    assert paid["return_pct"] <= free["return_pct"]


# ---------------- live broker (fake ccxt exchange) ----------------
class FakeEx:
    def __init__(self, fail_stop=False, open_qty=0.0):
        self.orders, self.fail_stop, self.open_qty = [], fail_stop, open_qty
        self.leverage = self.margin_mode = None

    def set_margin_mode(self, mode, symbol):
        self.margin_mode = mode

    def set_leverage(self, lev, symbol):
        self.leverage = lev

    def amount_to_precision(self, s, a):
        return f"{a:.3f}"

    def price_to_precision(self, s, p):
        return f"{p:.1f}"

    def create_order(self, symbol, typ, side, amount, price=None, params=None):
        params = params or {}
        if "stopLossPrice" in params and self.fail_stop:
            raise RuntimeError("stop rejected")
        self.orders.append((side, amount, params))
        if "stopLossPrice" not in params:
            self.open_qty = amount if not params.get("reduceOnly") else 0.0
        return {"filled": amount, "average": 100.0, "fee": {"cost": 0.05, "currency": "USDT"}}

    def fetch_positions(self, symbols):
        side = "short" if self.orders and self.orders[0][0] == "sell" else "long"
        return [{"contracts": self.open_qty, "contractSize": 1, "side": side}]

    def cancel_all_orders(self, symbol):
        self.cancelled = True

    def fetch_balance(self):
        return {"total": {"USDT": 500.0}, "free": {"USDT": 400.0}}


class FakeLiveMarket:
    def __init__(self, ex):
        self.ex = ex

    def price(self, s):
        return 100.0


def test_live_open_sets_isolated_leverage_and_stop():
    ex = FakeEx()
    b = LiveBroker(FakeLiveMarket(ex), "USDT", futures=True)
    f = b.open("BTC/USDT:USDT", "long", 100, 5, sl_price=98.0)
    assert ex.margin_mode == "isolated" and ex.leverage == 5
    assert f.qty == pytest.approx(5.0) and f.fee == 0.05
    side, amt, params = ex.orders[1]
    assert side == "sell" and params == {"stopLossPrice": 98.0, "reduceOnly": True}


def test_live_closes_position_if_stop_cannot_be_placed():
    ex = FakeEx(fail_stop=True)
    b = LiveBroker(FakeLiveMarket(ex), "USDT", futures=True)
    with pytest.raises(RuntimeError, match="Protective stop"):
        b.open("BTC/USDT:USDT", "short", 100, 5, sl_price=102.0)
    assert ex.orders[-1][0] == "buy" and ex.orders[-1][2] == {"reduceOnly": True}   # flattened again


def test_live_close_returns_none_when_exchange_already_closed():
    ex = FakeEx(open_qty=0.0)
    b = LiveBroker(FakeLiveMarket(ex), "USDT", futures=True)
    assert b.close("BTC/USDT:USDT", 5.0) is None
    assert b.wallet() == 500.0
