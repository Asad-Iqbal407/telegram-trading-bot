"""Trading strategies. Every strategy turns candles into one signal per candle:

    'long' / 'short'          open (or flip to) that side
    'exit_long' / 'exit_short' close that side if open
    ''                         do nothing

Signals are computed on CLOSED candles only and are causal (no look-ahead).
"""
import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, macd, rsi, sma, supertrend


def from_direction(d: pd.Series) -> pd.Series:
    """Turn a +1/-1/0 'state' series into entry/exit events on changes."""
    prev = d.shift(1).fillna(0)
    out = pd.Series("", index=d.index)
    out[(d == 0) & (prev == 1)] = "exit_long"
    out[(d == 0) & (prev == -1)] = "exit_short"
    out[(d == 1) & (prev != 1)] = "long"
    out[(d == -1) & (prev != -1)] = "short"
    return out


def crossed_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def crossed_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


class Strategy:
    name = ""
    description = ""
    min_candles = 50

    def direction(self, df: pd.DataFrame) -> pd.Series:
        """+1 / -1 / 0 market 'state' (used by the ensemble)."""
        return pd.Series(0.0, index=df.index)

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return from_direction(self.direction(df))

    def signal(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_candles:
            return ""
        return self.signals(df).iloc[-1]


class SmaRsiStrategy(Strategy):
    name = "sma_rsi"
    description = "SMA crossover, skipping entries into overbought/oversold RSI"

    def __init__(self, fast: int = 10, slow: int = 30, rsi_period: int = 14, rsi_max_buy: float = 70.0):
        self.fast, self.slow = fast, slow
        self.rsi_period, self.rsi_max_buy = rsi_period, rsi_max_buy

    @property
    def min_candles(self) -> int:
        return max(self.slow, self.rsi_period) + 2

    def direction(self, df):
        f, s = sma(df["close"], self.fast), sma(df["close"], self.slow)
        return pd.Series(np.sign(f - s), index=df.index).fillna(0.0)

    def signals(self, df):
        close = df["close"]
        f, s, r = sma(close, self.fast), sma(close, self.slow), rsi(close, self.rsi_period)
        out = pd.Series("", index=df.index)
        out[crossed_down(f, s)] = "short"
        out[crossed_up(f, s) & (r < self.rsi_max_buy)] = "long"
        out[crossed_down(f, s) & (r < 100 - self.rsi_max_buy)] = "exit_long"  # already oversold
        return out


class EmaTrendStrategy(Strategy):
    name = "ema_trend"
    description = "EMA 9/21 cross, only in the direction of the EMA 200 trend"
    min_candles = 230

    def __init__(self, fast: int = 9, slow: int = 21, trend: int = 200):
        self.fast, self.slow, self.trend = fast, slow, trend

    def direction(self, df):
        c = df["close"]
        f, s, t = ema(c, self.fast), ema(c, self.slow), ema(c, self.trend)
        d = pd.Series(0.0, index=df.index)
        d[(f > s) & (c > t)] = 1.0
        d[(f < s) & (c < t)] = -1.0
        return d

    def signals(self, df):
        c = df["close"]
        f, s, t = ema(c, self.fast), ema(c, self.slow), ema(c, self.trend)
        out = pd.Series("", index=df.index)
        out[crossed_up(f, s)] = "exit_short"
        out[crossed_down(f, s)] = "exit_long"
        out[crossed_up(f, s) & (c > t)] = "long"
        out[crossed_down(f, s) & (c < t)] = "short"
        return out


class SupertrendStrategy(Strategy):
    name = "supertrend"
    description = "ATR-based trend follower (always long or short after warm-up)"

    def __init__(self, period: int = 10, mult: float = 3.0):
        self.period, self.mult = period, mult

    @property
    def min_candles(self):
        return self.period * 3

    def direction(self, df):
        return supertrend(df, self.period, self.mult)


class MacdStrategy(Strategy):
    name = "macd_trend"
    description = "MACD histogram zero-cross, filtered by EMA 200 trend"
    min_candles = 230

    def direction(self, df):
        _, _, hist = macd(df["close"])
        return pd.Series(np.sign(hist), index=df.index).fillna(0.0)

    def signals(self, df):
        c = df["close"]
        _, _, hist = macd(c)
        zero = pd.Series(0.0, index=df.index)
        t = ema(c, 200)
        out = pd.Series("", index=df.index)
        out[crossed_up(hist, zero)] = "exit_short"
        out[crossed_down(hist, zero)] = "exit_long"
        out[crossed_up(hist, zero) & (c > t)] = "long"
        out[crossed_down(hist, zero) & (c < t)] = "short"
        return out


class BollingerReversionStrategy(Strategy):
    name = "bb_reversion"
    description = "Mean reversion: fade Bollinger-band extremes with RSI, only when ADX says ranging"
    min_candles = 60

    def __init__(self, period: int = 20, k: float = 2.0, rsi_low: float = 30.0,
                 rsi_high: float = 70.0, adx_max: float = 25.0):
        self.period, self.k = period, k
        self.rsi_low, self.rsi_high, self.adx_max = rsi_low, rsi_high, adx_max

    def signals(self, df):
        c = df["close"]
        mid, up, lo = bollinger(c, self.period, self.k)
        r, a = rsi(c, 14), adx(df, 14)
        ranging = a < self.adx_max
        out = pd.Series("", index=df.index)
        out[c >= mid] = "exit_long"
        out[c <= mid] = "exit_short"
        out[(c < lo) & (r < self.rsi_low) & ranging] = "long"
        out[(c > up) & (r > self.rsi_high) & ranging] = "short"
        return out


class DonchianBreakoutStrategy(Strategy):
    name = "breakout"
    description = "Donchian channel breakout confirmed by above-average volume"
    min_candles = 60

    def __init__(self, period: int = 20, vol_mult: float = 1.2):
        self.period, self.vol_mult = period, vol_mult

    def direction(self, df):
        hi = df["high"].rolling(self.period).max().shift(1)
        lo = df["low"].rolling(self.period).min().shift(1)
        vol_ok = df["volume"] > self.vol_mult * df["volume"].rolling(self.period).mean().shift(1)
        d = pd.Series(np.nan, index=df.index)
        d[(df["close"] > hi) & vol_ok] = 1.0
        d[(df["close"] < lo) & vol_ok] = -1.0
        return d.ffill().fillna(0.0)

    def signals(self, df):
        d = self.direction(df)
        out = pd.Series("", index=df.index)
        out[(d == 1) & (d.shift(1) != 1)] = "long"
        out[(d == -1) & (d.shift(1) != -1)] = "short"
        return out


class EnsembleStrategy(Strategy):
    name = "ensemble"
    description = "Trade only when 2 of 3 trend models agree (EMA trend, Supertrend, MACD) and ADX > 20"
    min_candles = 230

    def __init__(self, need: int = 2, adx_min: float = 20.0):
        self.parts = [EmaTrendStrategy(), SupertrendStrategy(), MacdStrategy()]
        self.need, self.adx_min = need, adx_min

    def direction(self, df):
        votes = sum(p.direction(df) for p in self.parts)
        d = pd.Series(0.0, index=df.index)
        strong = adx(df, 14) > self.adx_min
        d[(votes >= self.need) & strong] = 1.0
        d[(votes <= -self.need) & strong] = -1.0
        return d


STRATEGIES = {c.name: c for c in (SmaRsiStrategy, EmaTrendStrategy, SupertrendStrategy, MacdStrategy,
                                  BollingerReversionStrategy, DonchianBreakoutStrategy, EnsembleStrategy)}


def build_strategy(name: str, cfg=None) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Options: {', '.join(STRATEGIES)}")
    if name == "sma_rsi" and cfg is not None:
        return SmaRsiStrategy(cfg.fast_sma, cfg.slow_sma, cfg.rsi_period, cfg.rsi_max_buy)
    return STRATEGIES[name]()
