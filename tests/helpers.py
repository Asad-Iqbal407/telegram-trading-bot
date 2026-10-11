import numpy as np
import pandas as pd


def make_df(closes, spread=0.001, volume=1.0, freq="5min"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq=freq, tz="UTC")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"ts": idx.astype("int64") // 10**6, "open": opens,
                         "high": np.maximum(opens, closes) * (1 + spread),
                         "low": np.minimum(opens, closes) * (1 - spread),
                         "close": closes, "volume": volume}, index=idx)


def synthetic(n=3000, seed=1):
    """Random walk with alternating trend/range regimes and volume spikes."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0004, 0.0, 0.0005], size=n // 200 + 1), 200)[:n]
    ret = drift + rng.normal(0, 0.0015, n)
    closes = 30000 * np.exp(np.cumsum(ret))
    df = make_df(closes, spread=0.0008)
    df["volume"] = rng.lognormal(0, 0.5, n)
    return df


class FakeMarket:
    def __init__(self, closes, spread=0.001):
        self.df, self.p, self.spread = make_df(closes, spread), float(closes[-1]), spread

    def set(self, closes):
        self.df, self.p = make_df(closes, self.spread), float(closes[-1])

    def ohlcv(self, *a, **k):
        return self.df

    def price(self, symbol):
        return self.p
