import pandas as pd

from .indicators import atr
from .risk import direction, entry_margin, liq_price, stop_fractions, stop_prices


def backtest(df: pd.DataFrame, strategy, cfg, start: float = 1000.0) -> dict:
    """Candle-by-candle simulation using the same sizing, stops, fees, slippage
    and liquidation rules as the live engine.

    Signals come from closed candle i and are executed at the OPEN of candle
    i+1; then stops/targets/liquidation are checked against that candle's
    high/low (worst case first: liquidation, stop, target). Not modelled:
    funding fees, partial fills, the daily loss limit, exchange outages."""
    sigs = strategy.signals(df)
    atr_s = atr(df, 14)
    fee, slip, lev = cfg.fee_rate, cfg.slippage_pct / 100, cfg.leverage
    wallet, pos = start, None
    trades, curve, liquidations = [], [], 0

    def close(price, reason):
        nonlocal wallet, pos
        d = direction(pos["side"])
        px = price * (1 - d * slip) if reason != "tp" else price
        pnl = d * (px - pos["entry"]) * pos["qty"] - pos["fee"] - pos["qty"] * px * fee
        wallet += d * (px - pos["entry"]) * pos["qty"] - pos["qty"] * px * fee
        trades.append(pnl)
        pos = None

    n = len(df)
    for i in range(strategy.min_candles, n - 1):
        nxt, sig = df.iloc[i + 1], sigs.iloc[i]
        if sig == "short" and not cfg.allow_short:
            sig = "exit_long"
        if pos and (sig in ("long", "short") and sig != pos["side"] or sig == f"exit_{pos['side']}"):
            close(nxt["open"], "signal")
        if not pos and sig in ("long", "short") and wallet > 1:
            side, a = sig, atr_s.iloc[i]
            entry = nxt["open"] * (1 + direction(side) * slip)
            sl_f, tp_f = stop_fractions(entry, a, cfg)
            margin = entry_margin(wallet, sl_f, cfg)
            notional = margin * lev
            if margin >= 1 and margin + notional * fee <= wallet:
                sl, tp = stop_prices(side, entry, sl_f, tp_f)
                wallet -= notional * fee
                pos = dict(side=side, qty=notional / entry, entry=entry, margin=margin,
                           fee=notional * fee, sl=sl, tp=tp,
                           liq=liq_price(side, entry, lev) if lev > 1 else None)
        if pos:
            long = pos["side"] == "long"
            worst, best = (nxt["low"], nxt["high"]) if long else (nxt["high"], nxt["low"])
            hit = (lambda a, b: a <= b) if long else (lambda a, b: a >= b)
            if pos["liq"] and hit(worst, pos["liq"]):
                wallet -= pos["margin"]
                trades.append(-pos["margin"] - pos["fee"])
                liquidations += 1
                pos = None
            elif hit(worst, pos["sl"]):
                close(pos["sl"], "sl")
            elif hit(pos["tp"], best):
                close(pos["tp"], "tp")
        mark = wallet
        if pos:
            mark += direction(pos["side"]) * (nxt["close"] - pos["entry"]) * pos["qty"]
        curve.append(mark)

    final = curve[-1] if curve else start
    peak, max_dd = start, 0.0
    for e in curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)
    wins = [t for t in trades if t > 0]
    losses = [-t for t in trades if t <= 0]
    first = df["close"].iloc[strategy.min_candles]
    return dict(trades=len(trades), win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
                return_pct=(final / start - 1) * 100, buy_hold_pct=(df["close"].iloc[-1] / first - 1) * 100,
                max_drawdown_pct=max_dd * 100, liquidations=liquidations,
                profit_factor=(sum(wins) / sum(losses)) if losses and sum(losses) > 0 else float("inf") if wins else 0.0,
                candles=n - strategy.min_candles)
