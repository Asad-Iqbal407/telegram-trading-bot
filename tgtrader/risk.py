"""Position sizing, stop placement and liquidation math shared by the live
engine and the backtester so both behave the same."""

MMR = 0.005  # maintenance margin rate used to estimate liquidation


def direction(side: str) -> int:
    return 1 if side == "long" else -1


def liq_price(side: str, entry: float, leverage: float, mmr: float = MMR) -> float:
    """Approximate isolated-margin liquidation price (1x long can't be liquidated)."""
    if side == "long":
        return max(0.0, entry * (1 - 1 / leverage + mmr))
    return entry * (1 + 1 / leverage - mmr)


def stop_fractions(price: float, atr_value, cfg) -> tuple:
    """(stop_loss_fraction, take_profit_fraction) of price.
    ATR mode adapts to volatility; STOP_LOSS_PCT / TAKE_PROFIT_PCT are the caps
    (and the fixed values when ATR multipliers are 0)."""
    sl_cap, tp_cap = cfg.stop_loss_pct / 100, cfg.take_profit_pct / 100
    if cfg.sl_atr_mult > 0 and atr_value and atr_value == atr_value:
        sl = min(max(cfg.sl_atr_mult * atr_value / price, cfg.min_stop_pct / 100), sl_cap)
    else:
        sl = sl_cap
    if cfg.tp_atr_mult > 0 and atr_value and atr_value == atr_value:
        tp = min(max(cfg.tp_atr_mult * atr_value / price, sl), tp_cap)
    else:
        tp = tp_cap
    return sl, tp


def entry_margin(equity: float, sl_frac: float, cfg) -> float:
    """Margin (collateral) for a new position. With RISK_PER_TRADE_PCT > 0 the
    size is chosen so that hitting the stop loses about that % of equity."""
    if cfg.risk_per_trade_pct > 0:
        margin = equity * cfg.risk_per_trade_pct / 100 / (cfg.leverage * sl_frac)
    else:
        margin = cfg.trade_size_quote
    return min(margin, equity * cfg.max_margin_pct / 100)


def stop_prices(side: str, entry: float, sl_frac: float, tp_frac: float) -> tuple:
    d = direction(side)
    return entry * (1 - d * sl_frac), entry * (1 + d * tp_frac)
