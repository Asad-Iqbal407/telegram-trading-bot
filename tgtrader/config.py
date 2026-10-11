import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .risk import MMR

LIVE_PHRASE = "I_ACCEPT_THE_RISK"
LEVERAGE_PHRASE = "I_UNDERSTAND_LIQUIDATION"


@dataclass
class Config:
    telegram_token: str = ""
    allowed_chat_ids: set = field(default_factory=set)
    mode: str = "paper"
    market_type: str = "futures"          # "futures" (leverage, long+short) or "spot"
    exchange: str = "binanceusdm"
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "5m"
    strategy: str = "ensemble"
    # sma_rsi parameters
    fast_sma: int = 10
    slow_sma: int = 30
    rsi_period: int = 14
    rsi_max_buy: float = 70.0
    # leverage / risk
    leverage: float = 3.0
    max_leverage: float = 10.0
    trade_size_quote: float = 50.0        # margin per trade when RISK_PER_TRADE_PCT = 0
    risk_per_trade_pct: float = 1.0       # % of equity lost if the stop is hit (0 = fixed size)
    max_margin_pct: float = 50.0          # max % of equity used as margin in one trade
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    min_stop_pct: float = 0.2
    stop_loss_pct: float = 1.5            # cap (or fixed value when ATR multiplier is 0)
    take_profit_pct: float = 3.0
    daily_loss_limit_pct: float = 5.0
    max_daily_trades: int = 20
    fee_rate: float = 0.0005
    slippage_pct: float = 0.02
    paper_start_balance: float = 1000.0
    poll_seconds: int = 30
    state_file: str = "state.json"

    @property
    def futures(self) -> bool:
        return self.market_type == "futures"

    @property
    def allow_short(self) -> bool:
        return self.futures

    @property
    def quote(self) -> str:
        return self.symbol.split("/")[1].split(":")[0]

    @property
    def base(self) -> str:
        return self.symbol.split("/")[0]

    def validate_leverage(self, leverage: float = None) -> None:
        lev = self.leverage if leverage is None else leverage
        if lev < 1 or lev > self.max_leverage:
            raise ValueError(f"Leverage must be between 1 and MAX_LEVERAGE ({self.max_leverage:g})")
        if not self.futures and lev != 1:
            raise ValueError("Leverage needs MARKET_TYPE=futures")
        liq_dist = (1 / lev - MMR) * 100
        if lev > 1 and self.stop_loss_pct * 1.25 >= liq_dist:
            raise ValueError(f"At {lev:g}x you are liquidated after a ~{liq_dist:.1f}% move; "
                             f"STOP_LOSS_PCT ({self.stop_loss_pct:g}%) is too close to that. "
                             "Lower the leverage or the stop.")

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
        if self.market_type not in ("futures", "spot"):
            raise ValueError("MARKET_TYPE must be 'futures' or 'spot'")
        if not 1 <= self.max_leverage <= 25:
            raise ValueError("MAX_LEVERAGE must be between 1 and 25")
        self.validate_leverage()
        if self.mode == "live":
            if os.environ.get("LIVE_CONFIRM") != LIVE_PHRASE:
                raise ValueError(f"Live mode requires LIVE_CONFIRM={LIVE_PHRASE}")
            if self.leverage > 1 and os.environ.get("LIVE_LEVERAGE_CONFIRM") != LEVERAGE_PHRASE:
                raise ValueError(f"Live leveraged trading requires LIVE_LEVERAGE_CONFIRM={LEVERAGE_PHRASE}")
            if not (self.api_key and self.api_secret):
                raise ValueError("Live mode requires API_KEY and API_SECRET")
        if self.fast_sma >= self.slow_sma:
            raise ValueError("FAST_SMA must be smaller than SLOW_SMA")
        if self.trade_size_quote <= 0 or not 0 <= self.risk_per_trade_pct <= 5:
            raise ValueError("TRADE_SIZE_QUOTE must be > 0 and RISK_PER_TRADE_PCT between 0 and 5")
        if not 0 < self.max_margin_pct <= 100:
            raise ValueError("MAX_MARGIN_PCT must be in (0, 100]")


def normalize_market(exchange: str, symbol: str, market_type: str) -> tuple:
    """Make EXCHANGE/SYMBOL consistent with MARKET_TYPE so old .env files keep working."""
    if market_type == "futures":
        if exchange == "binance":
            exchange = "binanceusdm"
        if ":" not in symbol and "/" in symbol:
            symbol = f"{symbol}:{symbol.split('/')[1]}"
    else:
        if exchange == "binanceusdm":
            exchange = "binance"
        symbol = symbol.split(":")[0]
    return exchange, symbol


def load_config() -> Config:
    load_dotenv()
    e = os.environ.get
    ids = {int(x) for x in e("ALLOWED_CHAT_IDS", "").replace(" ", "").split(",") if x}
    market_type = e("MARKET_TYPE", "futures").lower()
    exchange, symbol = normalize_market(e("EXCHANGE", "binanceusdm"), e("SYMBOL", "BTC/USDT"), market_type)
    d = Config()
    f = lambda k, default: float(e(k, default))
    cfg = Config(
        telegram_token=e("TELEGRAM_BOT_TOKEN", ""), allowed_chat_ids=ids,
        mode=e("TRADING_MODE", "paper").lower(), market_type=market_type,
        exchange=exchange, api_key=e("API_KEY", ""), api_secret=e("API_SECRET", ""),
        symbol=symbol, timeframe=e("TIMEFRAME", d.timeframe), strategy=e("STRATEGY", d.strategy),
        fast_sma=int(e("FAST_SMA", d.fast_sma)), slow_sma=int(e("SLOW_SMA", d.slow_sma)),
        rsi_period=int(e("RSI_PERIOD", d.rsi_period)), rsi_max_buy=f("RSI_MAX_BUY", d.rsi_max_buy),
        leverage=f("LEVERAGE", d.leverage) if market_type == "futures" else 1.0,
        max_leverage=f("MAX_LEVERAGE", d.max_leverage),
        trade_size_quote=f("TRADE_SIZE_QUOTE", d.trade_size_quote),
        risk_per_trade_pct=f("RISK_PER_TRADE_PCT", d.risk_per_trade_pct),
        max_margin_pct=f("MAX_MARGIN_PCT", d.max_margin_pct),
        sl_atr_mult=f("SL_ATR_MULT", d.sl_atr_mult), tp_atr_mult=f("TP_ATR_MULT", d.tp_atr_mult),
        min_stop_pct=f("MIN_STOP_PCT", d.min_stop_pct),
        stop_loss_pct=f("STOP_LOSS_PCT", d.stop_loss_pct), take_profit_pct=f("TAKE_PROFIT_PCT", d.take_profit_pct),
        daily_loss_limit_pct=f("DAILY_LOSS_LIMIT_PCT", d.daily_loss_limit_pct),
        max_daily_trades=int(e("MAX_DAILY_TRADES", d.max_daily_trades)),
        fee_rate=f("FEE_RATE", d.fee_rate), slippage_pct=f("SLIPPAGE_PCT", d.slippage_pct),
        paper_start_balance=f("PAPER_START_BALANCE", d.paper_start_balance),
        poll_seconds=int(e("POLL_SECONDS", d.poll_seconds)),
    )
    cfg.validate()
    return cfg
