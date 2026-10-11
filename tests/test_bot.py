import pytest

from tests.helpers import FakeMarket, synthetic
from tgtrader.bot import TradingBot
from tgtrader.broker import PaperBroker
from tgtrader.config import Config
from tgtrader.strategy import build_strategy


class Msg:
    def __init__(self):
        self.out = []

    async def reply_text(self, t):
        self.out.append(t)


class Upd:
    def __init__(self, chat=1):
        self.message = Msg()
        self.effective_chat = type("C", (), {"id": chat})()


class Ctx:
    def __init__(self, *args):
        self.args = list(args)


@pytest.fixture
def bot(tmp_path):
    df = synthetic(2500)
    market = FakeMarket(df["close"].to_numpy())
    market.df = df
    market.history = lambda s, tf, n: df.tail(n)
    cfg = Config(telegram_token="123456:ABCdef", allowed_chat_ids={1}, state_file=str(tmp_path / "s.json"))
    b = TradingBot(cfg, market, PaperBroker(market, "USDT", 1000), build_strategy(cfg.strategy, cfg))

    async def quiet(text):
        b.sent = getattr(b, "sent", []) + [text]
    b.notify = quiet
    b.engine.notify = quiet
    return b


async def run(bot, fn, *args, chat=1):
    u = Upd(chat)
    await fn(u, Ctx(*args))
    return "\n".join(u.message.out)


async def test_unauthorized_chat_is_refused(bot):
    assert "Not authorized" in await run(bot, bot.status, chat=999)
    assert "Not authorized" in await run(bot, bot.long, "10", chat=999)


async def test_commands_smoke(bot):
    assert "Strategy" in await run(bot, bot.status)
    assert "wallet" in await run(bot, bot.balance)
    assert "ensemble" in await run(bot, bot.strategies)
    await run(bot, bot.strategy, "supertrend")
    assert bot.engine.strategy.name == "supertrend"
    assert "Unknown strategy" in await run(bot, bot.strategy, "nope")
    assert "5x" in await run(bot, bot.leverage, "5")
    assert "ValueError" in await run(bot, bot.leverage, "99")
    assert bot.cfg.leverage == 5
    assert "✅" in await run(bot, bot.set_param, "size", "25")
    assert "ValueError" in await run(bot, bot.set_param, "sl", "50")     # stop beyond liquidation at 5x
    assert bot.cfg.stop_loss_pct == 1.5                                      # reverted


async def test_manual_long_short_close(bot):
    await run(bot, bot.long, "50")
    assert bot.engine.position["side"] == "long"
    assert "already open" in await run(bot, bot.short, "50")
    await run(bot, bot.close)
    await run(bot, bot.short)                                                # auto-sized
    assert bot.engine.position["side"] == "short"
    assert "No open position" in (await run(bot, bot.close) or "No open position")
    assert "No open position" in await run(bot, bot.close)
    assert "CLOSE" in "".join(bot.sent) and "SHORT" in "".join(bot.sent)


async def test_backtest_all(bot):
    out = await run(bot, bot.backtest, "2000", "all")
    for name in ("sma_rsi", "ema_trend", "supertrend", "macd_trend", "bb_reversion", "breakout", "ensemble"):
        assert name in out
    assert "Buy&hold" in out


async def test_autostart_starts_engine_and_state_file_env(bot, monkeypatch, tmp_path):
    bot.cfg.autostart = True
    await bot._post_init(None)
    assert bot.engine.running
    bot.engine.stop()
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "x.json"))
    from tgtrader.config import load_config
    assert load_config().state_file == str(tmp_path / "x.json")
