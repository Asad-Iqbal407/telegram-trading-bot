import logging

from tgtrader.bot import TradingBot
from tgtrader.broker import LiveBroker, Market, PaperBroker
from tgtrader.config import load_config
from tgtrader.strategy import build_strategy


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs URLs that contain the bot token
    cfg = load_config()
    if not cfg.telegram_token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    if not cfg.allowed_chat_ids:
        logging.warning("ALLOWED_CHAT_IDS is empty: send /myid to the bot, add the id to .env, restart.")
    if cfg.mode == "live":
        market = Market(cfg.exchange, cfg.api_key, cfg.api_secret, cfg.futures)
        broker = LiveBroker(market, cfg.quote, cfg.futures)
    else:
        market = Market(cfg.exchange, futures=cfg.futures)  # public data only, no keys
        broker = PaperBroker(market, cfg.quote, cfg.paper_start_balance, cfg.fee_rate, cfg.slippage_pct)
    TradingBot(cfg, market, broker, build_strategy(cfg.strategy, cfg)).run()


if __name__ == "__main__":
    main()
