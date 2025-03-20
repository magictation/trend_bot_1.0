# Bitcoin Trend Correction Trading Bot

This project implements a trading bot that detects trend corrections in Bitcoin price and sends trading signals via Telegram. The bot uses a combination of technical indicators to identify high-probability trade setups where the price pulls back to key levels during an established trend.

## Features

- **Real-time Trend Correction Detection**: Identify potential trade setups when Bitcoin price corrects to key levels
- **Multiple Technical Indicators**: Uses SMA, RSI, MACD, Fibonacci, and more for signal generation
- **Telegram Notifications**: Receive trading signals directly to your Telegram
- **Backtesting Engine**: Test the strategy on historical data with detailed performance metrics
- **Docker Support**: Easy deployment via Docker containers

## Strategy Overview

The trend correction strategy works on the principle that price tends to return to the mean during established trends. When Bitcoin is in a clear uptrend or downtrend, temporary corrections offer potential entry points for traders. The bot identifies these corrections using:

1. **Trend Identification**: Determines the primary trend using moving average relationships
2. **Correction Detection**: Identifies pullbacks to key support/resistance levels (50 SMA, Fibonacci retracements)
3. **Entry Confirmation**: Confirms potential entries using RSI, MACD, and volume indicators
4. **Confidence Scoring**: Calculates a confidence score for each signal based on indicator confluence

## Installation

### Prerequisites

- Python 3.8+
- Cryptocurrency exchange account (Binance recommended)
- Telegram bot token and chat ID

### Option 1: Direct Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bitcoin-trend-bot.git
   cd bitcoin-trend-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   cp .env.example .env
   # Edit the .env file with your credentials
   ```

### Option 2: Docker Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bitcoin-trend-bot.git
   cd bitcoin-trend-bot
   ```

2. Create a `.env` file with your API keys:
   ```
   cp .env.example .env
   # Edit the .env file with your credentials
   ```

3. Build and run with Docker Compose:
   ```
   docker-compose up -d
   ```

## Usage

### Running the Trading Bot

To start the bot in trading mode:

```
python run.py trade --timeframe 1h --interval 300
```

Options:
- `--exchange`: Exchange to use (default: binance)
- `--symbol`: Trading pair (default: BTC/USDT)
- `--timeframe`: Timeframe for analysis (default: 1h)
- `--interval`: Check interval in seconds (default: 300)

### Running Backtests

To backtest the strategy on historical data:

```

python run.py backtest --start 2023-01-01 --end 2023-12-31 --plot

More examples:

python download_historical_data.py --start 2023-01-01 --end 2023-12-31

python download_historical_data.py --start 2024-01-01 --end 2024-12-31

python run.py backtest --start 2024-01-01 --end 2024-12-31 --csv .\data\BTC_USDT_1h_20240101_20241231.csv --plot


Examples for multiple timeframes check:

python download_multi_timeframe_data.py --start 2023-01-01 --end 2023-12-31


```

Options:
- `--symbol`: Trading pair to backtest (default: BTC/USDT)
- `--timeframe`: Timeframe for backtesting (default: 1h)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--capital`: Initial capital (default: 10000)
- `--fee`: Trading fee rate (default: 0.001)
- `--csv`: Path to CSV file with historical data (optional)
- `--plot`: Plot backtest results (flag)

## Setting Up Telegram Notifications

1. Create a Telegram bot using [BotFather](https://t.me/botfather)
2. Get your bot token from BotFather
3. Create a new group or channel and add your bot to it
4. Get your chat ID (use [@userinfobot](https://t.me/userinfobot) or similar)
5. Add these credentials to your `.env` file:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

## LLM Integration

The bot exports Bitcoin data periodically to CSV files that can be used with Language Model APIs for enhanced analysis:

1. The bot automatically exports data to the `data/` directory
2. You can feed this data to an LLM API (like ChatGPT or Claude) for pattern recognition
3. Use the LLM insights alongside the bot's signals for better decision-making

Example LLM prompt:
```
Analyze this Bitcoin price data CSV with focus on trend corrections. Identify potential entry points and determine if there's a high-probability setup forming currently. Focus on the interplay between the technical indicators.
```

## Customizing the Strategy

To modify the strategy's parameters or add new indicators:

1. Edit `src/indicators.py` to add new technical indicators
2. Modify `src/signal_generator.py` to change signal generation logic
3. Adjust confidence thresholds in `src/trading_bot.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational and research purposes only. Use it at your own risk. Cryptocurrency trading involves significant risk and you can lose your investment. Always do your own research before making trading decisions.