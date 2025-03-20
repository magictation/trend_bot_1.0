#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point script for the Multi-Timeframe Bitcoin Trend Correction Trading Bot.

This script provides command-line options to run the multi-timeframe bot in
trading mode or backtesting mode.
"""

import os
import argparse
import logging
from datetime import datetime

from src.multi_timeframe_bot import MultiTimeframeTrendBot
from backtesting.multi_timeframe_backtester import MultiTimeframeBacktester
from backtesting.performance import print_performance_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"multi_timeframe_bot_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_timeframe_bot")


def main():
    """Main function to parse arguments and run the bot"""
    
    parser = argparse.ArgumentParser(description='Multi-Timeframe Bitcoin Trend Correction Trading Bot')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Trading mode arguments
    trade_parser = subparsers.add_parser('trade', help='Run the bot in trading mode')
    trade_parser.add_argument('--exchange', type=str, default='binance', help='Exchange (default: binance)')
    trade_parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    trade_parser.add_argument('--entry-timeframe', type=str, default='5m', help='Entry timeframe (default: 5m)')
    trade_parser.add_argument('--exit-timeframe', type=str, default='1h', help='Exit timeframe (default: 1h)')
    trade_parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    
    # Backtest mode arguments
    backtest_parser = subparsers.add_parser('backtest', help='Run the bot in backtesting mode')
    backtest_parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    backtest_parser.add_argument('--entry-timeframe', type=str, default='5m', help='Entry timeframe (default: 5m)')
    backtest_parser.add_argument('--exit-timeframe', type=str, default='1h', help='Exit timeframe (default: 1h)')
    backtest_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital (default: 10000)')
    backtest_parser.add_argument('--fee', type=float, default=0.001, help='Trading fee (default: 0.001)')
    backtest_parser.add_argument('--entry-csv', type=str, help='Path to CSV file with entry timeframe data')
    backtest_parser.add_argument('--exit-csv', type=str, help='Path to CSV file with exit timeframe data')
    backtest_parser.add_argument('--plot', action='store_true', help='Plot backtest results')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    if args.mode == 'trade':
        # Run in trading mode
        logger.info("Starting Multi-Timeframe Bitcoin Trend Correction Trading Bot in trading mode")
        
        bot = MultiTimeframeTrendBot(
            exchange_id=args.exchange,
            entry_timeframe=args.entry_timeframe,
            exit_timeframe=args.exit_timeframe
        )
        
        bot.run(
            symbol=args.symbol,
            check_interval=args.interval
        )
        
    elif args.mode == 'backtest':
        # Run in backtesting mode
        logger.info("Starting Multi-Timeframe Bitcoin Trend Correction Trading Bot in backtesting mode")
        
        backtester = MultiTimeframeBacktester(
            start_date=args.start,
            end_date=args.end,
            entry_timeframe=args.entry_timeframe,
            exit_timeframe=args.exit_timeframe,
            initial_capital=args.capital,
            fee_rate=args.fee
        )
        
        # Load data from CSV or exchange
        if args.entry_csv and args.exit_csv:
            entry_df, exit_df = backtester.load_data_from_csv(args.exit_csv, args.entry_csv)
        else:
            entry_df, exit_df = backtester.load_data(symbol=args.symbol)
        
        # Run backtest
        if entry_df is not None and exit_df is not None:
            results = backtester.run(entry_df, exit_df)
            print_performance_report(results)
            
            if args.plot:
                backtester.plot_results('multi_timeframe_backtest_results.png')
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()