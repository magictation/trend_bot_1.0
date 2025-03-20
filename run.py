#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point script for the Enhanced Bitcoin Trend Correction Trading Bot.

This script provides command-line options to run the enhanced bot in
trading mode or backtesting mode with multi-timeframe support.
"""

import os
import argparse
import logging
from datetime import datetime

from src.trading_bot import EnhancedTrendBot
from backtesting.backtester import EnhancedBacktester
from backtesting.performance import print_performance_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"enhanced_bot_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_bitcoin_bot")


def main():
    """Main function to parse arguments and run the enhanced bot"""
    
    parser = argparse.ArgumentParser(description='Enhanced Bitcoin Trend Correction Trading Bot')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Trading mode arguments
    trade_parser = subparsers.add_parser('trade', help='Run the bot in trading mode')
    trade_parser.add_argument('--exchange', type=str, default='binance', help='Exchange (default: binance)')
    trade_parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    trade_parser.add_argument('--primary-timeframe', type=str, default='2h', help='Primary timeframe (default: 2h)')  # Changed from 1h
    trade_parser.add_argument('--confirmation-timeframe', type=str, default='8h', help='Trend confirmation timeframe (default: 8h)')  # Changed from 4h
    trade_parser.add_argument('--scanning-timeframe', type=str, default='30m', help='Entry scanning timeframe (default: 30m)')  # Changed from 15m
    trade_parser.add_argument('--interval', type=int, default=180, help='Check interval in seconds (default: 180)')
    
    # Backtest mode arguments
    backtest_parser = subparsers.add_parser('backtest', help='Run the bot in backtesting mode')
    backtest_parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    backtest_parser.add_argument('--primary-timeframe', type=str, default='2h', help='Primary timeframe (default: 2h)')  # Changed from 1h
    backtest_parser.add_argument('--confirmation-timeframe', type=str, default='8h', help='Trend confirmation timeframe (default: 8h)')  # Changed from 4h
    backtest_parser.add_argument('--scanning-timeframe', type=str, default='30m', help='Entry scanning timeframe (default: 30m)')  # Changed from 15m
    backtest_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital (default: 10000)')
    backtest_parser.add_argument('--fee', type=float, default=0.001, help='Trading fee (default: 0.001)')
    backtest_parser.add_argument('--primary-csv', type=str, help='Path to CSV file with primary timeframe data')
    backtest_parser.add_argument('--confirmation-csv', type=str, help='Path to CSV file with confirmation timeframe data')
    backtest_parser.add_argument('--scanning-csv', type=str, help='Path to CSV file with scanning timeframe data')
    backtest_parser.add_argument('--plot', action='store_true', help='Plot backtest results')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    if args.mode == 'trade':
        # Run in trading mode
        logger.info("Starting Enhanced Bitcoin Trend Correction Trading Bot in trading mode")
        
        bot = EnhancedTrendBot(
            exchange_id=args.exchange,
            primary_timeframe=args.primary_timeframe,
            confirmation_timeframe=args.confirmation_timeframe,
            scanning_timeframe=args.scanning_timeframe
        )
        
        bot.run(
            symbol=args.symbol,
            interval=args.interval
        )
        
    elif args.mode == 'backtest':
        # Run in backtesting mode
        logger.info("Starting Enhanced Bitcoin Trend Correction Trading Bot in backtesting mode")
        
        backtester = EnhancedBacktester(
            start_date=args.start,
            end_date=args.end,
            primary_timeframe=args.primary_timeframe,
            confirmation_timeframe=args.confirmation_timeframe,
            scanning_timeframe=args.scanning_timeframe,
            initial_capital=args.capital,
            fee_rate=args.fee
        )
        
        # Load data from CSV or exchange
        if args.primary_csv:
            scanning_df, primary_df, confirmation_df = backtester.load_data_from_csv(
                args.primary_csv, 
                args.confirmation_csv, 
                args.scanning_csv
            )
        else:
            scanning_df, primary_df, confirmation_df = backtester.load_data(symbol=args.symbol)
        
        # Run backtest
        if primary_df is not None:
            results = backtester.run(scanning_df, primary_df, confirmation_df)
            print_performance_report(results)
            
            if args.plot:
                backtester.plot_results('enhanced_backtest_results.png')
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()