#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to download historical Bitcoin data for backtesting.

This script downloads historical OHLCV data for Bitcoin from an exchange
and saves it to a CSV file for later use in backtesting.
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_downloader")


def download_historical_data(exchange_id='binance', symbol='BTC/USDT', timeframe='1h', 
                             start_date=None, end_date=None, output_dir='data'):
    """
    Download historical OHLCV data and save to CSV
    
    Args:
        exchange_id (str): Exchange to use (default: binance)
        symbol (str): Trading pair (default: BTC/USDT)
        timeframe (str): Timeframe (default: 1h)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Output directory for CSV files
    
    Returns:
        str: Path to the output CSV file
    """
    # Initialize exchange
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
    })
    
    # Parse dates
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
    if start_date is None:
        start_dt = end_dt - timedelta(days=90)  # Default to 90 days
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Convert to timestamps
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    logger.info(f"Downloading {symbol} data from {start_dt.date()} to {end_dt.date()}")
    
    # Get the timeframe in milliseconds
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    
    # Fetch data in chunks
    all_candles = []
    current = start_ts
    
    while current < end_ts:
        try:
            logger.info(f"Fetching data from {datetime.fromtimestamp(current/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current,
                limit=1000
            )
            
            if not candles:
                logger.warning("No data returned from exchange")
                break
            
            all_candles.extend(candles)
            
            # Update progress
            current = candles[-1][0] + timeframe_ms
            
            # Respect rate limits
            exchange.sleep(exchange.rateLimit)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            # Try to continue from the next time period
            current += timeframe_ms * 500
    
    if not all_candles:
        logger.error("Failed to fetch any data")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates and sort
    df = df[~df.duplicated(subset=['timestamp'])]
    df.sort_values('timestamp', inplace=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    filename = f"{output_dir}/{symbol.replace('/', '_')}_{timeframe}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    logger.info(f"Saved {len(df)} candles to {filename}")
    
    return filename


def main():
    """Main function to parse arguments and download data"""
    
    parser = argparse.ArgumentParser(description='Download historical cryptocurrency data')
    
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to use (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe (default: 1h)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    download_historical_data(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()