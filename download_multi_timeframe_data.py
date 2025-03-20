#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to download historical Bitcoin data for both entry (5m) and exit (1h) timeframes.

This script downloads historical OHLCV data for Bitcoin from an exchange
and saves it to CSV files for later use in multi-timeframe backtesting.
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_timeframe_data_downloader")


def download_historical_data(exchange_id='binance', symbol='BTC/USDT', 
                             entry_timeframe='5m', exit_timeframe='1h',
                             start_date=None, end_date=None, output_dir='data'):
    """
    Download historical OHLCV data for multiple timeframes and save to CSV
    
    Args:
        exchange_id (str): Exchange to use (default: binance)
        symbol (str): Trading pair (default: BTC/USDT)
        entry_timeframe (str): Entry timeframe (default: 5m)
        exit_timeframe (str): Exit timeframe (default: 1h)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Output directory for CSV files
    
    Returns:
        tuple: Paths to the output CSV files (entry_path, exit_path)
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
    
    # Add extra days for the entry timeframe to ensure we have enough data
    entry_start_dt = start_dt - timedelta(days=1)
    
    # Convert to timestamps
    entry_start_ts = int(entry_start_dt.timestamp() * 1000)
    exit_start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    logger.info(f"Downloading {symbol} data from {start_dt.date()} to {end_dt.date()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download exit timeframe data (1h)
    exit_filename = f"{output_dir}/{symbol.replace('/', '_')}_{exit_timeframe}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    logger.info(f"Downloading {exit_timeframe} data...")
    
    exit_data = download_timeframe(exchange, symbol, exit_timeframe, exit_start_ts, end_ts)
    
    if exit_data is not None and not exit_data.empty:
        # Save to CSV
        exit_data.to_csv(exit_filename, index=False)
        logger.info(f"Saved {len(exit_data)} {exit_timeframe} candles to {exit_filename}")
    else:
        logger.error(f"Failed to download {exit_timeframe} data")
        return None, None
    
    # Download entry timeframe data (5m)
    entry_filename = f"{output_dir}/{symbol.replace('/', '_')}_{entry_timeframe}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    logger.info(f"Downloading {entry_timeframe} data...")
    
    entry_data = download_timeframe(exchange, symbol, entry_timeframe, entry_start_ts, end_ts)
    
    if entry_data is not None and not entry_data.empty:
        # Save to CSV
        entry_data.to_csv(entry_filename, index=False)
        logger.info(f"Saved {len(entry_data)} {entry_timeframe} candles to {entry_filename}")
    else:
        logger.error(f"Failed to download {entry_timeframe} data")
        return exit_filename, None
    
    return entry_filename, exit_filename


def download_timeframe(exchange, symbol, timeframe, start_ts, end_ts):
    """
    Download historical data for a specific timeframe
    
    Args:
        exchange: ccxt exchange instance
        symbol (str): Trading pair
        timeframe (str): Timeframe
        start_ts (int): Start timestamp in milliseconds
        end_ts (int): End timestamp in milliseconds
    
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data
    """
    # Get the timeframe in milliseconds
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    
    # Calculate approximate number of candles
    total_ms = end_ts - start_ts
    estimated_candles = int(total_ms / timeframe_ms)
    
    # Fetch data in chunks
    all_candles = []
    current = start_ts
    
    with tqdm(total=estimated_candles, desc=f"Downloading {timeframe} data") as pbar:
        while current < end_ts:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current,
                    limit=1000
                )
                
                if not candles:
                    logger.warning(f"No {timeframe} data returned from exchange for period starting at {datetime.fromtimestamp(current/1000)}")
                    break
                
                all_candles.extend(candles)
                
                # Update progress
                pbar.update(len(candles))
                
                # Update timestamp for next batch
                current = candles[-1][0] + timeframe_ms
                
                # Respect rate limits
                exchange.sleep(exchange.rateLimit)
                
            except Exception as e:
                logger.error(f"Error fetching {timeframe} data: {e}")
                # Try to continue from the next time period
                current += timeframe_ms * 100
    
    if not all_candles:
        logger.error(f"Failed to fetch any {timeframe} data")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates and sort
    df = df[~df.duplicated(subset=['timestamp'])]
    df.sort_values('timestamp', inplace=True)
    
    return df


def main():
    """Main function to parse arguments and download data"""
    
    parser = argparse.ArgumentParser(description='Download historical data for multi-timeframe backtesting')
    
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to use (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--entry-timeframe', type=str, default='5m',
                        help='Entry timeframe (default: 5m)')
    parser.add_argument('--exit-timeframe', type=str, default='1h',
                        help='Exit timeframe (default: 1h)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    download_historical_data(
        exchange_id=args.exchange,
        symbol=args.symbol,
        entry_timeframe=args.entry_timeframe,
        exit_timeframe=args.exit_timeframe,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()