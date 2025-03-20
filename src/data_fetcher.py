#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Fetcher for Bitcoin Trend Correction Trading Bot.

This module handles fetching OHLCV data from cryptocurrency exchanges.
"""

import os
import logging
import pandas as pd
import ccxt

logger = logging.getLogger("bitcoin_trend_bot.data_fetcher")


class DataFetcher:
    """Class for fetching price data from exchanges"""
    
    def __init__(self, exchange_id='binance'):
        """
        Initialize the DataFetcher
        
        Args:
            exchange_id (str): The exchange to use (default: 'binance')
        """
        self.exchange_id = exchange_id
        
        # Initialize exchange
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': os.getenv('EXCHANGE_API_KEY'),
            'secret': os.getenv('EXCHANGE_API_SECRET'),
            'enableRateLimit': True,
        })
        
        logger.info(f"DataFetcher initialized with {exchange_id}")
    
    def fetch_ohlcv_data(self, symbol='BTC/USDT', timeframe='1h', limit=100):
        """
        Fetch OHLCV data from the exchange
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for data
            limit (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data from the exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe} timeframe")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None
    
    def fetch_historical_data(self, symbol='BTC/USDT', timeframe='1h', since=None, limit=1000):
        """
        Fetch historical OHLCV data from the exchange
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for data
            since (int): Timestamp in milliseconds to fetch data since
            limit (int): Maximum number of candles per request
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            all_ohlcv = []
            
            # If since is not provided, use a default lookback period
            if since is None:
                # Get approximately 3 months of data
                timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
                since = self.exchange.milliseconds() - (90 * 24 * 60 * 60 * 1000)  # 90 days back
            
            # Fetch data in chunks to avoid exchange limits
            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # Update since for the next batch
                since = ohlcv[-1][0] + 1
                
                # If we got less than the requested limit, we've reached the end
                if len(ohlcv) < limit:
                    break
            
            # Convert to DataFrame
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                
                logger.info(f"Fetched {len(df)} historical candles for {symbol} on {timeframe} timeframe")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None