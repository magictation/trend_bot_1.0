#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Timeframe Backtester for the Bitcoin Trend Correction Trading Bot.

This module implements a backtesting engine to evaluate the performance
of the multi-timeframe strategy that uses 5m data for entries and 1h data for exits.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm import tqdm

from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from backtesting.backtester import Backtester
from src.multi_timeframe_bot import MultiTimeframeTrendBot

logger = logging.getLogger("bitcoin_trend_bot.multi_timeframe_backtester")


class MultiTimeframeBacktester(Backtester):
    """Class for backtesting the multi-timeframe Bitcoin trend correction strategy"""
    
    def __init__(self, start_date=None, end_date=None, entry_timeframe='5m', exit_timeframe='1h', 
                 initial_capital=10000, fee_rate=0.001):
        """
        Initialize the multi-timeframe backtester
        
        Args:
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD')
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD')
            entry_timeframe (str): Timeframe for entry signals (default: '5m')
            exit_timeframe (str): Timeframe for exit signals (default: '1h')
            initial_capital (float): Initial capital in USD
            fee_rate (float): Trading fee rate (e.g., 0.001 = 0.1%)
        """
        # Call parent class constructor with exit timeframe
        super().__init__(start_date, end_date, exit_timeframe, initial_capital, fee_rate)
        
        # Add entry timeframe
        self.entry_timeframe = entry_timeframe
        
        # Create bot instance for signal logic
        self.bot = MultiTimeframeTrendBot(entry_timeframe=entry_timeframe, exit_timeframe=exit_timeframe)
        
        # Additional tracking for multi-timeframe
        self.entry_data = None
        self.exit_data = None
        
        logger.info(f"Multi-Timeframe Backtester initialized with entry: {entry_timeframe}, exit: {exit_timeframe}")
    
    def load_data(self, symbol='BTC/USDT'):
        """
        Load historical data for both timeframes
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            tuple: (entry_df, exit_df) with data for both timeframes
        """
        # Fetch exit timeframe data first (1h)
        exit_df = super().load_data(symbol)
        
        if exit_df is not None:
            # Save exit data
            self.exit_data = exit_df.copy()
            
            # Since entry_timeframe (5m) is smaller than exit_timeframe (1h),
            # we need more data points to cover the same time period
            tf_ratio = self._get_timeframe_ratio(self.entry_timeframe, self.timeframe)
            
            # Fetch entry timeframe data
            since = int((self.start_date - timedelta(days=1)).timestamp() * 1000)  # Add buffer
            
            entry_df = self.data_fetcher.fetch_historical_data(
                symbol=symbol,
                timeframe=self.entry_timeframe,
                since=since
            )
            
            if entry_df is not None:
                # Filter by date range
                entry_df = entry_df[(entry_df.index >= pd.Timestamp(self.start_date)) & 
                                   (entry_df.index <= pd.Timestamp(self.end_date))]
                
                logger.info(f"Loaded {len(entry_df)} {self.entry_timeframe} candles from {entry_df.index.min()} to {entry_df.index.max()}")
                
                # Save entry data
                self.entry_data = entry_df.copy()
                
                return entry_df, exit_df
            else:
                logger.error(f"Failed to load {self.entry_timeframe} data")
        
        return None, None
    
    def load_data_from_csv(self, exit_filepath, entry_filepath=None):
        """
        Load historical data from CSV files for both timeframes
        
        Args:
            exit_filepath (str): Path to the CSV file with exit timeframe data
            entry_filepath (str): Path to the CSV file with entry timeframe data
            
        Returns:
            tuple: (entry_df, exit_df) with data for both timeframes
        """
        # Load exit timeframe data
        exit_df = super().load_data_from_csv(exit_filepath)
        
        if exit_df is not None:
            # Save exit data
            self.exit_data = exit_df.copy()
            
            # If entry filepath is not provided, try to derive it
            if entry_filepath is None:
                entry_filepath = exit_filepath.replace(self.timeframe, self.entry_timeframe)
                
            # Check if entry file exists
            if not os.path.exists(entry_filepath):
                logger.error(f"Entry timeframe data file not found: {entry_filepath}")
                return None, None
            
            try:
                # Load entry timeframe data from CSV
                entry_df = pd.read_csv(entry_filepath)
                
                # Check if the CSV has the correct format
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in entry_df.columns for col in required_columns):
                    logger.error(f"CSV file missing required columns: {required_columns}")
                    return None, None
                
                # Convert timestamp to datetime
                if 'timestamp' in entry_df.columns:
                    if pd.api.types.is_numeric_dtype(entry_df['timestamp']):
                        entry_df['timestamp'] = pd.to_datetime(entry_df['timestamp'], unit='ms')
                    else:
                        entry_df['timestamp'] = pd.to_datetime(entry_df['timestamp'])
                        
                    entry_df.set_index('timestamp', inplace=True)
                
                # Filter by date range
                entry_df = entry_df[(entry_df.index >= pd.Timestamp(self.start_date)) & 
                                    (entry_df.index <= pd.Timestamp(self.end_date))]
                
                logger.info(f"Loaded {len(entry_df)} rows from entry CSV")
                
                # Save entry data
                self.entry_data = entry_df.copy()
                
                return entry_df, exit_df
                
            except Exception as e:
                logger.error(f"Error loading data from entry CSV: {e}")
                return None, None
        
        return None, None
    
    def run(self, entry_df=None, exit_df=None):
        """
        Run backtest on historical data for both timeframes
        
        Args:
            entry_df (pandas.DataFrame): Historical OHLCV data for entry timeframe
            exit_df (pandas.DataFrame): Historical OHLCV data for exit timeframe
            
        Returns:
            dict: Backtest results
        """
        # Use stored data if not provided
        if entry_df is None and self.entry_data is not None:
            entry_df = self.entry_data
        
        if exit_df is None and self.exit_data is not None:
            exit_df = self.exit_data
        
        if entry_df is None or exit_df is None or entry_df.empty or exit_df.empty:
            logger.error("No data available for backtesting")
            return None
        
        logger.info(f"Running multi-timeframe backtest on {len(entry_df)} entry candles and {len(exit_df)} exit candles")
        
        # Store price data for plotting
        self.price_data = exit_df.copy()
        
        # Calculate indicators for both timeframes
        processed_entry_df = self._process_entry_timeframe(entry_df)
        processed_exit_df = self.indicators.calculate_all(exit_df)
        
        # Initialize backtest variables
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_time = None
        
        # Trailing stop tracking
        highest_since_entry = 0
        lowest_since_entry = float('inf')
        trailing_stop_pct = 0.05  # 5% trailing stop
        
        # Performance tracking
        self.equity_curve = [(exit_df.index[0], capital)]
        self.trades = []
        
        # Signal tracking for plotting
        self.signals = []
        
        # Process each exit timeframe candle
        for i in tqdm(range(200, len(processed_exit_df)), desc="Backtesting"):
            exit_time = processed_exit_df.index[i]
            exit_price = processed_exit_df['close'].iloc[i]
            
            # Get entry candles up to current exit candle time
            entry_candles_to_now = processed_entry_df[processed_entry_df.index <= exit_time]
            
            # Skip if we don't have enough entry candles
            if len(entry_candles_to_now) < 100:
                continue
            
            # Check exit signals if in position
            if position != 0:
                # Update trailing stop values
                if position == 1:  # Long position
                    highest_since_entry = max(highest_since_entry, exit_price)
                    stop_level = highest_since_entry * (1 - trailing_stop_pct)
                    
                    if exit_price < stop_level:
                        # Trailing stop hit
                        exit_reason = f"Trailing stop hit at ${stop_level:.2f} (high: ${highest_since_entry:.2f})"
                        
                        # Calculate P&L
                        trade_duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                        profit_loss = position * (exit_price - entry_price) / entry_price * capital
                        capital += profit_loss - (capital * self.fee_rate)  # Apply fee
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': "LONG",
                            'profit_loss': profit_loss,
                            'profit_pct': (exit_price - entry_price) / entry_price * 100,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason,
                        })
                        
                        # Add signal to tracking
                        self.signals.append({
                            'time': exit_time,
                            'price': exit_price,
                            'type': 'EXIT_LONG',
                            'equity': capital,
                            'reason': exit_reason
                        })
                        
                        position = 0
                        
                elif position == -1:  # Short position
                    lowest_since_entry = min(lowest_since_entry, exit_price)
                    stop_level = lowest_since_entry * (1 + trailing_stop_pct)
                    
                    if exit_price > stop_level:
                        # Trailing stop hit
                        exit_reason = f"Trailing stop hit at ${stop_level:.2f} (low: ${lowest_since_entry:.2f})"
                        
                        # Calculate P&L
                        trade_duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                        profit_loss = position * (entry_price - exit_price) / entry_price * capital
                        capital += profit_loss - (capital * self.fee_rate)  # Apply fee
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': "SHORT",
                            'profit_loss': profit_loss,
                            'profit_pct': (entry_price - exit_price) / entry_price * 100,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason,
                        })
                        
                        # Add signal to tracking
                        self.signals.append({
                            'time': exit_time,
                            'price': exit_price,
                            'type': 'EXIT_SHORT',
                            'equity': capital,
                            'reason': exit_reason
                        })
                        
                        position = 0
                
                # Check additional exit conditions from 1h chart
                if position != 0:  # Still in position after trailing stop check
                    exit_signal, exit_reason = self._check_exit_signals(
                        processed_exit_df.iloc[:i+1], 
                        exit_price, 
                        position
                    )
                    
                    if exit_signal:
                        # Calculate P&L
                        trade_duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                        
                        if position == 1:
                            profit_loss = (exit_price - entry_price) / entry_price * capital
                            profit_pct = (exit_price - entry_price) / entry_price * 100
                            position_type = "LONG"
                        else:
                            profit_loss = (entry_price - exit_price) / entry_price * capital
                            profit_pct = (entry_price - exit_price) / entry_price * 100
                            position_type = "SHORT"
                            
                        capital += profit_loss - (capital * self.fee_rate)  # Apply fee
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position_type,
                            'profit_loss': profit_loss,
                            'profit_pct': profit_pct,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason,
                        })
                        
                        # Add signal to tracking
                        self.signals.append({
                            'time': exit_time,
                            'price': exit_price,
                            'type': f'EXIT_{position_type}',
                            'equity': capital,
                            'reason': exit_reason
                        })
                        
                        position = 0
            
            # Check entry signals if not in position
            if position == 0:
                # Get last 100 entry candles up to current exit candle
                recent_entry_candles = entry_candles_to_now.iloc[-100:]
                
                # Check for entry signals
                entry_signal, confidence, reason = self._check_entry_signals(recent_entry_candles)
                
                if entry_signal and confidence >= 75:
                    # Enter position
                    position = 1 if entry_signal == "LONG" else -1
                    entry_price = exit_price  # Using exit_price as approximation for last entry price
                    entry_time = exit_time
                    
                    # Reset trailing stop values
                    if position == 1:
                        highest_since_entry = entry_price
                    else:
                        lowest_since_entry = entry_price
                    
                    # Apply entry fee
                    capital -= capital * self.fee_rate
                    
                    # Add signal to tracking
                    self.signals.append({
                        'time': exit_time,
                        'price': exit_price,
                        'type': f'ENTER_{"LONG" if position == 1 else "SHORT"}',
                        'equity': capital,
                        'confidence': confidence,
                        'reason': reason
                    })
            
            # Update equity curve
            if position == 0:
                current_equity = capital
            elif position == 1:
                # Long position value
                unrealized_pnl = (exit_price - entry_price) / entry_price * capital
                current_equity = capital + unrealized_pnl
            else:  # position == -1
                # Short position value
                unrealized_pnl = (entry_price - exit_price) / entry_price * capital
                current_equity = capital + unrealized_pnl
                
            self.equity_curve.append((exit_time, current_equity))
        
        # Close any open position at the end
        if position != 0:
            exit_time = processed_exit_df.index[-1]
            exit_price = processed_exit_df['close'].iloc[-1]
            
            # Calculate P&L
            trade_duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            
            if position == 1:
                profit_loss = (exit_price - entry_price) / entry_price * capital
                profit_pct = (exit_price - entry_price) / entry_price * 100
                position_type = "LONG"
            else:
                profit_loss = (entry_price - exit_price) / entry_price * capital
                profit_pct = (entry_price - exit_price) / entry_price * 100
                position_type = "SHORT"
                
            capital += profit_loss - (capital * self.fee_rate)  # Apply fee
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position_type,
                'profit_loss': profit_loss,
                'profit_pct': profit_pct,
                'duration_hours': trade_duration,
                'exit_reason': "End of backtest",
            })
            
            # Add signal to tracking
            self.signals.append({
                'time': exit_time,
                'price': exit_price,
                'type': f'EXIT_{position_type}',
                'equity': capital,
                'reason': "End of backtest"
            })
        
        # Calculate performance metrics
        from backtesting.performance import calculate_performance
        results = calculate_performance(self.trades, self.equity_curve, self.initial_capital)
        
        return results
    
    def _process_entry_timeframe(self, df):
        """Process the entry timeframe data with specialized indicators"""
        # Use the bot's method
        return self.bot._process_entry_timeframe(df)
    
    def _check_entry_signals(self, df):
        """Check for entry signals on the 5m timeframe"""
        # Use the bot's method but with data window
        bot = self.bot  # Create a temporary bot instance
        
        # Call the entry signal method directly
        return bot._check_entry_signals(df)
    
    def _check_exit_signals(self, df, current_price, position):
        """
        Check for exit signals on the 1h timeframe
        
        Args:
            df (pandas.DataFrame): Historical data with indicators
            current_price (float): Current price
            position (int): Current position (1 for long, -1 for short)
            
        Returns:
            tuple: (exit_signal, reason)
        """
        # Get the most recent data points
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Initialize
        exit_signal = False
        reason = ""
        
        # 1. Moving average crossover (trend change)
        if position == 1 and previous['sma20'] > previous['sma50'] and current['sma20'] < current['sma50']:
            exit_signal = True
            reason = "Bearish moving average crossover (20 SMA crossed below 50 SMA)"
        elif position == -1 and previous['sma20'] < previous['sma50'] and current['sma20'] > current['sma50']:
            exit_signal = True
            reason = "Bullish moving average crossover (20 SMA crossed above 50 SMA)"
        
        # 2. Trend exhaustion via RSI
        if not exit_signal:
            if position == 1 and current['rsi'] > 80 and current['rsi'] < previous['rsi']:
                exit_signal = True
                reason = f"Potential trend exhaustion (RSI: {current['rsi']:.2f})"
            elif position == -1 and current['rsi'] < 20 and current['rsi'] > previous['rsi']:
                exit_signal = True
                reason = f"Potential trend exhaustion (RSI: {current['rsi']:.2f})"
        
        # 3. Volume climax
        if not exit_signal and current['volume_ratio'] > 3.0:
            # Extreme volume can signal exhaustion
            if (position == 1 and current['close'] < current['open']) or \
               (position == -1 and current['close'] > current['open']):
                exit_signal = True
                reason = f"Volume climax with adverse price action (volume: {current['volume_ratio']:.2f}x average)"
        
        return exit_signal, reason
    
    def _get_timeframe_ratio(self, smaller_tf, larger_tf):
        """Calculate ratio between two timeframes"""
        # Convert timeframes to minutes
        def tf_to_minutes(tf):
            unit = tf[-1]
            value = int(tf[:-1])
            
            if unit == 'm':
                return value
            elif unit == 'h':
                return value * 60
            elif unit == 'd':
                return value * 1440
            else:
                return value
        
        smaller_mins = tf_to_minutes(smaller_tf)
        larger_mins = tf_to_minutes(larger_tf)
        
        return larger_mins / smaller_mins
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results with signal markers
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        # Call the parent class method
        super().plot_results(save_path)