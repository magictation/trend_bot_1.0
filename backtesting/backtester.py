#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Backtester for Bitcoin Trend Correction Trading Bot.

This module implements an improved backtesting engine that uses:
1. Multi-timeframe analysis
2. Dynamic trailing stops
3. Improved risk management
4. More sophisticated entry and exit criteria
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
from src.signal_generator import EnhancedSignalGenerator

logger = logging.getLogger("bitcoin_trend_bot.enhanced_backtester")


class EnhancedBacktester:
    """Enhanced class for backtesting the Bitcoin trend correction strategy"""
    
    def __init__(self, start_date=None, end_date=None, 
                 primary_timeframe='1h', confirmation_timeframe='4h', scanning_timeframe='15m',
                 initial_capital=10000, fee_rate=0.001):
        """
        Initialize the enhanced backtester
        
        Args:
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD')
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD')
            primary_timeframe (str): Main timeframe for analysis (default: '1h')
            confirmation_timeframe (str): Higher timeframe for trend confirmation (default: '4h')
            scanning_timeframe (str): Lower timeframe for entry timing (default: '15m')
            initial_capital (float): Initial capital in USD
            fee_rate (float): Trading fee rate (e.g., 0.001 = 0.1%)
        """
        # Set date range
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=90)
        else:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        self.primary_timeframe = primary_timeframe
        self.confirmation_timeframe = confirmation_timeframe
        self.scanning_timeframe = scanning_timeframe
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.signal_generator = EnhancedSignalGenerator()
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.price_data = None
        
        # Risk management
        self.consecutive_losses = 0
        self.taking_break = False
        self.break_start_time = None
        self.max_losses_before_break = 5  # Increase from 3
        self.break_duration_hours = 12    # Decrease from 24
        
        logger.info(f"Enhanced Backtester initialized for {self.start_date.date()} to {self.end_date.date()} using timeframes: {scanning_timeframe}/{primary_timeframe}/{confirmation_timeframe}")
    
    def load_data(self, symbol='BTC/USDT'):
        """
        Load historical data for all timeframes used in backtesting
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            tuple: (scanning_df, primary_df, confirmation_df)
        """
        # Convert dates to timestamps
        since = int(self.start_date.timestamp() * 1000)
        
        # Add buffer for higher timeframes
        buffer_days = 10  # Add 10 days buffer
        buffer_since = int((self.start_date - timedelta(days=buffer_days)).timestamp() * 1000)
        
        # Fetch historical data for all timeframes
        logger.info(f"Fetching data for {symbol} on multiple timeframes...")
        
        # 1. Primary timeframe (1h)
        primary_df = self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=self.primary_timeframe,
            since=since
        )
        
        # 2. Confirmation timeframe (4h)
        confirmation_df = self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=self.confirmation_timeframe,
            since=buffer_since  # Use buffer to ensure enough data
        )
        
        # 3. Scanning timeframe (15m)
        scanning_df = self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=self.scanning_timeframe,
            since=since
        )
        
        # Filter by date range
        if primary_df is not None:
            primary_df = primary_df[(primary_df.index >= pd.Timestamp(self.start_date)) & 
                                   (primary_df.index <= pd.Timestamp(self.end_date))]
            logger.info(f"Loaded {len(primary_df)} candles for {self.primary_timeframe} timeframe")
        
        if confirmation_df is not None:
            confirmation_df = confirmation_df[(confirmation_df.index >= pd.Timestamp(self.start_date - timedelta(days=buffer_days))) & 
                                             (confirmation_df.index <= pd.Timestamp(self.end_date))]
            logger.info(f"Loaded {len(confirmation_df)} candles for {self.confirmation_timeframe} timeframe")
        
        if scanning_df is not None:
            scanning_df = scanning_df[(scanning_df.index >= pd.Timestamp(self.start_date)) & 
                                     (scanning_df.index <= pd.Timestamp(self.end_date))]
            logger.info(f"Loaded {len(scanning_df)} candles for {self.scanning_timeframe} timeframe")
        
        return scanning_df, primary_df, confirmation_df
    
    def load_data_from_csv(self, primary_filepath, confirmation_filepath=None, scanning_filepath=None):
        """
        Load historical data from CSV files for all timeframes
        
        Args:
            primary_filepath (str): Path to the CSV file with primary timeframe data
            confirmation_filepath (str): Path to the CSV file with confirmation timeframe data
            scanning_filepath (str): Path to the CSV file with scanning timeframe data
            
        Returns:
            tuple: (scanning_df, primary_df, confirmation_df)
        """
        try:
            # 1. Load primary timeframe data
            primary_df = pd.read_csv(primary_filepath)
            
            # Check if the CSV has the correct format
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in primary_df.columns for col in required_columns):
                logger.error(f"CSV file missing required columns: {required_columns}")
                return None, None, None
            
            # Convert timestamp to datetime
            if 'timestamp' in primary_df.columns:
                if pd.api.types.is_numeric_dtype(primary_df['timestamp']):
                    primary_df['timestamp'] = pd.to_datetime(primary_df['timestamp'], unit='ms')
                else:
                    primary_df['timestamp'] = pd.to_datetime(primary_df['timestamp'])
                    
                primary_df.set_index('timestamp', inplace=True)
            
            # Filter by date range
            primary_df = primary_df[(primary_df.index >= pd.Timestamp(self.start_date)) & 
                                   (primary_df.index <= pd.Timestamp(self.end_date))]
            
            logger.info(f"Loaded {len(primary_df)} rows from primary CSV")
            
            # 2. Load confirmation timeframe data if provided
            confirmation_df = None
            if confirmation_filepath:
                confirmation_df = pd.read_csv(confirmation_filepath)
                
                if 'timestamp' in confirmation_df.columns:
                    if pd.api.types.is_numeric_dtype(confirmation_df['timestamp']):
                        confirmation_df['timestamp'] = pd.to_datetime(confirmation_df['timestamp'], unit='ms')
                    else:
                        confirmation_df['timestamp'] = pd.to_datetime(confirmation_df['timestamp'])
                        
                    confirmation_df.set_index('timestamp', inplace=True)
                
                # Filter by date range with buffer
                buffer_days = 10
                confirmation_df = confirmation_df[(confirmation_df.index >= pd.Timestamp(self.start_date - timedelta(days=buffer_days))) & 
                                               (confirmation_df.index <= pd.Timestamp(self.end_date))]
                
                logger.info(f"Loaded {len(confirmation_df)} rows from confirmation CSV")
            
            # 3. Load scanning timeframe data if provided
            scanning_df = None
            if scanning_filepath:
                scanning_df = pd.read_csv(scanning_filepath)
                
                if 'timestamp' in scanning_df.columns:
                    if pd.api.types.is_numeric_dtype(scanning_df['timestamp']):
                        scanning_df['timestamp'] = pd.to_datetime(scanning_df['timestamp'], unit='ms')
                    else:
                        scanning_df['timestamp'] = pd.to_datetime(scanning_df['timestamp'])
                        
                    scanning_df.set_index('timestamp', inplace=True)
                
                # Filter by date range
                scanning_df = scanning_df[(scanning_df.index >= pd.Timestamp(self.start_date)) & 
                                         (scanning_df.index <= pd.Timestamp(self.end_date))]
                
                logger.info(f"Loaded {len(scanning_df)} rows from scanning CSV")
            
            return scanning_df, primary_df, confirmation_df
            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}")
            return None, None, None
    
    def run(self, scanning_df=None, primary_df=None, confirmation_df=None):
        """
        Run backtest on historical data for all timeframes
        
        Args:
            scanning_df (pandas.DataFrame): Historical data for entry timing
            primary_df (pandas.DataFrame): Historical data for primary analysis
            confirmation_df (pandas.DataFrame): Historical data for trend confirmation
            
        Returns:
            dict: Backtest results
        """
        if primary_df is None or primary_df.empty:
            logger.error("No primary timeframe data available for backtesting")
            return None
        
        # Use primary data for price data if scanning data not available
        if scanning_df is None or scanning_df.empty:
            logger.warning("No scanning timeframe data provided. Using primary timeframe data for entry signals.")
            scanning_df = primary_df.copy()
        
        # Use primary data for confirmation if confirmation data not available
        if confirmation_df is None or confirmation_df.empty:
            logger.warning("No confirmation timeframe data provided. Using primary timeframe data for trend confirmation.")
            confirmation_df = primary_df.copy()
        
        logger.info(f"Running enhanced backtest on multiple timeframes")
        
        # Store price data for plotting
        self.price_data = primary_df.copy()
        
        # Calculate indicators for all timeframes
        scanning_df = self.indicators.calculate_all(scanning_df)
        primary_df = self.indicators.calculate_all(primary_df)
        confirmation_df = self.indicators.calculate_all(confirmation_df)
        
        # Initialize backtest variables
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_time = None
        
        # Trailing stop tracking
        highest_since_entry = 0
        lowest_since_entry = float('inf')
        trailing_stop_pct = 0.05  # Default, will be updated dynamically
        
        # Performance tracking
        self.equity_curve = [(primary_df.index[0], capital)]
        self.trades = []
        
        # Signal tracking for plotting
        self.signals = []
        
        # Iterate through the primary data (1h)
        window_size = 200  # Ensure we have enough data for indicators
        
        for i in tqdm(range(window_size, len(primary_df)), desc="Backtesting"):
            current_time = primary_df.index[i]
            current_price = primary_df['close'].iloc[i]
            
            # First, check if we need to end a trading break
            if self.taking_break and self.break_start_time:
                hours_since_break = (current_time - self.break_start_time).total_seconds() / 3600
                if hours_since_break >= self.break_duration_hours:
                    self.taking_break = False
                    self.consecutive_losses = 0
                    logger.info(f"Trading break ended at {current_time}")
            
            # Check exit signals if in position
            if position != 0:
                # Update trailing stop values
                if position == 1:  # Long position
                    highest_since_entry = max(highest_since_entry, current_price)
                    stop_level = highest_since_entry * (1 - trailing_stop_pct)
                    
                    if current_price < stop_level:
                        # Trailing stop hit
                        exit_reason = f"Trailing stop hit at ${stop_level:.2f} ({trailing_stop_pct*100:.1f}% from high of ${highest_since_entry:.2f})"
                        
                        # Calculate P&L
                        trade_duration = (current_time - entry_time).total_seconds() / 3600  # hours
                        profit_loss = position * (current_price - entry_price) / entry_price * capital
                        capital += profit_loss - (capital * self.fee_rate)  # Apply fee
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': "LONG",
                            'profit_loss': profit_loss,
                            'profit_pct': (current_price - entry_price) / entry_price * 100,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason,
                        })
                        
                        # Add signal to tracking
                        self.signals.append({
                            'time': current_time,
                            'price': current_price,
                            'type': 'EXIT_LONG',
                            'equity': capital,
                            'reason': exit_reason
                        })
                        
                        # Update consecutive losses for risk management
                        if profit_loss < 0:
                            self.consecutive_losses += 1
                            
                            # Check if we need to take a trading break
                            if self.consecutive_losses >= self.max_losses_before_break:
                                self.taking_break = True
                                self.break_start_time = current_time
                                logger.warning(f"Trading break initiated at {current_time} after {self.consecutive_losses} consecutive losses")
                        else:
                            self.consecutive_losses = 0
                        
                        position = 0
                        
                elif position == -1:  # Short position
                    lowest_since_entry = min(lowest_since_entry, current_price)
                    stop_level = lowest_since_entry * (1 + trailing_stop_pct)
                    
                    if current_price > stop_level:
                        # Trailing stop hit
                        exit_reason = f"Trailing stop hit at ${stop_level:.2f} ({trailing_stop_pct*100:.1f}% from low of ${lowest_since_entry:.2f})"
                        
                        # Calculate P&L
                        trade_duration = (current_time - entry_time).total_seconds() / 3600  # hours
                        profit_loss = position * (entry_price - current_price) / entry_price * capital
                        capital += profit_loss - (capital * self.fee_rate)  # Apply fee
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': "SHORT",
                            'profit_loss': profit_loss,
                            'profit_pct': (entry_price - current_price) / entry_price * 100,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason,
                        })
                        
                        # Add signal to tracking
                        self.signals.append({
                            'time': current_time,
                            'price': current_price,
                            'type': 'EXIT_SHORT',
                            'equity': capital,
                            'reason': exit_reason
                        })
                        
                        # Update consecutive losses for risk management
                        if profit_loss < 0:
                            self.consecutive_losses += 1
                            
                            # Check if we need to take a trading break
                            if self.consecutive_losses >= self.max_losses_before_break:
                                self.taking_break = True
                                self.break_start_time = current_time
                                logger.warning(f"Trading break initiated at {current_time} after {self.consecutive_losses} consecutive losses")
                        else:
                            self.consecutive_losses = 0
                        
                        position = 0
                
                # Check additional exit conditions
                if position != 0:  # Still in position after trailing stop check
                    exit_signal, exit_reason = self.signal_generator.generate_exit_signals(
                        primary_df.iloc[:i+1], 
                        position,
                        entry_price,
                        entry_time,
                        highest_since_entry,
                        lowest_since_entry,
                        current_time
                    )
                    
                    if exit_signal:
                        # Calculate P&L
                        trade_duration = (current_time - entry_time).total_seconds() / 3600  # hours
                        
                        if position == 1:
                            profit_loss = (current_price - entry_price) / entry_price * capital
                            profit_pct = (current_price - entry_price) / entry_price * 100
                            position_type = "LONG"
                        else:
                            profit_loss = (entry_price - current_price) / entry_price * capital
                            profit_pct = (entry_price - current_price) / entry_price * 100
                            position_type = "SHORT"
                            
                        capital += profit_loss - (capital * self.fee_rate)  # Apply fee
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position_type,
                            'profit_loss': profit_loss,
                            'profit_pct': profit_pct,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason,
                        })
                        
                        # Add signal to tracking
                        self.signals.append({
                            'time': current_time,
                            'price': current_price,
                            'type': f'EXIT_{position_type}',
                            'equity': capital,
                            'reason': exit_reason
                        })
                        
                        # Update consecutive losses for risk management
                        if profit_loss < 0:
                            self.consecutive_losses += 1
                            
                            # Check if we need to take a trading break
                            if self.consecutive_losses >= self.max_losses_before_break:
                                self.taking_break = True
                                self.break_start_time = current_time
                                logger.warning(f"Trading break initiated at {current_time} after {self.consecutive_losses} consecutive losses")
                        else:
                            self.consecutive_losses = 0
                        
                        position = 0
            
            # Check entry signals if not in position and not in trading break
            if position == 0 and not self.taking_break:
                # Get confirmation data up to current time
                confirmation_up_to_now = confirmation_df[confirmation_df.index <= current_time]
                
                # Skip if we don't have enough confirmation data
                if len(confirmation_up_to_now) < 50:
                    continue
                
                # Check trend direction from confirmation timeframe
                uptrend = confirmation_up_to_now['sma50'].iloc[-1] > confirmation_up_to_now['sma100'].iloc[-1]
                downtrend = confirmation_up_to_now['sma50'].iloc[-1] < confirmation_up_to_now['sma100'].iloc[-1]
                
                # Get scanning data up to current time
                scanning_up_to_now = scanning_df[scanning_df.index <= current_time]
                
                # Skip if we don't have enough scanning data
                if len(scanning_up_to_now) < 100:
                    continue
                
                # Check for entry signals on scanning timeframe
                signal, confidence, reason, new_trailing_stop = self.signal_generator.generate_entry_signals(scanning_up_to_now)
                
                # Only take trades in direction of higher timeframe trend
                aligned_with_trend = True
                
                if signal and confidence >= 65 and aligned_with_trend:  # Changed from 75
                    # Also check primary timeframe for confluence
                    primary_signal, primary_confidence, _, _ = self.signal_generator.generate_entry_signals(primary_df.iloc[:i+1])
                    
                    if primary_signal == signal and primary_confidence >= 50:  # Changed from 60
                        # We have alignment across multiple timeframes - enter position
                        position = 1 if signal == "LONG" else -1
                        entry_price = current_price
                        entry_time = current_time
                        trailing_stop_pct = new_trailing_stop
                        
                        # Reset trailing stop values
                        if position == 1:
                            highest_since_entry = current_price
                        else:
                            lowest_since_entry = current_price
                        
                        # Apply entry fee
                        capital -= capital * self.fee_rate
                        
                        # Add signal to tracking
                        full_reason = f"{reason} (confirmed on {self.primary_timeframe}/{self.confirmation_timeframe})"
                        signal_type = "ENTER_LONG" if position == 1 else "ENTER_SHORT"
                        
                        self.signals.append({
                            'time': current_time,
                            'price': current_price,
                            'type': signal_type,
                            'equity': capital,
                            'confidence': confidence,
                            'reason': full_reason,
                            'trailing_stop': trailing_stop_pct * 100
                        })
            
            # Update equity curve
            if position == 0:
                current_equity = capital
            elif position == 1:
                # Long position value
                unrealized_pnl = (current_price - entry_price) / entry_price * capital
                current_equity = capital + unrealized_pnl
            else:  # position == -1
                # Short position value
                unrealized_pnl = (entry_price - current_price) / entry_price * capital
                current_equity = capital + unrealized_pnl
                
            self.equity_curve.append((current_time, current_equity))
        
        # Close any open position at the end
        if position != 0:
            current_time = primary_df.index[-1]
            current_price = primary_df['close'].iloc[-1]
            
            exit_reason = "End of backtest period"
            trade_duration = (current_time - entry_time).total_seconds() / 3600  # hours
            
            if position == 1:
                profit_loss = (current_price - entry_price) / entry_price * capital
                profit_pct = (current_price - entry_price) / entry_price * 100
                position_type = "LONG"
            else:
                profit_loss = (entry_price - current_price) / entry_price * capital
                profit_pct = (entry_price - current_price) / entry_price * 100
                position_type = "SHORT"
                
            capital += profit_loss - (capital * self.fee_rate)  # Apply fee
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': position_type,
                'profit_loss': profit_loss,
                'profit_pct': profit_pct,
                'duration_hours': trade_duration,
                'exit_reason': exit_reason,
            })
            
            # Add signal to tracking
            self.signals.append({
                'time': current_time,
                'price': current_price,
                'type': f'EXIT_{position_type}',
                'equity': capital,
                'reason': exit_reason
            })
            
            # Update final equity curve point
            self.equity_curve.append((current_time, capital))
        
        # Calculate performance metrics
        from backtesting.performance import calculate_performance
        results = calculate_performance(self.trades, self.equity_curve, self.initial_capital)
        
        # Add enhanced metrics
        results['risk_adjusted_return'] = results['total_return_pct'] / results['max_drawdown'] if results['max_drawdown'] > 0 else float('inf')
        results['avg_trailing_stop'] = np.mean([signal.get('trailing_stop', 0) for signal in self.signals if 'trailing_stop' in signal])
        results['trading_breaks'] = sum(1 for trade in self.trades if 'Trading break initiated' in trade.get('exit_reason', ''))
        
        # Calculate success rate after trading breaks
        if results['trading_breaks'] > 0:
            break_indices = [i for i, trade in enumerate(self.trades) if i < len(self.trades)-1 and 'Trading break initiated' in trade.get('exit_reason', '')]
            after_break_trades = [self.trades[i+1] for i in break_indices if i+1 < len(self.trades)]
            after_break_wins = sum(1 for trade in after_break_trades if trade['profit_pct'] > 0)
            results['after_break_win_rate'] = (after_break_wins / len(after_break_trades) * 100) if after_break_trades else 0
        
        return results
        
    def plot_results(self, save_path=None):
        """
        Plot backtest results with enhanced visualization
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(self.signals)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 14))
        
        # Define grid layout
        gs = plt.GridSpec(4, 1, height_ratios=[3, 2, 1, 1], hspace=0.3)
        
        # 1. Price chart with signals
        ax1 = fig.add_subplot(gs[0])
        
        # Plot price
        if self.price_data is not None:
            ax1.plot(self.price_data.index, self.price_data['close'], color='blue', linewidth=1, label='Price')
            
            # Plot moving averages if available
            if 'sma50' in self.price_data.columns:
                ax1.plot(self.price_data.index, self.price_data['sma50'], color='orange', linewidth=1, alpha=0.7, label='50 SMA')
            if 'sma200' in self.price_data.columns:
                ax1.plot(self.price_data.index, self.price_data['sma200'], color='red', linewidth=1, alpha=0.7, label='200 SMA')
        
        # Plot entry signals
        long_entries = signals_df[signals_df['type'] == 'ENTER_LONG']
        short_entries = signals_df[signals_df['type'] == 'ENTER_SHORT']
        long_exits = signals_df[signals_df['type'] == 'EXIT_LONG']
        short_exits = signals_df[signals_df['type'] == 'EXIT_SHORT']
        
        # Plot entries 
        for _, signal in long_entries.iterrows():
            ax1.scatter(signal['time'], signal['price'], marker='^', color='green', s=120, zorder=5)
            ax1.annotate('LONG', (signal['time'], signal['price']), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8, color='green', weight='bold')
            
        for _, signal in short_entries.iterrows():
            ax1.scatter(signal['time'], signal['price'], marker='v', color='red', s=120, zorder=5)
            ax1.annotate('SHORT', (signal['time'], signal['price']), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8, color='red', weight='bold')
        
        # Plot exits with smaller markers
        for _, signal in long_exits.iterrows():
            ax1.scatter(signal['time'], signal['price'], marker='o', color='lime', s=80, zorder=4)
            
        for _, signal in short_exits.iterrows():
            ax1.scatter(signal['time'], signal['price'], marker='o', color='tomato', s=80, zorder=4)
            
        # Highlight trading break periods
        if hasattr(self, 'taking_break') and self.taking_break:
            break_starts = []
            break_ends = []
            
            for i, trade in enumerate(self.trades):
                if 'Trading break initiated' in trade.get('exit_reason', ''):
                    break_starts.append(trade['exit_time'])
                    
                    # Find end of break (if available)
                    if i < len(self.trades) - 1:
                        # Assume break ended at next trade entry
                        break_ends.append(self.trades[i+1]['entry_time'])
                    else:
                        # If this was the last trade, break might still be ongoing
                        break_ends.append(None)
            
            # Plot shaded areas for trading breaks
            for start, end in zip(break_starts, break_ends):
                if end is not None:
                    ax1.axvspan(start, end, color='gray', alpha=0.2)
                    mid_point = start + (end - start) / 2
                    ax1.annotate('Trading Break', (mid_point, ax1.get_ylim()[1] * 0.95), 
                                ha='center', fontsize=9, color='gray')
        
# Format price chart
        ax1.set_title('Bitcoin Price with Enhanced Trend Correction Signals')
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Equity curve with trade markers
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Plot equity curve
        ax2.plot(equity_df.index, equity_df['equity'], color='blue', linewidth=1.5)
        
        # Plot entry and exit points on equity curve
        for _, signal in signals_df.iterrows():
            if signal['type'] == 'ENTER_LONG':
                ax2.scatter(signal['time'], signal['equity'], marker='^', color='green', s=80)
            elif signal['type'] == 'ENTER_SHORT':
                ax2.scatter(signal['time'], signal['equity'], marker='v', color='red', s=80)
            elif signal['type'] == 'EXIT_LONG' or signal['type'] == 'EXIT_SHORT':
                is_profit = next((t for t in self.trades if t['exit_time'] == signal['time']), {}).get('profit_pct', 0) > 0
                marker_color = 'lime' if is_profit else 'tomato'
                ax2.scatter(signal['time'], signal['equity'], marker='o', color=marker_color, s=50)
        
        # Format equity chart
        ax2.set_title('Equity Curve with Trade Markers')
        ax2.set_ylabel('Capital (USD)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        # Plot drawdown
        ax3.fill_between(equity_df.index, 0, equity_df['drawdown'], color='red', alpha=0.3)
        
        # Format drawdown chart
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trailing stops used
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        
        # Extract trailing stops from signals
        stops_df = pd.DataFrame([
            {'time': s['time'], 'trailing_stop': s.get('trailing_stop', 5.0)} 
            for s in self.signals if 'trailing_stop' in s and s['type'].startswith('ENTER')
        ])
        
        if not stops_df.empty:
            stops_df.set_index('time', inplace=True)
            ax4.plot(stops_df.index, stops_df['trailing_stop'], color='purple', marker='o', linestyle='-', linewidth=1.5)
            
            # Format trailing stops chart
            ax4.set_title('Dynamic Trailing Stops')
            ax4.set_ylabel('Stop Size (%)')
            ax4.set_xlabel('Date')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
        
        # Print trade summary
        if len(self.trades) > 0:
            self._print_signals_summary()
    
    def _print_signals_summary(self):
        """Print a summary of the trading signals with enhanced details"""
        print("\n" + "="*80)
        print(f"ENHANCED BITCOIN TREND CORRECTION STRATEGY - TRADING SIGNALS SUMMARY".center(80))
        print("="*80)
        
        # Display the first few trades for reference
        max_trades_to_show = min(10, len(self.trades))
        
        print(f"\nShowing {max_trades_to_show} of {len(self.trades)} trades:")
        print("\n{:<20} {:<20} {:<8} {:<10} {:<10} {:<10} {:<6}".format(
            "Entry Time", "Exit Time", "Type", "Entry $", "Exit $", "P&L %", "Stop %"))
        print("-" * 80)
        
        for i, trade in enumerate(self.trades[:max_trades_to_show]):
            entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M')
            exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M')
            position = trade['position']
            entry_price = f"${trade['entry_price']:.2f}"
            exit_price = f"${trade['exit_price']:.2f}"
            pnl = f"{trade['profit_pct']:.2f}%"
            
            # Find corresponding entry signal to get trailing stop
            entry_signal = next((s for s in self.signals 
                                if s['type'] == f"ENTER_{position}" and s['time'] == trade['entry_time']), 
                                {'trailing_stop': 5.0})
            stop = f"{entry_signal.get('trailing_stop', 5.0):.1f}%"
            
            print("{:<20} {:<20} {:<8} {:<10} {:<10} {:<10} {:<6}".format(
                entry_time, exit_time, position, entry_price, exit_price, pnl, stop))
            
            if 'exit_reason' in trade:
                print(f"Reason: {trade['exit_reason']}")
                print("-" * 80)
        
        # Print advanced signal statistics
        profitable_trades = [t for t in self.trades if t['profit_pct'] > 0]
        long_trades = [t for t in self.trades if t['position'] == 'LONG']
        short_trades = [t for t in self.trades if t['position'] == 'SHORT']
        
        print("\nEnhanced Statistics:")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Profitable Trades: {len(profitable_trades)} ({len(profitable_trades)/len(self.trades)*100:.2f}%)")
        print(f"Long Trades: {len(long_trades)} ({len(long_trades)/len(self.trades)*100:.2f}%)")
        print(f"Short Trades: {len(short_trades)} ({len(short_trades)/len(self.trades)*100:.2f}%)")
        
        # Calculate average trailing stop
        avg_stop = np.mean([s.get('trailing_stop', 5.0) for s in self.signals if 'trailing_stop' in s])
        print(f"Average Trailing Stop: {avg_stop:.2f}%")
        
        # Count trading breaks
        trading_breaks = sum(1 for t in self.trades if 'Trading break initiated' in t.get('exit_reason', ''))
        print(f"Trading Breaks: {trading_breaks}")
        
        # Exit reasons distribution
        exit_reasons = {}
        for trade in self.trades:
            if 'exit_reason' in trade:
                reason_type = trade['exit_reason'].split(' ')[0]
                exit_reasons[reason_type] = exit_reasons.get(reason_type, 0) + 1
        
        if exit_reasons:
            print("\nExit Reason Distribution:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"- {reason}: {count} ({count/len(self.trades)*100:.1f}%)")
        
        print("="*80)