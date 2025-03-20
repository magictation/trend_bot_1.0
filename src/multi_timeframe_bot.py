#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Timeframe Bitcoin Trend Correction Trading Bot

This module implements an enhanced trading bot that uses different timeframes
for entry (5m) and exit (1h) decisions, with special focus on volume patterns.
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from src.signal_generator import SignalGenerator
from src.notifier import TelegramNotifier
from src.trading_bot import BitcoinTrendBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_timeframe_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_timeframe_bot")


class MultiTimeframeTrendBot(BitcoinTrendBot):
    """Enhanced Bitcoin Trading Bot using multiple timeframes"""
    
    def __init__(self, exchange_id='binance', entry_timeframe='5m', exit_timeframe='1h', 
                 telegram_token=None, telegram_chat_id=None):
        """
        Initialize the Multi-Timeframe Trading Bot
        
        Args:
            exchange_id (str): The exchange to use (default: 'binance')
            entry_timeframe (str): Timeframe for entry signals (default: '5m')
            exit_timeframe (str): Timeframe for exit signals (default: '1h')
            telegram_token (str): Telegram bot token for notifications
            telegram_chat_id (str): Telegram chat ID for notifications
        """
        # Initialize parent class with exit timeframe (used for main analysis)
        super().__init__(exchange_id, exit_timeframe, telegram_token, telegram_chat_id)
        
        # Add entry-specific timeframe
        self.entry_timeframe = entry_timeframe
        
        # Additional state tracking
        self.current_position = 0  # 0: none, 1: long, -1: short
        self.entry_price = 0
        self.entry_time = None
        self.position_size = 0
        
        # Track volume baseline for spike detection
        self.volume_baseline = None
        self.volume_baseline_periods = 20
        
        # Advanced trailing stop settings
        self.trailing_stop_pct = 0.05  # 5% trailing stop
        self.highest_since_entry = 0
        self.lowest_since_entry = float('inf')
        
        logger.info(f"Multi-Timeframe Bot initialized with entry: {entry_timeframe}, exit: {exit_timeframe}")
    
    def run(self, symbol='BTC/USDT', check_interval=60, export_frequency=24):
        """
        Run the trading bot
        
        Args:
            symbol (str): Trading pair symbol
            check_interval (int): Check interval in seconds
            export_frequency (int): How often to export data for LLM (in hours)
        """
        logger.info(f"Starting Multi-Timeframe Bot for {symbol}, checking every {check_interval} seconds")
        
        last_export_time = time.time()
        export_interval = export_frequency * 3600  # Convert hours to seconds
        
        # Track last check time for each timeframe to optimize API calls
        last_entry_check = 0
        last_exit_check = 0
        
        # Entry timeframe interval in seconds
        entry_interval = self._timeframe_to_seconds(self.entry_timeframe)
        # Exit timeframe interval in seconds
        exit_interval = self._timeframe_to_seconds(self.timeframe)
        
        while True:
            try:
                current_time = time.time()
                current_checks = []
                
                # Check entry signals (5m) if not in a position or if sufficient time has passed
                if self.current_position == 0 and (current_time - last_entry_check >= entry_interval):
                    entry_df = self.data_fetcher.fetch_ohlcv_data(symbol, self.entry_timeframe, limit=100)
                    
                    if entry_df is not None and not entry_df.empty:
                        # Process entry timeframe data
                        entry_data = self._process_entry_timeframe(entry_df)
                        
                        # Check for entry signals
                        entry_signal, entry_confidence, entry_reason = self._check_entry_signals(entry_data)
                        
                        if entry_signal and entry_confidence >= 75:
                            # Enter position
                            self._enter_position(symbol, entry_signal, entry_data['close'].iloc[-1], entry_confidence, entry_reason)
                    
                    last_entry_check = current_time
                    current_checks.append("entry")
                
                # Check exit signals (1h) if in a position
                if self.current_position != 0 and (current_time - last_exit_check >= min(exit_interval, 300)):  # Check at least every 5 minutes
                    exit_df = self.data_fetcher.fetch_ohlcv_data(symbol, self.timeframe, limit=100)
                    
                    if exit_df is not None and not exit_df.empty:
                        # Process exit timeframe data
                        exit_data = self.indicators.calculate_all(exit_df)
                        
                        # Update trailing stop values
                        current_price = exit_data['close'].iloc[-1]
                        if self.current_position == 1:  # Long position
                            self.highest_since_entry = max(self.highest_since_entry, current_price)
                        elif self.current_position == -1:  # Short position
                            self.lowest_since_entry = min(self.lowest_since_entry, current_price)
                        
                        # Check for exit signals
                        exit_signal, exit_reason = self._check_exit_signals(exit_data, current_price)
                        
                        if exit_signal:
                            # Exit position
                            self._exit_position(symbol, current_price, exit_reason)
                    
                    last_exit_check = current_time
                    current_checks.append("exit")
                
                # Log activity
                if current_checks:
                    logger.debug(f"Checked {', '.join(current_checks)} signals for {symbol}")
                
                # Export data for LLM analysis periodically
                if self.current_position != 0 and (current_time - last_export_time > export_interval):
                    if exit_df is not None:
                        self.export_data_for_llm(exit_df)
                        last_export_time = current_time
                
                # Sleep until next check
                time.sleep(check_interval)
            
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Sleep for a minute before retrying
    
    def _timeframe_to_seconds(self, timeframe):
        """Convert timeframe string to seconds"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            return 300  # Default to 5 minutes
    
    def _process_entry_timeframe(self, df):
        """
        Process the entry timeframe data with specialized indicators
        focused on volume and short-term price action
        """
        # Make a copy to avoid modifying original
        entry_df = df.copy()
        
        # Calculate basic indicators
        entry_df = self.indicators.calculate_all(entry_df)
        
        # Add specialized entry indicators
        # 1. Volume indicators for spike detection
        entry_df['volume_sma20'] = entry_df['volume'].rolling(window=20).mean()
        entry_df['volume_ratio'] = entry_df['volume'] / entry_df['volume_sma20']
        
        # 2. Rate of change (momentum)
        entry_df['price_roc_10'] = entry_df['close'].pct_change(periods=10) * 100
        
        # 3. Volume-weighted price momentum
        entry_df['vol_price_momentum'] = entry_df['price_roc_10'] * entry_df['volume_ratio']
        
        # 4. Enhanced volatility measure
        entry_df['range_pct'] = (entry_df['high'] - entry_df['low']) / entry_df['low'] * 100
        entry_df['range_sma10'] = entry_df['range_pct'].rolling(window=10).mean()
        entry_df['volatility_ratio'] = entry_df['range_pct'] / entry_df['range_sma10']
        
        # 5. Volume Delta (difference between up and down volume)
        up_volume = entry_df['volume'] * (entry_df['close'] > entry_df['open']).astype(int)
        down_volume = entry_df['volume'] * (entry_df['close'] <= entry_df['open']).astype(int)
        entry_df['volume_delta'] = (up_volume - down_volume) / (up_volume + down_volume)
        
        # Update volume baseline
        self.volume_baseline = entry_df['volume'].rolling(window=self.volume_baseline_periods).mean().iloc[-1]
        
        return entry_df
    
    def _check_entry_signals(self, df):
        """
        Check for entry signals based on volume patterns and price action
        on the shorter timeframe (5m)
        
        Returns:
            tuple: (signal, confidence, reason)
        """
        # Get the most recent data points
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Initialize variables
        signal = None
        confidence = 0
        reasons = []
        
        # Check primary trend direction from 1h timeframe (stored property)
        uptrend = True if getattr(current, 'trend', 0) > 0 else False  # Default to trend from current frame if not set
        
        # 1. Volume Spike Detection - key entry signal
        if current['volume_ratio'] > 2.0:
            confidence += 25
            reasons.append(f"Major volume spike ({current['volume_ratio']:.2f}x average)")
        elif current['volume_ratio'] > 1.5:
            confidence += 15
            reasons.append(f"Volume spike ({current['volume_ratio']:.2f}x average)")
        
        # 2. Price-Volume Relationship
        if current['volume_delta'] > 0.6 and current['close'] > current['open']:
            # Strong buying pressure
            signal = "LONG"
            confidence += 20
            reasons.append(f"Strong buying pressure (delta: {current['volume_delta']:.2f})")
        elif current['volume_delta'] < -0.6 and current['close'] < current['open']:
            # Strong selling pressure
            signal = "SHORT"
            confidence += 20
            reasons.append(f"Strong selling pressure (delta: {current['volume_delta']:.2f})")
        
        # 3. Moving Average Interactions (more sensitive on short timeframe)
        # Bullish MA cross or bounce
        if (previous['close'] < previous['sma20'] and current['close'] > current['sma20']) or \
           (current['low'] <= current['sma20'] * 1.005 and current['close'] > current['sma20'] * 1.01 and current['close'] > current['open']):
            if not signal or signal == "LONG":
                signal = "LONG"
                confidence += 15
                reasons.append("Bullish MA interaction")
        # Bearish MA cross or bounce
        elif (previous['close'] > previous['sma20'] and current['close'] < previous['sma20']) or \
             (current['high'] >= current['sma20'] * 0.995 and current['close'] < current['sma20'] * 0.99 and current['close'] < current['open']):
            if not signal or signal == "SHORT":
                signal = "SHORT"
                confidence += 15
                reasons.append("Bearish MA interaction")
        
        # 4. RSI conditions (more extreme on short timeframe)
        if current['rsi'] < 30 and current['rsi'] > previous['rsi']:
            # Oversold with potential reversal
            signal = "LONG"
            confidence += 15
            reasons.append(f"Oversold reversal (RSI: {current['rsi']:.2f})")
        elif current['rsi'] > 70 and current['rsi'] < previous['rsi']:
            # Overbought with potential reversal
            signal = "SHORT"
            confidence += 15
            reasons.append(f"Overbought reversal (RSI: {current['rsi']:.2f})")
        
        # 5. Volatility expansion with direction
        if current['volatility_ratio'] > 1.5:
            if current['close'] > current['open'] and (not signal or signal == "LONG"):
                signal = "LONG"
                confidence += 10
                reasons.append(f"Volatility expansion with bullish bias ({current['volatility_ratio']:.2f}x)")
            elif current['close'] < current['open'] and (not signal or signal == "SHORT"):
                signal = "SHORT"
                confidence += 10
                reasons.append(f"Volatility expansion with bearish bias ({current['volatility_ratio']:.2f}x)")
        
        # 6. Volume-weighted momentum
        if current['vol_price_momentum'] > 10 and (not signal or signal == "LONG"):
            signal = "LONG"
            confidence += 15
            reasons.append(f"Strong volume-supported momentum ({current['vol_price_momentum']:.2f})")
        elif current['vol_price_momentum'] < -10 and (not signal or signal == "SHORT"):
            signal = "SHORT"
            confidence += 15
            reasons.append(f"Strong volume-supported downward momentum ({current['vol_price_momentum']:.2f})")
        
        # 7. Ensure signal aligns with exit timeframe trend - reduce confidence if counter-trend
        if self.current_position == 0:  # Only check if not in a position
            try:
                # Get the most recent 1h data
                exit_tf_data = self.data_fetcher.fetch_ohlcv_data(self.symbol, self.timeframe, limit=100)
                if exit_tf_data is not None and not exit_tf_data.empty:
                    exit_data = self.indicators.calculate_all(exit_tf_data)
                    
                    # Determine trend
                    exit_uptrend = exit_data['sma50'].iloc[-1] > exit_data['sma100'].iloc[-1]
                    
                    # If taking a counter-trend trade, reduce confidence
                    if (signal == "LONG" and not exit_uptrend) or (signal == "SHORT" and exit_uptrend):
                        confidence -= 20
                        reasons.append("Counter-trend to 1h timeframe (-20 confidence)")
                    else:
                        confidence += 10
                        reasons.append("Aligned with 1h trend direction (+10 confidence)")
            except Exception as e:
                logger.warning(f"Error checking exit timeframe trend: {e}")
        
        # Compile final reason text
        reason_text = ", ".join(reasons) if reasons else "No clear pattern"
        
        return signal, min(confidence, 100), reason_text
    
    def _check_exit_signals(self, df, current_price):
        """
        Check for exit signals based on the longer timeframe (1h)
        
        Returns:
            tuple: (exit_signal, reason)
        """
        if self.current_position == 0:
            return False, ""
        
        # Get recent data
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Initialize
        exit_signal = False
        reason = ""
        
        # 1. Trailing stop check (primary exit method)
        if self.current_position == 1:  # Long position
            # Calculate trailing stop level
            stop_level = self.highest_since_entry * (1 - self.trailing_stop_pct)
            
            if current_price < stop_level:
                exit_signal = True
                reason = f"Trailing stop hit at ${stop_level:.2f} (high: ${self.highest_since_entry:.2f})"
                
        elif self.current_position == -1:  # Short position
            # Calculate trailing stop level
            stop_level = self.lowest_since_entry * (1 + self.trailing_stop_pct)
            
            if current_price > stop_level:
                exit_signal = True
                reason = f"Trailing stop hit at ${stop_level:.2f} (low: ${self.lowest_since_entry:.2f})"
        
        # 2. Moving average crossover (trend change)
        if not exit_signal:
            if self.current_position == 1 and previous['sma20'] > previous['sma50'] and current['sma20'] < current['sma50']:
                exit_signal = True
                reason = "Bearish moving average crossover (20 SMA crossed below 50 SMA)"
            elif self.current_position == -1 and previous['sma20'] < previous['sma50'] and current['sma20'] > current['sma50']:
                exit_signal = True
                reason = "Bullish moving average crossover (20 SMA crossed above 50 SMA)"
        
        # 3. Trend exhaustion via RSI
        if not exit_signal:
            if self.current_position == 1 and current['rsi'] > 80 and current['rsi'] < previous['rsi']:
                exit_signal = True
                reason = f"Potential trend exhaustion (RSI: {current['rsi']:.2f})"
            elif self.current_position == -1 and current['rsi'] < 20 and current['rsi'] > previous['rsi']:
                exit_signal = True
                reason = f"Potential trend exhaustion (RSI: {current['rsi']:.2f})"
        
        # 4. Volume climax
        if not exit_signal and current['volume_ratio'] > 3.0:
            # Extreme volume can signal exhaustion
            if (self.current_position == 1 and current['close'] < current['open']) or \
               (self.current_position == -1 and current['close'] > current['open']):
                exit_signal = True
                reason = f"Volume climax with adverse price action (volume: {current['volume_ratio']:.2f}x average)"
        
        # 5. Extended target achieved
        if not exit_signal:
            # Calculate current profit/loss
            if self.current_position == 1:
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                if profit_pct > 5.0:  # Take profit at 5%
                    exit_signal = True
                    reason = f"Take profit target achieved ({profit_pct:.2f}%)"
            elif self.current_position == -1:
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100
                if profit_pct > 5.0:  # Take profit at 5%
                    exit_signal = True
                    reason = f"Take profit target achieved ({profit_pct:.2f}%)"
        
        # 6. Time-based exit
        if not exit_signal and self.entry_time:
            # Exit if position held for more than 3 days with minimal profit
            position_duration = (datetime.now() - self.entry_time).total_seconds() / 3600  # hours
            
            if position_duration > 72:  # 3 days
                if self.current_position == 1:
                    profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                else:
                    profit_pct = (self.entry_price - current_price) / self.entry_price * 100
                
                if profit_pct < 2.0:  # Less than 2% profit after 3 days
                    exit_signal = True
                    reason = f"Time-based exit after {position_duration:.1f} hours with low profit ({profit_pct:.2f}%)"
        
        return exit_signal, reason
    
    def _enter_position(self, symbol, signal, price, confidence, reason):
        """Enter a new trading position"""
        if self.current_position != 0:
            logger.warning(f"Cannot enter new {signal} position - already in a position")
            return False
        
        self.current_position = 1 if signal == "LONG" else -1
        self.entry_price = price
        self.entry_time = datetime.now()
        
        # Reset trailing stop values
        if self.current_position == 1:
            self.highest_since_entry = price
        else:
            self.lowest_since_entry = price
        
        # Store reference for alerts
        self.symbol = symbol
        
        # Send notification
        message = self._generate_entry_message(symbol, signal, price, confidence, reason)
        self.notifier.send_message(message)
        
        logger.info(f"Entered {signal} position at ${price:.2f} with {confidence}% confidence: {reason}")
        
        return True
    
    def _exit_position(self, symbol, price, reason):
        """Exit current trading position"""
        if self.current_position == 0:
            logger.warning("Cannot exit position - not in a position")
            return False
        
        position_type = "LONG" if self.current_position == 1 else "SHORT"
        
        # Calculate profit/loss
        if self.current_position == 1:
            profit_pct = (price - self.entry_price) / self.entry_price * 100
        else:
            profit_pct = (self.entry_price - price) / self.entry_price * 100
        
        # Calculate duration
        duration_hours = (datetime.now() - self.entry_time).total_seconds() / 3600 if self.entry_time else 0
        
        # Send notification
        message = self._generate_exit_message(symbol, position_type, price, profit_pct, duration_hours, reason)
        self.notifier.send_message(message)
        
        logger.info(f"Exited {position_type} position at ${price:.2f} with {profit_pct:.2f}% P&L: {reason}")
        
        # Reset position tracking
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        
        return True
    
    def _generate_entry_message(self, symbol, signal, price, confidence, reason):
        """Generate Telegram message for position entry"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🚨 *BITCOIN ENTRY SIGNAL - {signal}* 🚨\n\n" \
                 f"*Symbol:* {symbol}\n" \
                 f"*Action:* Enter {signal} position\n" \
                 f"*Price:* ${price:,.2f}\n" \
                 f"*Time:* {current_time}\n" \
                 f"*Confidence:* {confidence}%\n" \
                 f"*Timeframe:* {self.entry_timeframe}\n\n" \
                 f"*Signal Triggers:*\n{reason}\n\n" \
                 f"This is an automated alert. Please review the market before making trading decisions."
        
        return message
    
    def _generate_exit_message(self, symbol, position_type, price, profit_pct, duration_hours, reason):
        """Generate Telegram message for position exit"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine emoji based on profit/loss
        if profit_pct > 0:
            emoji = "✅"
        else:
            emoji = "🔴"
        
        message = f"{emoji} *BITCOIN EXIT SIGNAL* {emoji}\n\n" \
                 f"*Symbol:* {symbol}\n" \
                 f"*Action:* Exit {position_type} position\n" \
                 f"*Entry Price:* ${self.entry_price:,.2f}\n" \
                 f"*Exit Price:* ${price:,.2f}\n" \
                 f"*P&L:* {profit_pct:,.2f}%\n" \
                 f"*Duration:* {duration_hours:.1f} hours\n" \
                 f"*Time:* {current_time}\n" \
                 f"*Timeframe:* {self.timeframe}\n\n" \
                 f"*Exit Reason:*\n{reason}\n\n" \
                 f"This is an automated alert. Please review the market before making trading decisions."
        
        return message


if __name__ == "__main__":
    # This allows the module to be run directly
    bot = MultiTimeframeTrendBot()
    bot.run()