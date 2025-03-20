#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Bitcoin Trend Correction Trading Bot

This module implements an improved trading bot that combines:
1. Dynamic trailing stops based on volatility
2. Enhanced trend detection
3. Better risk management
4. Multiple timeframe confirmation
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
from src.notifier import TelegramNotifier
from src.signal_generator import EnhancedSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bitcoin_trend_bot.enhanced")


class EnhancedTrendBot:
    """Enhanced Bitcoin Trend Correction Trading Bot"""
    
    def __init__(self, exchange_id='binance', primary_timeframe='1h', 
                 confirmation_timeframe='4h', scanning_timeframe='15m',
                 telegram_token=None, telegram_chat_id=None):
        """
        Initialize the Enhanced Bitcoin Trading Bot
        
        Args:
            exchange_id (str): The exchange to use (default: 'binance')
            primary_timeframe (str): Main timeframe for analysis (default: '1h')
            confirmation_timeframe (str): Higher timeframe for trend confirmation (default: '4h')
            scanning_timeframe (str): Lower timeframe for entry timing (default: '15m')
            telegram_token (str): Telegram bot token for notifications
            telegram_chat_id (str): Telegram chat ID for notifications
        """
        # Load environment variables
        load_dotenv()
        
        self.exchange_id = exchange_id
        self.primary_timeframe = primary_timeframe
        self.confirmation_timeframe = confirmation_timeframe
        self.scanning_timeframe = scanning_timeframe
        
        self.telegram_token = telegram_token or os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize components
        self.data_fetcher = DataFetcher(exchange_id)
        self.indicators = TechnicalIndicators()
        self.signal_generator = EnhancedSignalGenerator()
        self.notifier = TelegramNotifier(self.telegram_token, self.telegram_chat_id)
        
        # Position tracking
        self.current_position = 0  # 0: none, 1: long, -1: short
        self.entry_price = 0
        self.entry_time = None
        self.highest_since_entry = 0
        self.lowest_since_entry = float('inf')
        self.trailing_stop_pct = 0.05  # Default, will be dynamically updated
        self.symbol = None
        
        # Risk management
        self.consecutive_losses = 0
        self.taking_break = False
        self.break_start_time = None
        self.max_losses_before_break = 5  # Increase from 3
        self.break_duration_hours = 12    # Decrease from 24
                
        logger.info(f"Enhanced Trading Bot initialized with timeframes: {primary_timeframe}/{confirmation_timeframe}/{scanning_timeframe}")
    
    def run(self, symbol='BTC/USDT', interval=180, export_frequency=12):
        """
        Run the enhanced trading bot
        
        Args:
            symbol (str): Trading pair symbol
            interval (int): Check interval in seconds
            export_frequency (int): How often to export data for LLM (in hours)
        """
        logger.info(f"Starting Enhanced Trading Bot for {symbol}, checking every {interval} seconds")
        
        self.symbol = symbol
        last_export_time = time.time()
        export_interval = export_frequency * 3600  # Convert hours to seconds
        
        # Track last check time for each timeframe to optimize API calls
        last_check = {
            'primary': 0,
            'confirmation': 0,
            'scanning': 0
        }
        
        # Approximate timeframe intervals in seconds
        timeframe_seconds = {
            'primary': self._timeframe_to_seconds(self.primary_timeframe),
            'confirmation': self._timeframe_to_seconds(self.confirmation_timeframe),
            'scanning': self._timeframe_to_seconds(self.scanning_timeframe)
        }
        
        # Main loop
        while True:
            try:
                current_time = time.time()
                
                # Check if we need to end a trading break
                if self.taking_break and self.break_start_time:
                    hours_since_break = (datetime.now() - self.break_start_time).total_seconds() / 3600
                    if hours_since_break >= self.break_duration_hours:
                        self.taking_break = False
                        self.consecutive_losses = 0
                        logger.info(f"Trading break ended after {hours_since_break:.1f} hours")
                        
                        # Send notification
                        self.notifier.send_message("🔄 *Trading Break Ended*\n\nBot is now actively looking for new trade setups.")
                
                # If we're in a trading break, just check exits for existing positions
                if self.taking_break and self.current_position == 0:
                    time.sleep(interval)
                    continue
                
                # Check primary timeframe (1h)
                if current_time - last_check['primary'] >= min(timeframe_seconds['primary'] / 2, 300):
                    primary_df = self.data_fetcher.fetch_ohlcv_data(symbol, self.primary_timeframe, limit=200)
                    
                    if primary_df is not None and not primary_df.empty:
                        # Calculate technical indicators
                        primary_df = self.indicators.calculate_all(primary_df)
                        
                        # Check for exit signals if in a position
                        if self.current_position != 0:
                            exit_signal, exit_reason = self.signal_generator.generate_exit_signals(
                                primary_df, 
                                self.current_position,
                                self.entry_price,
                                self.entry_time,
                                self.highest_since_entry,
                                self.lowest_since_entry,
                                datetime.now()
                            )
                            
                            if exit_signal:
                                current_price = primary_df['close'].iloc[-1]
                                self._exit_position(symbol, current_price, exit_reason)
                        
                        # Process trailing stops
                        if self.current_position != 0:
                            current_price = primary_df['close'].iloc[-1]
                            self._check_trailing_stop(current_price)
                            
                            # Update highest/lowest since entry
                            if self.current_position == 1:  # Long position
                                self.highest_since_entry = max(self.highest_since_entry, current_price)
                            elif self.current_position == -1:  # Short position
                                self.lowest_since_entry = min(self.lowest_since_entry, current_price)
                    
                    last_check['primary'] = current_time
                
                # Only check for new entries if not in a position
                if self.current_position == 0 and not self.taking_break:
                    # Check confirmation timeframe (4h) for trend direction
                    if current_time - last_check['confirmation'] >= min(timeframe_seconds['confirmation'] / 2, 600):
                        confirmation_df = self.data_fetcher.fetch_ohlcv_data(symbol, self.confirmation_timeframe, limit=100)
                        
                        if confirmation_df is not None and not confirmation_df.empty:
                            # Calculate technical indicators
                            confirmation_df = self.indicators.calculate_all(confirmation_df)
                            
                            # Determine primary trend from 4h timeframe
                            uptrend = self.signal_generator._determine_uptrend(confirmation_df)
                            downtrend = self.signal_generator._determine_downtrend(confirmation_df)
                            
                            trend_direction = "UPTREND" if uptrend else ("DOWNTREND" if downtrend else "NEUTRAL")
                            logger.info(f"Current BTC trend ({self.confirmation_timeframe}): {trend_direction} at ${confirmation_df['close'].iloc[-1]:,.2f}")
                        
                        last_check['confirmation'] = current_time
                    
                    # Check scanning timeframe (15m) for entry signals
                    if current_time - last_check['scanning'] >= min(timeframe_seconds['scanning'] / 2, 120):
                        scanning_df = self.data_fetcher.fetch_ohlcv_data(symbol, self.scanning_timeframe, limit=200)
                        
                        if scanning_df is not None and not scanning_df.empty and primary_df is not None and confirmation_df is not None:
                            # Calculate technical indicators
                            scanning_df = self.indicators.calculate_all(scanning_df)
                            
                            # Generate entry signals
                            signal, confidence, reason, trailing_stop = self.signal_generator.generate_entry_signals(scanning_df)
                            
                            # If we have a signal from scanning timeframe, validate with higher timeframes
                            if signal and confidence >= 75:
                                # Get trend direction from confirmation timeframe
                                higher_timeframe_uptrend = self.signal_generator._determine_uptrend(confirmation_df)
                                higher_timeframe_downtrend = self.signal_generator._determine_downtrend(confirmation_df)
                                
                                # Only allow trades in direction of higher timeframe trend
                                aligned_with_trend = (signal == "LONG" and higher_timeframe_uptrend) or (signal == "SHORT" and higher_timeframe_downtrend)
                                
                                if aligned_with_trend:
                                    # Also check primary timeframe for confluence
                                    primary_signal, primary_confidence, _, _ = self.signal_generator.generate_entry_signals(primary_df)
                                    
                                    if primary_signal == signal and primary_confidence >= 60:
                                        # We have alignment across multiple timeframes - enter position
                                        current_price = scanning_df['close'].iloc[-1]
                                        self.trailing_stop_pct = trailing_stop  # Set dynamic trailing stop
                                        
                                        # Enter position
                                        self._enter_position(symbol, signal, current_price, confidence, 
                                                           f"{reason} (confirmed on {self.primary_timeframe}/{self.confirmation_timeframe})")
                        
                        last_check['scanning'] = current_time
                
                # Export data for LLM analysis periodically
                if current_time - last_export_time > export_interval:
                    if primary_df is not None:
                        self._export_data_for_llm(primary_df)
                        last_export_time = current_time
                
                # Sleep until next check
                time.sleep(interval)
            
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
        
        # Update consecutive losses tracking for risk management
        if profit_pct < 0:
            self.consecutive_losses += 1
            
            # Check if we need to take a trading break after multiple losses
            if self.consecutive_losses >= self.max_losses_before_break:
                self.taking_break = True
                self.break_start_time = datetime.now()
                break_message = f"🛑 *Trading Break Initiated* 🛑\n\nAfter {self.consecutive_losses} consecutive losses, the bot will pause new entries for {self.break_duration_hours} hours to avoid market conditions unfavorable to the strategy. Existing positions will still be managed."
                self.notifier.send_message(break_message)
                logger.warning(f"Taking a trading break after {self.consecutive_losses} consecutive losses")
        else:
            # Reset consecutive losses counter on a win
            self.consecutive_losses = 0
        
        # Reset position tracking
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        
        return True
    
    def _check_trailing_stop(self, current_price):
        """Check if trailing stop has been hit"""
        if self.current_position == 0:
            return False
        
        if self.current_position == 1:  # Long position
            # Calculate trailing stop level
            stop_level = self.highest_since_entry * (1 - self.trailing_stop_pct)
            
            if current_price < stop_level:
                reason = f"Trailing stop hit at ${stop_level:.2f} ({self.trailing_stop_pct*100:.1f}% from high of ${self.highest_since_entry:.2f})"
                self._exit_position(self.symbol, current_price, reason)
                return True
                
        elif self.current_position == -1:  # Short position
            # Calculate trailing stop level
            stop_level = self.lowest_since_entry * (1 + self.trailing_stop_pct)
            
            if current_price > stop_level:
                reason = f"Trailing stop hit at ${stop_level:.2f} ({self.trailing_stop_pct*100:.1f}% from low of ${self.lowest_since_entry:.2f})"
                self._exit_position(self.symbol, current_price, reason)
                return True
        
        return False
    
    def _generate_entry_message(self, symbol, signal, price, confidence, reason):
        """Generate Telegram message for position entry"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🚨 *BITCOIN ENTRY SIGNAL - {signal}* 🚨\n\n" \
                 f"*Symbol:* {symbol}\n" \
                 f"*Action:* Enter {signal} position\n" \
                 f"*Price:* ${price:,.2f}\n" \
                 f"*Time:* {current_time}\n" \
                 f"*Confidence:* {confidence}%\n" \
                 f"*Trailing Stop:* {self.trailing_stop_pct*100:.1f}%\n" \
                 f"*Timeframes:* {self.scanning_timeframe} → {self.primary_timeframe} → {self.confirmation_timeframe}\n\n" \
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
                 f"*Timeframe:* {self.primary_timeframe}\n\n" \
                 f"*Exit Reason:*\n{reason}\n\n" \
                 f"This is an automated alert. Please review the market before making trading decisions."
        
        return message
    
    def _export_data_for_llm(self, df, filename=None):
        """
        Export recent data to CSV for LLM analysis
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to the exported file
        """
        if filename is None:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/btc_data_{timestamp}.csv"
            
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
        
        # Export to CSV
        df.to_csv(filename)
        logger.info(f"Exported data to {filename} for LLM analysis")
        
        return filename