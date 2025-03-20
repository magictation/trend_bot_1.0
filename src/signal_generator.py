#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Signal Generator for Bitcoin Trend Correction Trading Bot.

This module improves upon the original signal generator by:
1. Adding dynamic trailing stops based on volatility
2. Implementing better trend filtering
3. Adding volume profile analysis
4. Incorporating early exit signals
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("bitcoin_trend_bot.enhanced_signal_generator")


class EnhancedSignalGenerator:
    """Enhanced class for generating trading signals based on trend correction patterns"""
    
    def __init__(self):
        """Initialize the enhanced signal generator"""
        # Configuration settings
        self.min_confidence_threshold = 60  # Lower further
        self.max_counter_trend_confidence = 90  # Increase further
        self.atr_multiplier = 8.0           # Increase for wider trailing stops
        self.min_trailing_stop_pct = 0.08    # Keep this value
        self.max_trailing_stop_pct = 0.18    # Increase from 0.15
                
        # Strategy state
        self.last_signal = None
        self.last_signal_time = None
    
    def generate_entry_signals(self, df):
        """
        Generate entry trading signals based on enhanced trend correction patterns
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            
        Returns:
            tuple: (signal, confidence, reason, trailing_stop_pct)
        """
        if df is None or df.empty or len(df) < 50:
            return None, 0, "Insufficient data", 0
        
        # Get recent candles for analysis
        current = df.iloc[-1]
        previous = df.iloc[-2]
        prev_5 = df.iloc[-5:]
        
        # Initialize variables
        signal = None
        confidence = 0
        reasons = []
        
        # Identify primary trend direction (more robust method)
        uptrend = self._determine_uptrend(df)
        downtrend = self._determine_downtrend(df)
        
        # ---- ENTRY SIGNALS ----
        
        # 1. Trend Correction Pattern: Price pulls back to key moving average in established trend
        if uptrend and abs(current['close'] - current['sma50']) / current['sma50'] < 0.01:
            # Price touching 50 SMA in uptrend (potential LONG opportunity)
            signal = "LONG"
            confidence += 30
            reasons.append("Price pulled back to 50 SMA in uptrend")
            
        elif downtrend and abs(current['close'] - current['sma50']) / current['sma50'] < 0.01:
            # Price touching 50 SMA in downtrend (potential SHORT opportunity)
            signal = "SHORT"
            confidence += 30
            reasons.append("Price pulled back to 50 SMA in downtrend")
        
        # 2. RSI Divergence detection (enhanced correction signal)
        if uptrend and current['rsi'] < 40:
            # Oversold in uptrend
            if signal is None or signal == "LONG":
                signal = "LONG"
                confidence += 20
                reasons.append(f"Oversold RSI ({current['rsi']:.2f}) in uptrend")
            
        elif downtrend and current['rsi'] > 60:
            # Overbought in downtrend
            if signal is None or signal == "SHORT":
                signal = "SHORT"
                confidence += 20
                reasons.append(f"Overbought RSI ({current['rsi']:.2f}) in downtrend")
        
        # 3. Swing Point Confirmation
        # Recent swing low formed in uptrend
        if signal == "LONG" and prev_5['swing_low'].any():
            confidence += 15
            reasons.append("Recent swing low confirmed")
            
        # Recent swing high formed in downtrend
        elif signal == "SHORT" and prev_5['swing_high'].any():
            confidence += 15
            reasons.append("Recent swing high confirmed")
        
        # 4. Volume confirmation of correction end
        if current['volume_ratio'] > 1.3:
            if signal:
                confidence += 15
                reasons.append(f"High volume confirmation ({current['volume_ratio']:.2f}x average)")
        
        # 5. MACD confirmation of trend resumption
        if signal == "LONG" and current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
            confidence += 15
            reasons.append("MACD bullish crossover signaling trend resumption")
            
        elif signal == "SHORT" and current['macd'] < current['macd_signal'] and previous['macd'] >= previous['macd_signal']:
            confidence += 15
            reasons.append("MACD bearish crossover signaling trend resumption")
        
        # 6. Check for price rejection from key levels
        # Price rejection from below (in downtrend)
        if signal == "SHORT" and (current['high'] > current['sma20'] and current['close'] < current['sma20']):
            confidence += 15
            reasons.append("Price rejection from 20 SMA resistance")
            
        # Price rejection from above (in uptrend)
        elif signal == "LONG" and (current['low'] < current['sma20'] and current['close'] > current['sma20']):
            confidence += 15
            reasons.append("Price rejection from 20 SMA support")
            
        # 7. Enhanced trend strength filter
        if signal and 'adx' in current:
            if current['adx'] > 30:  # Strong trend
                confidence += 10
                reasons.append(f"Strong trend confirmed (ADX: {current['adx']:.2f})")
            elif current['adx'] < 20:  # Weak trend - reduce confidence
                confidence -= 15
                reasons.append(f"Weak trend detected (ADX: {current['adx']:.2f})")
        
        # 8. Bollinger Band Squeeze (volatility contraction before expansion)
        if 'band_width' in current and previous['band_width'] < current['band_width'] and np.mean(df['band_width'].iloc[-5:-1]) < 0.03:
            if signal is not None:
                confidence += 10
                reasons.append("Volatility expansion after contraction")
        
        # 9. Recent failed signals filter (reduce confidence)
        if self.last_signal == signal and self.last_signal_time is not None and abs(df.index[-1].timestamp() - self.last_signal_time) < 86400:
            confidence -= 20
            reasons.append("Recent similar signal failed, reducing confidence")
        
        # 10. IMPORTANT: Counter-trend protection
        is_counter_trend = (signal == "LONG" and not uptrend) or (signal == "SHORT" and not downtrend)
        
        if is_counter_trend:
            # Cap confidence for counter-trend trades
            confidence = min(confidence, self.max_counter_trend_confidence)
            reasons.append("Counter-trend trade (confidence capped)")
        
        # Calculate dynamic trailing stop based on ATR
        trailing_stop_pct = self._calculate_trailing_stop(df)
        
        # Check for confluence of multiple indicators
        reason_text = ", ".join(reasons) if reasons else "No clear pattern"
        
        # Only return signals with sufficient confidence
        if signal and confidence >= self.min_confidence_threshold:
            return signal, min(confidence, 100), reason_text, trailing_stop_pct
        else:
            return None, confidence, reason_text, trailing_stop_pct
    
    def generate_exit_signals(self, df, current_position, entry_price, entry_time, highest_price, lowest_price, current_time):
        """Generate simplified exit signals with fewer, more reliable conditions"""
        if current_position == 0 or df is None or df.empty:
            return False, ""
        
        # Get the most recent data points
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Initialize
        exit_signal = False
        reason = ""
        
        current_price = current['close']
        
        # Calculate profit percentage
        profit_pct = ((current_price - entry_price) / entry_price * 100) if current_position == 1 else ((entry_price - current_price) / entry_price * 100)
        
        # 1. RSI extremes as exit signal (reliable)
        if current_position == 1 and current['rsi'] > 80 and current['rsi'] < previous['rsi']:
            exit_signal = True
            reason = f"RSI reversal from overbought levels (RSI: {current['rsi']:.2f})"
        elif current_position == -1 and current['rsi'] < 20 and current['rsi'] > previous['rsi']:
            exit_signal = True
            reason = f"RSI reversal from oversold levels (RSI: {current['rsi']:.2f})"
        
        # 2. Large profit taking exit (let winners run but secure major profits)
        if not exit_signal and profit_pct > 20:
            exit_signal = True
            reason = f"Taking profit at {profit_pct:.2f}% gain"
        
        # 3. MACD divergence exit
        if not exit_signal:
            # Check last 5 candles for divergence
            if current_position == 1 and 'macd' in current:
                # Bearish divergence: higher highs in price, lower highs in MACD
                price_making_higher_high = current['high'] > max(df['high'].iloc[-6:-1])
                macd_making_lower_high = current['macd'] < max(df['macd'].iloc[-6:-1])
                
                if price_making_higher_high and macd_making_lower_high and current['macd'] > 0:
                    exit_signal = True
                    reason = "Bearish MACD divergence detected"
                    
            elif current_position == -1 and 'macd' in current:
                # Bullish divergence: lower lows in price, higher lows in MACD
                price_making_lower_low = current['low'] < min(df['low'].iloc[-6:-1])
                macd_making_higher_low = current['macd'] > min(df['macd'].iloc[-6:-1])
                
                if price_making_lower_low and macd_making_higher_low and current['macd'] < 0:
                    exit_signal = True
                    reason = "Bullish MACD divergence detected"
        
        # 4. Time-based exit only for losing trades after extended period
        if not exit_signal and entry_time:
            hours_in_trade = (current_time - entry_time).total_seconds() / 3600
            
            if hours_in_trade > 240 and profit_pct < 0:  # 10 days with negative returns
                exit_signal = True
                reason = f"Time-based exit after {hours_in_trade:.1f} hours with negative return ({profit_pct:.2f}%)"
        
        return exit_signal, reason
    
    def _determine_uptrend(self, df):
        """
        Enhanced method to determine if we're in an uptrend
        """
        # Need at least the last 50 candles
        if len(df) < 50:
            return False
        
        recent = df.iloc[-1]
        
        # More robust trend determination
        # 1. Moving average alignment
        ma_alignment = recent['sma20'] > recent['sma50'] > recent['sma100']
        
        # 2. Price above major moving averages
        price_above_ma = recent['close'] > recent['sma50']
        
        # 3. Check higher highs and higher lows pattern
        last_20 = df.iloc[-20:]
        highs_increasing = last_20['high'].iloc[-5:].max() > last_20['high'].iloc[:10].max()
        lows_increasing = last_20['low'].iloc[-5:].min() > last_20['low'].iloc[:10].min()
        
        # 4. MACD histogram increasing
        macd_trending_up = recent['macd_hist'] > 0 and recent['macd_hist'] > df['macd_hist'].iloc[-3:].mean()
        
        # 5. RSI showing strength
        rsi_strength = recent['rsi'] > 50
        
        # Calculate uptrend score
        uptrend_score = sum([
            ma_alignment * 2,  # Most important
            price_above_ma * 2,
            highs_increasing,
            lows_increasing,
            macd_trending_up,
            rsi_strength
        ])
        
        # Need at least 3 points to confirm uptrend (CHANGED FROM 4)
        return uptrend_score >= 3
    
    def _determine_downtrend(self, df):
        """
        Enhanced method to determine if we're in a downtrend
        """
        # Need at least the last 50 candles
        if len(df) < 50:
            return False
        
        recent = df.iloc[-1]
        
        # More robust trend determination
        # 1. Moving average alignment
        ma_alignment = recent['sma20'] < recent['sma50'] < recent['sma100']
        
        # 2. Price below major moving averages
        price_below_ma = recent['close'] < recent['sma50']
        
        # 3. Check lower highs and lower lows pattern
        last_20 = df.iloc[-20:]
        highs_decreasing = last_20['high'].iloc[-5:].max() < last_20['high'].iloc[:10].max()
        lows_decreasing = last_20['low'].iloc[-5:].min() < last_20['low'].iloc[:10].min()
        
        # 4. MACD histogram decreasing
        macd_trending_down = recent['macd_hist'] < 0 and recent['macd_hist'] < df['macd_hist'].iloc[-3:].mean()
        
        # 5. RSI showing weakness
        rsi_weakness = recent['rsi'] < 50
        
        # Calculate downtrend score
        downtrend_score = sum([
            ma_alignment * 2,  # Most important
            price_below_ma * 2,
            highs_decreasing,
            lows_decreasing,
            macd_trending_down,
            rsi_weakness
        ])
        
        # Need at least 3 points to confirm downtrend (CHANGED FROM 4)
        return downtrend_score >= 3
    
    def _calculate_trailing_stop(self, df):
        """
        Calculate dynamic trailing stop percentage based on market volatility (ATR)
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            
        Returns:
            float: Trailing stop percentage
        """
        if 'atr' not in df.columns or df['atr'].iloc[-1] is None:
            return 0.05  # Default if ATR not available
        
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        # ATR as percentage of price
        atr_pct = current_atr / current_price
        
        # Calculate trailing stop as multiple of ATR
        trailing_stop = atr_pct * self.atr_multiplier
        
        # Clamp between min and max values
        return max(self.min_trailing_stop_pct, min(trailing_stop, self.max_trailing_stop_pct))