#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical indicators for Bitcoin Trend Correction Trading Bot.

This module calculates various technical indicators used for
identifying trend corrections in Bitcoin price.
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    def calculate_all(self, df):
        """
        Calculate all technical indicators for trend correction detection
        
        Args:
            df (pandas.DataFrame): OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with indicators
        """
        # Copy the dataframe to avoid modifying the original
        df = df.copy()
        
        # Moving averages
        df = self.calculate_moving_averages(df)
        
        # Bollinger Bands
        df = self.calculate_bollinger_bands(df)
        
        # RSI
        df = self.calculate_rsi(df)
        
        # MACD
        df = self.calculate_macd(df)
        
        # ATR
        df = self.calculate_atr(df)
        
        # Volume indicators
        df = self.calculate_volume_indicators(df)
        
        # ADX (trend strength)
        df = self.calculate_adx(df)
        
        # Fibonacci retracement levels
        df = self.calculate_fibonacci_levels(df)
        
        # Swing highs and lows
        df = self.identify_swing_points(df)
        
        # Trend direction
        df['trend'] = np.where(df['sma50'] > df['sma100'], 1, -1)
        
        # Distance from moving averages (for mean reversion)
        df['dist_from_sma50'] = (df['close'] - df['sma50']) / df['sma50'] * 100
        df['dist_from_sma200'] = (df['close'] - df['sma200']) / df['sma200'] * 100
        
        return df
        
    def calculate_moving_averages(self, df):
        """
        Calculate various moving averages
        
        Args:
            df (pandas.DataFrame): OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with moving averages
        """
        # Simple Moving Averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma100'] = df['close'].rolling(window=100).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        return df
    
    def calculate_bollinger_bands(self, df):
        """
        Calculate Bollinger Bands
        
        Args:
            df (pandas.DataFrame): OHLCV data with SMA20
            
        Returns:
            pandas.DataFrame: DataFrame with Bollinger Bands
        """
        # Standard deviation for Bollinger Bands
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        df['band_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
        
        return df
    
    def calculate_rsi(self, df, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df (pandas.DataFrame): OHLCV data
            period (int): RSI period
            
        Returns:
            pandas.DataFrame: DataFrame with RSI
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_macd(self, df):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            df (pandas.DataFrame): OHLCV data with EMA12 and EMA26
            
        Returns:
            pandas.DataFrame: DataFrame with MACD
        """
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            df (pandas.DataFrame): OHLCV data
            period (int): ATR period
            
        Returns:
            pandas.DataFrame: DataFrame with ATR
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['atr'] = true_range.rolling(period).mean()
        
        return df
    
    def calculate_volume_indicators(self, df):
        """
        Calculate volume-based indicators
        
        Args:
            df (pandas.DataFrame): OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with volume indicators
        """
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma20']
        
        return df
    
    def calculate_adx(self, df, period=14):
        """
        Calculate Average Directional Index (ADX) for trend strength
        
        Args:
            df (pandas.DataFrame): OHLCV data
            period (int): ADX period
            
        Returns:
            pandas.DataFrame: DataFrame with ADX
        """
        # Calculate +DI and -DI
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().multiply(-1)
        
        plus_dm = np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0
        )
        minus_dm = np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0
        )
        
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # Smooth with Wilder's smoothing
        tr_period = true_range.rolling(period).sum()
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / tr_period)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / tr_period)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(period).mean()
        
        return df
    
    def identify_swing_points(self, df, window=5):
        """
        Identify swing highs and lows in price data
        
        Args:
            df (pandas.DataFrame): OHLCV data
            window (int): Window size for comparison
            
        Returns:
            pandas.DataFrame: DataFrame with swing points
        """
        # Swing high: current high is higher than 'window' periods before and after
        df['swing_high'] = df['high'].rolling(window * 2 + 1, center=True).apply(
            lambda x: x[window] == max(x), raw=True
        )
        
        # Swing low: current low is lower than 'window' periods before and after
        df['swing_low'] = df['low'].rolling(window * 2 + 1, center=True).apply(
            lambda x: x[window] == min(x), raw=True
        )
        
        # Fill NaN values
        df['swing_high'] = df['swing_high'].fillna(False)
        df['swing_low'] = df['swing_low'].fillna(False)
        
        return df
    
    def calculate_fibonacci_levels(self, df):
        """
        Calculate Fibonacci retracement levels for recent price swings
        
        Args:
            df (pandas.DataFrame): OHLCV data
            
        Returns:
            pandas.DataFrame: Original data with Fibonacci levels
        """
        # Find recent significant high and low (last 30 candles)
        window = min(30, len(df) - 1)
        recent_df = df.iloc[-window:]
        
        recent_high = recent_df['high'].max()
        recent_low = recent_df['low'].min()
        range_size = recent_high - recent_low
        
        # Calculate Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
        df['fib_236'] = recent_high - 0.236 * range_size
        df['fib_382'] = recent_high - 0.382 * range_size
        df['fib_500'] = recent_high - 0.500 * range_size
        df['fib_618'] = recent_high - 0.618 * range_size
        df['fib_786'] = recent_high - 0.786 * range_size
        
        return df