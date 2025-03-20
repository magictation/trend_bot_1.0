#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance metrics calculator for Bitcoin Trend Correction Trading Bot.

This module calculates various performance metrics for evaluating 
trading strategy performance.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("bitcoin_trend_bot.performance")


def calculate_performance(trades, equity_curve, initial_capital):
    """
    Calculate performance metrics from backtest results
    
    Args:
        trades (list): List of trade dictionaries
        equity_curve (list): List of (timestamp, equity) tuples
        initial_capital (float): Initial capital
        
    Returns:
        dict: Performance metrics
    """
    if not trades:
        logger.warning("No trades were executed in the backtest")
        return {
            'total_return': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_trade_duration': 0,
            'final_capital': initial_capital
        }
    
    # Convert trades to DataFrame for analysis
    trades_df = pd.DataFrame(trades)
    
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
    equity_df.set_index('timestamp', inplace=True)
    
    # Calculate returns
    total_return_pct = (equity_df['equity'].iloc[-1] / initial_capital - 1) * 100
    
    # Win rate
    trades_df['win'] = trades_df['profit_loss'] > 0
    win_rate = trades_df['win'].mean() * 100
    
    # Profit factor (gross profits / gross losses)
    gross_profits = trades_df.loc[trades_df['profit_loss'] > 0, 'profit_loss'].sum()
    gross_losses = abs(trades_df.loc[trades_df['profit_loss'] < 0, 'profit_loss'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Maximum drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_drawdown = abs(equity_df['drawdown'].min())
    
    # Sharpe ratio (annualized)
    equity_df['returns'] = equity_df['equity'].pct_change()
    
    # Fix for the Timedelta error
    try:
        # Calculate time difference in hours
        if len(equity_df.index) > 1:
            time_diff = (equity_df.index[1] - equity_df.index[0]).total_seconds() / 3600
            annualization_factor = np.sqrt(365 * 24 / time_diff)
        else:
            # Default to daily if we can't calculate
            annualization_factor = np.sqrt(365)
        
        sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * annualization_factor
    except (TypeError, ZeroDivisionError):
        # Fallback if there's an error with the time delta calculation
        sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(365)
    
    # Average trade duration (hours)
    avg_trade_duration = trades_df['duration_hours'].mean()
    
    # Monthly returns
    try:
        equity_df['month'] = equity_df.index.to_period('M')
        monthly_returns = equity_df.groupby('month')['equity'].last().pct_change() * 100
    except Exception as e:
        logger.warning(f"Error calculating monthly returns: {e}")
        monthly_returns = pd.Series()
    
    # Position breakdown
    long_trades = trades_df[trades_df['position'] == 'LONG']
    short_trades = trades_df[trades_df['position'] == 'SHORT']
    
    long_win_rate = long_trades['win'].mean() * 100 if len(long_trades) > 0 else 0
    short_win_rate = short_trades['win'].mean() * 100 if len(short_trades) > 0 else 0
    
    # Average winning and losing trade
    avg_win = trades_df.loc[trades_df['win'], 'profit_pct'].mean() if len(trades_df[trades_df['win']]) > 0 else 0
    avg_loss = trades_df.loc[~trades_df['win'], 'profit_pct'].mean() if len(trades_df[~trades_df['win']]) > 0 else 0
    
    # Maximum consecutive wins and losses
    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    
    for win in trades_df['win']:
        if win:
            if current_streak < 0:
                current_streak = 1
            else:
                current_streak += 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_streak > 0:
                current_streak = -1
            else:
                current_streak -= 1
            max_loss_streak = max(max_loss_streak, abs(current_streak))
    
    # Results
    results = {
        'total_return_pct': total_return_pct,
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_trade_duration': avg_trade_duration,
        'final_capital': equity_df['equity'].iloc[-1],
        'monthly_returns': monthly_returns,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'trade_summary': trades_df.describe()
    }
    
    logger.info(f"Backtest completed with {len(trades_df)} trades and {total_return_pct:.2f}% return")
    
    return results


def print_performance_report(results):
    """
    Print a comprehensive performance report
    
    Args:
        results (dict): Performance metrics
    """
    print("\n" + "="*80)
    print(f"BITCOIN TREND CORRECTION STRATEGY BACKTEST RESULTS".center(80))
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
    
    print(f"\nTrade Statistics:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Average Trade Duration: {results['avg_trade_duration']:.2f} hours")
    print(f"Average Win: {results['avg_win']:.2f}%")
    print(f"Average Loss: {results['avg_loss']:.2f}%")
    print(f"Max Win Streak: {results['max_win_streak']}")
    print(f"Max Loss Streak: {results['max_loss_streak']}")
    
    if results['total_trades'] > 0:
        print(f"\nPosition Breakdown:")
        print(f"Long Trades: {results['long_trades']} ({results['long_trades']/results['total_trades']*100:.2f}%)")
        print(f"Short Trades: {results['short_trades']} ({results['short_trades']/results['total_trades']*100:.2f}%)")
        print(f"Long Win Rate: {results['long_win_rate']:.2f}%")
        print(f"Short Win Rate: {results['short_win_rate']:.2f}%")
    
    print("\nMonthly Returns:")
    try:
        for date, ret in results['monthly_returns'].items():
            if pd.notnull(ret):
                print(f"{date}: {ret:.2f}%")
    except Exception as e:
        print("Unable to display monthly returns")
    
    print("="*80)