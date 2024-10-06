import numpy as np
import pandas as pd
from typing import List, Dict
import ta
from scipy import stats

class TechnicalIndicator:
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        return ta.momentum.RSIIndicator(data, window=window).rsi()

    @staticmethod
    def calculate_macd(data: pd.Series) -> pd.Series:
        macd = ta.trend.MACD(data)
        return macd.macd_diff()

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, window_dev: int = 2) -> pd.Series:
        bb = ta.volatility.BollingerBands(data, window=window, window_dev=window_dev)
        return (data - bb.bollinger_mavg()) / (bb.bollinger_hband() - bb.bollinger_lband())

class PositionSizing:
    @staticmethod
    def kelly_criterion(trades: List[Dict], balance: float) -> float:
        if len(trades) < 2:
            return 0.02  # Default to 2% if not enough trades

        wins = [t for t in trades if t['type'] == 'sell' and t['price'] > t['price']]
        losses = [t for t in trades if t['type'] == 'sell' and t['price'] <= t['price']]

        if not losses:
            return 0.02  # Default to 2% if no losses

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['price'] / t['price'] - 1 for t in wins]) if wins else 0
        avg_loss = np.mean([1 - t['price'] / t['price'] for t in losses])

        if avg_loss == 0:
            return 0.02  # Default to 2% if avg_loss is zero

        kelly_fraction = (win_rate / avg_loss) - ((1 - win_rate) / avg_loss)
        return max(0, min(kelly_fraction, 0.2))  # Limit to 0-20%

class PerformanceMetrics:
    @staticmethod
    def calculate_sharpe_ratio(trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        if len(trades) < 2:
            return 0

        returns = [(t['price'] / t['price'] - 1) for t in trades if t['type'] == 'sell']
        if not returns:
            return 0

        excess_returns = np.array(returns) - (risk_free_rate / 252)  # Assuming daily returns
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    @staticmethod
    def calculate_max_drawdown(initial_balance: float, trades: List[Dict]) -> float:
        peak = initial_balance
        max_drawdown = 0

        for trade in trades:
            if trade['type'] == 'sell':
                current_value = trade['price'] * trade['shares']
                if current_value > peak:
                    peak = current_value
                drawdown = (peak - current_value) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    @staticmethod
    def calculate_win_loss_ratio(trades: List[Dict]) -> float:
        wins = sum(1 for t in trades if t['type'] == 'sell' and t['price'] > t['price'])
        losses = sum(1 for t in trades if t['type'] == 'sell' and t['price'] <= t['price'])
        return wins / losses if losses > 0 else float('inf')

    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        gross_profit = sum(t['price'] * t['shares'] - t['price'] * t['shares'] for t in trades if t['type'] == 'sell' and t['price'] > t['price'])
        gross_loss = sum(t['price'] * t['shares'] - t['price'] * t['shares'] for t in trades if t['type'] == 'sell' and t['price'] <= t['price'])
        return gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

class BacktestTools:
    @staticmethod
    def run_monte_carlo(env, strategy, num_simulations: int = 1000):
        results = []
        for _ in range(num_simulations):
            env.reset()
            while env.current_step < len(env.data) - 1:
                state = env.get_state()
                action = strategy(state)
                _, _, done, _ = env.step(action)
                if done:
                    break
            results.append(env.balance)

        mean_result = np.mean(results)
        std_result = np.std(results)
        confidence_interval = stats.t.interval(0.95, len(results)-1, loc=mean_result, scale=std_result/np.sqrt(len(results)))

        return {
            "mean_balance": mean_result,
            "confidence_interval": confidence_interval
        }