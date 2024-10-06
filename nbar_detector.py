import numpy as np
import pandas as pd
from typing import List, Tuple, Literal

class NBARReversalDetector:
    def __init__(self, data: pd.DataFrame, 
                 num_bars: int = 7, 
                 min_bars_percent: float = 50, 
                 pattern_type: Literal["All", "Normal", "Enhanced"] = "All",
                 use_volume: bool = False,
                 use_atr: bool = False):
        """
        Initialize the NBARReversalDetector.
        
        :param data: DataFrame containing price data
        :param num_bars: Number of bars to consider for the pattern
        :param min_bars_percent: Minimum percentage of bars that should follow the pattern
        :param pattern_type: Type of patterns to detect ("All", "Normal", or "Enhanced")
        :param use_volume: Whether to incorporate volume in the detection
        :param use_atr: Whether to use Average True Range for confirmation
        """
        self.data = data
        self.num_bars = num_bars
        self.min_bars_percent = min_bars_percent / 100
        self.pattern_type = pattern_type
        self.use_volume = use_volume
        self.use_atr = use_atr

        if self.use_atr:
            self.calculate_atr()

    def calculate_atr(self, period: int = 14) -> None:
        """Calculate Average True Range."""
        high = self.data['Max(TL)']
        low = self.data['Min(TL)']
        close = self.data['Kapanış(TL)']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(window=period).mean()

    def is_reversal(self, window: pd.DataFrame, direction: Literal["Bullish", "Bearish"]) -> Tuple[float, bool]:
        """
        Check if the given window represents a reversal pattern.
        
        :param window: DataFrame window to check for reversal
        :param direction: "Bullish" or "Bearish"
        :return: Tuple of (extreme_price, is_reversal)
        """
        if direction == "Bullish":
            extreme_price = window['Min(TL)'].min()
            count = sum(window['AOF(TL)'] > window['Kapanış(TL)'])
            condition = window['Max(TL)'].iloc[-1] > window['Max(TL)'].iloc[0]
        else:  # Bearish
            extreme_price = window['Max(TL)'].max()
            count = sum(window['AOF(TL)'] < window['Kapanış(TL)'])
            condition = window['Min(TL)'].iloc[-1] < window['Min(TL)'].iloc[0]

        is_reversal = (count / (self.num_bars - 1) >= self.min_bars_percent) and condition

        if self.use_volume:
            volume_increase = window['Hacim(TL)'].iloc[-1] > window['Hacim(TL)'].mean()
            is_reversal = is_reversal and volume_increase

        if self.use_atr and 'ATR' in self.data.columns:
            atr_breakout = abs(window['Kapanış(TL)'].iloc[-1] - window['Kapanış(TL)'].iloc[0]) > window['ATR'].iloc[-1]
            is_reversal = is_reversal and atr_breakout

        return extreme_price, is_reversal

    def detect_reversals(self) -> List[Tuple[str, int, str]]:
        """
        Detect reversal patterns in the data.
        
        :return: List of tuples (reversal_type, index, pattern_type)
        """
        reversals = []
        for i in range(self.num_bars, len(self.data)):
            window = self.data.iloc[i-self.num_bars:i+1]
            
            bull_low, is_bullish = self.is_reversal(window, "Bullish")
            bear_high, is_bearish = self.is_reversal(window, "Bearish")
            
            last_close = window['Kapanış(TL)'].iloc[-1]
            first_high = window['Max(TL)'].iloc[0]
            first_low = window['Min(TL)'].iloc[0]
            
            if is_bullish:
                if self.pattern_type in ["All", "Normal"] and last_close < first_high:
                    reversals.append(("Bullish", i, "Normal"))
                elif self.pattern_type in ["All", "Enhanced"] and last_close > first_high:
                    reversals.append(("Bullish", i, "Enhanced"))
            elif is_bearish:
                if self.pattern_type in ["All", "Normal"] and last_close > first_low:
                    reversals.append(("Bearish", i, "Normal"))
                elif self.pattern_type in ["All", "Enhanced"] and last_close < first_low:
                    reversals.append(("Bearish", i, "Enhanced"))
        
        return reversals

    def get_reversal_strength(self, reversal: Tuple[str, int, str]) -> float:
        """
        Calculate the strength of a detected reversal.
        
        :param reversal: Tuple of (reversal_type, index, pattern_type)
        :return: Strength score (0.0 to 1.0)
        """
        reversal_type, index, pattern_type = reversal
        window = self.data.iloc[index-self.num_bars+1:index+1]
        
        if reversal_type == "Bullish":
            price_change = (window['Kapanış(TL)'].iloc[-1] - window['Min(TL)'].min()) / window['Min(TL)'].min()
        else:  # Bearish
            price_change = (window['Max(TL)'].max() - window['Kapanış(TL)'].iloc[-1]) / window['Max(TL)'].max()
        
        volume_factor = 1.0
        if self.use_volume:
            volume_change = window['Hacim(TL)'].iloc[-1] / window['Hacim(TL)'].mean()
            volume_factor = min(volume_change, 2.0)  # Cap at 2.0
        
        atr_factor = 1.0
        if self.use_atr and 'ATR' in self.data.columns:
            atr_multiple = abs(window['Kapanış(TL)'].iloc[-1] - window['Kapanış(TL)'].iloc[0]) / window['ATR'].iloc[-1]
            atr_factor = min(atr_multiple / 2, 2.0)  # Normalize and cap at 2.0
        
        strength = price_change * volume_factor * atr_factor
        return min(strength, 1.0)  # Cap at 1.0

    def get_reversals_with_strength(self) -> List[Tuple[str, int, str, float]]:
        """
        Detect reversals and calculate their strengths.
        
        :return: List of tuples (reversal_type, index, pattern_type, strength)
        """
        reversals = self.detect_reversals()
        return [(r[0], r[1], r[2], self.get_reversal_strength(r)) for r in reversals]

    def plot_reversals(self, start_date=None, end_date=None):
        """
        Plot the price chart with detected reversals.
        
        :param start_date: Start date for plotting (optional)
        :param end_date: End date for plotting (optional)
        """
        import matplotlib.pyplot as plt
        
        data_to_plot = self.data
        if start_date:
            data_to_plot = data_to_plot[data_to_plot.index >= start_date]
        if end_date:
            data_to_plot = data_to_plot[data_to_plot.index <= end_date]
        
        reversals = self.get_reversals_with_strength()
        
        plt.figure(figsize=(15, 7))
        plt.plot(data_to_plot.index, data_to_plot['Kapanış(TL)'], label='Kapanış Fiyatı')
        
        for reversal_type, index, pattern_type, strength in reversals:
            if data_to_plot.index[0] <= self.data.index[index] <= data_to_plot.index[-1]:
                color = 'g' if reversal_type == "Bullish" else 'r'
                marker = '^' if reversal_type == "Bullish" else 'v'
                plt.scatter(self.data.index[index], self.data['Kapanış(TL)'].iloc[index], 
                            color=color, s=100*strength, alpha=0.7, marker=marker)
        
        plt.title('NBAR Reversals')
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat (TL)')
        plt.legend()
        plt.grid(True)
        plt.show()