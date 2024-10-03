import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class SignalGenerator:
    def __init__(self, data):
        self.data = data
        self.signals = {}
        self.performance = {}
        self.generate_signals()

    def generate_signals(self):
        self.signals['SMA'] = self.sma_crossover()
        self.signals['RSI'] = self.rsi_signal()
        self.signals['BB'] = self.bollinger_bands_signal()

    def sma_crossover(self):
        sma_short = SMAIndicator(close=self.data['Kapanış(TL)'], window=10).sma_indicator()
        sma_long = SMAIndicator(close=self.data['Kapanış(TL)'], window=30).sma_indicator()
        return np.where(sma_short > sma_long, 1, np.where(sma_short < sma_long, -1, 0))

    def rsi_signal(self):
        rsi = RSIIndicator(close=self.data['Kapanış(TL)'], window=14).rsi()
        return np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))

    def bollinger_bands_signal(self):
        bb = BollingerBands(close=self.data['Kapanış(TL)'], window=20, window_dev=2)
        return np.where(self.data['Kapanış(TL)'] < bb.bollinger_lband(), 1, 
                        np.where(self.data['Kapanış(TL)'] > bb.bollinger_hband(), -1, 0))

    def get_signals(self, step):
        return {name: signal[step] for name, signal in self.signals.items()}

    def update_performance(self, signal_name, reward):
        if signal_name not in self.performance:
            self.performance[signal_name] = []
        self.performance[signal_name].append(reward)

    def get_best_signals(self, top_n=3):
        avg_performance = {name: np.mean(perf) for name, perf in self.performance.items()}
        return sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)[:top_n]
