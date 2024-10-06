
# StockTradingEnv sınıfı burada sona eriyor,
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ta.trend import EMAIndicator, MACD, IchimokuIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from collections import deque
from nbar_detector import NBARReversalDetector
from order_blocks import OrderBlockDetector
from tabulate import tabulate

class SignalGenerator:
    def __init__(self, data):
        self.data = data
        self.signals = {}
        self.weights = {}
        self.usage_count = {}
        self.reward_history = {}
        self.reward_deque = {}
        self.performance_score = {}
        self.n_steps_for_reward_avg = 10  # Ağırlık güncellemelerinde ortalamayı kullanma periyodu
        self.nbar_detector = NBARReversalDetector(data, use_volume=True, use_atr=True)
        self.ob_detector = OrderBlockDetector(data)
        self.ob_detector.detect_order_blocks()

        self.initialize_signals()
        self.initialize_weights()

    def initialize_signals(self):
        # Tüm sinyallerin başlatılması
        self.signals['Adaptive_SuperTrend'] = self.adaptive_supertrend()
        self.signals['EMA_5_10'] = self.ema_crossover(5, 10)
        self.signals['EMA_10_21'] = self.ema_crossover(10, 21)
        self.signals['MACD'] = self.macd_signal()
        self.signals['RSI'] = self.rsi_signal()
        self.signals['BB'] = self.bollinger_bands_signal()
        self.signals['PSAR'] = self.psar_signal()
        self.signals['NBAR'] = self.nbar_signal()
        self.signals['OrderBlocks'] = self.order_blocks_signal()
        self.signals['Ichimoku'] = self.ichimoku_signal()
        self.signals['Stochastic'] = self.stochastic_signal()
        self.signals['OBV'] = self.obv_signal()
        self.signals['Support_Resistance'] = self.support_resistance_signal()

    def initialize_weights(self):
        total_signals = len(self.signals)
        for signal in self.signals:
            if signal == 'Adaptive_SuperTrend':
                self.weights[signal] = 0.5  # Adaptive SuperTrend'in ağırlığını her zaman 0.5 yap
            elif signal == 'OrderBlocks':
                self.weights[signal] = 0.1  # Order Blocks için ağırlığı daha düşük yap
            else:
                self.weights[signal] = (1.0 - 0.5 - 0.1) / (total_signals - 2)  # Diğer sinyallerin ağırlığını dengeli dağıt
            self.usage_count[signal] = 0
            self.reward_history[signal] = []
            self.reward_deque[signal] = deque(maxlen=self.n_steps_for_reward_avg)
            self.performance_score[signal] = 0.5  # Başlangıç performans skoru


    # Doğrusal ağırlıklandırma
    def linear_weighting(self, performance_score):
        return performance_score  

    # Logaritmik ağırlıklandırma
    def logarithmic_weighting(self, performance_score):
        return np.log1p(performance_score)

    # Üssel ağırlıklandırma
    def exponential_weighting(self, performance_score):
        return np.exp(performance_score)

    def update_weights(self, rewards, method='linear'):
        """
        Ağırlıkları performans skoruna göre günceller.
        :param rewards: Her sinyalin aldığı ödüller.
        :param method: 'linear', 'logarithmic' ya da farklı ağırlıklandırma metodları.
        """
        # Seçilen metodun fonksiyonunu haritadan alalım
        method_map = {
            'linear': self.linear_weighting,
            'logarithmic': self.logarithmic_weighting,
            'exponential': self.exponential_weighting  # Ekstra metodlar buraya eklenebilir
        }

        if method not in method_map:
            raise ValueError(f"Invalid method. Available methods are: {list(method_map.keys())}")

        # Seçilen metodun fonksiyonunu çekiyoruz
        weighting_function = method_map[method]

        for signal, reward in rewards.items():
            if signal != 'Adaptive_SuperTrend':  # Adaptif SuperTrend sabit kalabilir
                self.reward_deque[signal].append(reward)
                avg_reward = np.mean(self.reward_deque[signal])

                # Performans skorunu güncelle
                self.performance_score[signal] = self.performance_score[signal] * 0.95 + (0.05 * (1 if avg_reward > 0 else 0))

                # Seçilen metodun fonksiyonunu kullanarak yeni ağırlığı hesapla
                new_weight = weighting_function(self.performance_score[signal])

                # Minimum ağırlık sınırı koy
                self.weights[signal] = max(0.01, new_weight)

        # Adaptive SuperTrend'in ağırlığını sabit tut
        self.weights['Adaptive_SuperTrend'] = 0.5

        # Nötr sinyalleri (ağırlık değeri 0'a yakın olanları) ayıkla ve diğerlerini normalize et
        total_weight = sum(w for s, w in self.weights.items() if s != 'Adaptive_SuperTrend' and abs(w) > 0.05)

        # Toplam ağırlığın 1 olmasını sağlamak için normalize et
        for signal in self.weights:
            if signal != 'Adaptive_SuperTrend' and abs(self.weights[signal]) > 0.05:
                self.weights[signal] /= total_weight
                self.weights[signal] *= 0.5  # Kalan ağırlıkları Adaptive_SuperTrend'e göre normalize et


    def adaptive_supertrend(self, atr_period=10, factor=3, training_period=100):
        """ Adaptif SuperTrend sinyali """
        high = self.data['Max(TL)']
        low = self.data['Min(TL)']
        close = self.data['Kapanış(TL)']

        atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=atr_period)
        atr = atr_indicator.average_true_range()

        volatility = atr.rolling(training_period).mean()
        volatility_array = volatility.dropna().values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(volatility_array)

        sorted_centers = np.sort(kmeans.cluster_centers_, axis=0)
        low_volatility, medium_volatility, high_volatility = sorted_centers.ravel()

        hl2 = (high + low) / 2
        supertrend = pd.Series(index=close.index, dtype='float64')
        direction = pd.Series(index=close.index, dtype='int')

        for i in range(len(close)):
            if i < training_period:
                supertrend.iloc[i] = np.nan
                direction.iloc[i] = 0
                continue

            current_volatility = volatility.iloc[i]
            if current_volatility <= low_volatility:
                current_factor = factor * 0.5
            elif current_volatility <= medium_volatility:
                current_factor = factor
            else:
                current_factor = factor * 1.5

            upperband = hl2.iloc[i] + current_factor * atr.iloc[i]
            lowerband = hl2.iloc[i] - current_factor * atr.iloc[i]

            if i > 0:
                if supertrend.iloc[i - 1] <= upperband:
                    supertrend.iloc[i] = min(upperband, supertrend.iloc[i - 1])
                else:
                    supertrend.iloc[i] = max(lowerband, supertrend.iloc[i - 1])

                if close.iloc[i] > supertrend.iloc[i]:
                    direction.iloc[i] = 1
                elif close.iloc[i] < supertrend.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i - 1]
            else:
                supertrend.iloc[i] = (upperband + lowerband) / 2
                direction.iloc[i] = 0

        return direction

    def ema_crossover(self, short_window, long_window):
        """ EMA crossover sinyali """
        short_ema = EMAIndicator(close=self.data['Kapanış(TL)'], window=short_window).ema_indicator()
        long_ema = EMAIndicator(close=self.data['Kapanış(TL)'], window=long_window).ema_indicator()
        signal = np.where(short_ema > long_ema, 1, -1)
        return pd.Series(signal, index=self.data.index)


    def macd_signal(self):
        """ MACD sinyali """
        macd = MACD(close=self.data['Kapanış(TL)'])
        macd_diff = macd.macd() - macd.macd_signal()
        signal = np.where(macd_diff > 0, 1, -1)
        return pd.Series(signal, index=self.data.index)

    def rsi_signal(self, window=14, overbought=65, oversold=35):
        """ RSI sinyali """
        rsi = RSIIndicator(close=self.data['Kapanış(TL)'], window=window).rsi()
        signal = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
        return pd.Series(signal, index=self.data.index)



    def bollinger_bands_signal(self, window=20, window_dev=2):
        """ Bollinger Band sinyali """
        indicator_bb = BollingerBands(close=self.data['Kapanış(TL)'], window=window, window_dev=window_dev)
        signal = np.where(self.data['Kapanış(TL)'] < indicator_bb.bollinger_lband(), 1,
                        np.where(self.data['Kapanış(TL)'] > indicator_bb.bollinger_hband(), -1, 0))
        return pd.Series(signal, index=self.data.index)


    def psar_signal(self, step=0.02, max_step=0.2):
        """ Parabolic SAR sinyali """
        psar = PSARIndicator(high=self.data['Max(TL)'], low=self.data['Min(TL)'], close=self.data['Kapanış(TL)'],
                            step=step, max_step=max_step)
        signal = np.where(self.data['Kapanış(TL)'] > psar.psar(), 1, -1)
        return pd.Series(signal, index=self.data.index)


    def nbar_signal(self):
        """ Geliştirilmiş NBAR sinyali """
        reversals_with_strength = self.nbar_detector.get_reversals_with_strength()
        signal = pd.Series(0, index=self.data.index)
        for reversal_type, index, pattern_type, strength in reversals_with_strength:
            if reversal_type == "Bullish":
                signal.iloc[index] = strength
            elif reversal_type == "Bearish":
                signal.iloc[index] = -strength
        return signal


    def order_blocks_signal(self):
        ob_signal = self.ob_detector.generate_ob_signal()
        
        
        # Sinyal değerlerini -1 ile 1 arasında sınırla
        ob_signal = ob_signal.clip(-1, 1)
        
        return ob_signal


    def ichimoku_signal(self):
        """ Ichimoku sinyali """
        ichimoku = IchimokuIndicator(high=self.data['Max(TL)'], low=self.data['Min(TL)'])
        tenkan = ichimoku.ichimoku_conversion_line()
        kijun = ichimoku.ichimoku_base_line()
        signal = np.where(tenkan > kijun, 1, np.where(tenkan < kijun, -1, 0))
        return pd.Series(signal, index=self.data.index)


    def stochastic_signal(self, window=14, smooth_window=3):
        """ Stokastik osilatör sinyali """
        stoch = StochasticOscillator(high=self.data['Max(TL)'], low=self.data['Min(TL)'], close=self.data['Kapanış(TL)'],
                                    window=window, smooth_window=smooth_window)
        signal = np.where(stoch.stoch_signal() > 80, -1, np.where(stoch.stoch_signal() < 20, 1, 0))
        return pd.Series(signal, index=self.data.index)


    def obv_signal(self):
        """ On Balance Volume (OBV) sinyali """
        obv = OnBalanceVolumeIndicator(close=self.data['Kapanış(TL)'], volume=self.data['Hacim(TL)'])
        obv_sma = obv.on_balance_volume().rolling(window=20).mean()
        signal = np.where(obv.on_balance_volume() > obv_sma, 1, -1)
        return pd.Series(signal, index=self.data.index)


    def support_resistance_signal(self, window=20):
        """ Destek ve direnç sinyali """
        pivot = (self.data['Max(TL)'] + self.data['Min(TL)'] + self.data['Kapanış(TL)']) / 3
        support = pivot - (self.data['Max(TL)'] - self.data['Min(TL)'])
        resistance = pivot + (self.data['Max(TL)'] - self.data['Min(TL)'])

        support_ma = support.rolling(window=window).mean()
        resistance_ma = resistance.rolling(window=window).mean()

        signal = np.where(self.data['Kapanış(TL)'] < support_ma, 1,
                        np.where(self.data['Kapanış(TL)'] > resistance_ma, -1, 0))
        return pd.Series(signal, index=self.data.index)


    def market_regime_signal(self, window=200):
        """ Piyasa rejimi sinyali """
        sma = EMAIndicator(close=self.data['Kapanış(TL)'], window=window).ema_indicator()
        volatility = self.data['Kapanış(TL)'].pct_change().rolling(window=window).std()

        trend = np.where(self.data['Kapanış(TL)'] > sma, 1, -1)
        regime = np.where(volatility > volatility.mean(), 0, trend)

        return pd.Series(regime, index=self.data.index)


    def get_signals(self, step):
        signals = {}
        for name, signal in self.signals.items():
            if isinstance(signal, pd.Series) and len(signal) > step:
                raw_signal = signal.iloc[step]
                # Sinyali -1 ile 1 arasında sınırla
                normalized_signal = np.clip(raw_signal, -1, 1)
                weighted_signal = normalized_signal * self.weights[name]
                signals[name] = {"value": weighted_signal, "status": self.get_signal_status(weighted_signal)}
            else:
                print(f"Error: {name} signal is invalid or step exceeded.")
                signals[name] = {"value": 0, "status": "N/A"}

        # OrderBlocks için ayrı işlem yapalım
        self.ob_detector.update_order_blocks()
        ob_signal = self.order_blocks_signal().iloc[step]
        normalized_ob_signal = np.clip(ob_signal, -1, 1)
        signals['OrderBlocks'] = {"value": normalized_ob_signal * self.weights['OrderBlocks'], "status": self.get_signal_status(normalized_ob_signal)}

        return signals
    

    def get_signal_status(self, signal_value):
        abs_value = abs(signal_value)
        
        # Sinyal değerlerine göre durumları güncelle (Aralıklar genişletildi)
        if abs_value > 0.5:  # Güçlü Al/Sat için aralığı genişletelim
            return "🟢 Strong Buy" if signal_value > 0 else "🔴 Strong Sell"
        elif abs_value > 0.2:  # Al/Sat için aralığı genişletelim
            return "🟡 Buy" if signal_value > 0 else "🟠 Sell"
        elif abs_value > 0.05:  # Nötr ve zayıf sinyaller için daha ince aralıklar
            return "⚪ Weak Buy" if signal_value > 0 else "⚪ Weak Sell"
        else:
            return "⚪ Neutral"  # Tamamen nötr sadece çok düşük değerlerde olmalı




    def get_total_signal(self, step):
        signals = self.get_signals(step)
        
        # Extract only the 'value' from the signal dictionaries for summing
        signal_values = [data['value'] for data in signals.values()]

        # Now sum the signal values
        total_signal = sum(signal_values)

        # Print the total signal and its status
        print(f"\n🔔 Total Signal: {total_signal:.4f}")
        print(f"Overall Status: {self.get_signal_status(total_signal)}")
        
        return total_signal





    
    def print_signal_summary(self, signals, statuses):
        for name, value in signals.items():
            status = statuses[name]
            weight = self.weights[name]
    
    
    

    def print_weights_and_usage(self):
        """ Print the weights and usage counts """
        print("\n🧮 Indicator Weights and Usage Counts:")
        for signal in self.weights:
            print(f"{signal}: Weight = {self.weights[signal]:.4f}, Usage Count = {self.usage_count[signal]}")
        
        most_used = max(self.usage_count, key=self.usage_count.get)
        print(f"\n🔥 Most Used Indicator: {most_used} (Usage Count: {self.usage_count[most_used]})")

    def reward_summary(self):
        """ Print the reward history and current weights """
        print("\n🏅 Reward History and Weights:")
        for signal in self.reward_history:
            avg_reward = np.mean(self.reward_history[signal]) if self.reward_history[signal] else 0
            print(f"{signal}: Average Reward = {avg_reward:.4f}, Current Weight = {self.weights[signal]:.4f}")

    def print_performance_summary(self):
        print("\n🏆 Performance Summary:")
        for signal, score in sorted(self.performance_score.items(), key=lambda x: x[1], reverse=True):
            print(f"{signal}: Performance Score = {score:.4f}, Weight = {self.weights[signal]:.4f}")
    def print_signal_summary_table(self, step):
        signals = self.get_signals(step)
        
        table_data = []
        for name, signal_info in signals.items():
            value = signal_info['value']
            status = signal_info['status']
            weight = self.weights[name]
            
            table_data.append([
                name, 
                f"{value:.4f}", 
                status, 
                f"{weight:.4f}"
            ])
        
        headers = ["Signal", "Value", "Status", "Weight"]
        table = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        
        print("\n📊 Signal Summary Table:")
        print(table)
        
        total_signal = sum(signal_info['value'] for signal_info in signals.values())
        print(f"\n🔔 Total Signal: {total_signal:.4f}")
        print(f"Overall Status: {self.get_signal_status(total_signal)}")