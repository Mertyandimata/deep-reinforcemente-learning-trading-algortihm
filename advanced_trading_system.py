import numpy as np
import pandas as pd
from scipy.stats import linregress
from tabulate import tabulate
from colorama import Fore, Style, init
from collections import deque
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

init(autoreset=True)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class AdvancedTradingSystem:
    def __init__(self, data):
        self.data = data
        self.signals = [
            'MACD', 'RSI', 'SMA_Crossover', 'Stochastic', 
            'Bollinger', 'Williams_R', 'Trend_Strength', 'Price_Action',
            'Volume_Price_Trend', 'Volatility'
        ]
        self.weights = {signal: 1/len(self.signals) for signal in self.signals}
        self.performance_score = {signal: 0.5 for signal in self.signals}
        self.reward_deque = {signal: deque(maxlen=100) for signal in self.signals}
        self.correct_signals = {signal: 0 for signal in self.signals}
        self.incorrect_signals = {signal: 0 for signal in self.signals}
        self.initialize_indicators()
        self.trend_predictor = self.initialize_trend_predictor()
        self.previous_signals = {signal: 0 for signal in self.signals}
        self.signal_performance = {signal: {'correct': 0, 'incorrect': 0} for signal in self.signals}
        self.current_step = 0 

    def evaluate_signal_performance(self, current_step):
        current_price = self.data['Kapanış(TL)'].iloc[current_step]
        previous_price = self.data['Kapanış(TL)'].iloc[current_step - 1] if current_step > 0 else current_price
        price_change = current_price - previous_price

        for signal in self.signals:
            previous_signal = self.previous_signals[signal]
            current_signal = self.generate_signal(signal, current_step)

            if (previous_signal > 0 and price_change > 0) or (previous_signal < 0 and price_change < 0):
                self.signal_performance[signal]['correct'] += 1
            elif previous_signal != 0:  # Sadece alım veya satım sinyali verildiyse yanlış sayılır
                self.signal_performance[signal]['incorrect'] += 1

            self.previous_signals[signal] = current_signal


    def initialize_indicators(self):
        # MACD
        macd = MACD(close=self.data['Kapanış(TL)'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Hist'] = macd.macd_diff()

        # RSI
        rsi = RSIIndicator(close=self.data['Kapanış(TL)'])
        self.data['RSI'] = rsi.rsi()

        # SMA Crossover
        self.data['SMA_50'] = SMAIndicator(close=self.data['Kapanış(TL)'], window=50).sma_indicator()
        self.data['SMA_200'] = SMAIndicator(close=self.data['Kapanış(TL)'], window=200).sma_indicator()

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=self.data['Max(TL)'], low=self.data['Min(TL)'], close=self.data['Kapanış(TL)'])
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()

        # Bollinger Bands
        bb = BollingerBands(close=self.data['Kapanış(TL)'])
        self.data['BB_Upper'] = bb.bollinger_hband()
        self.data['BB_Lower'] = bb.bollinger_lband()
        self.data['BB_Mid'] = bb.bollinger_mavg()

        # Williams %R
        williams_r = WilliamsRIndicator(high=self.data['Max(TL)'], low=self.data['Min(TL)'], close=self.data['Kapanış(TL)'])
        self.data['Williams_R'] = williams_r.williams_r()

        # Volatility (ATR)
        atr = AverageTrueRange(high=self.data['Max(TL)'], low=self.data['Min(TL)'], close=self.data['Kapanış(TL)'])
        self.data['ATR'] = atr.average_true_range()

        # Trend Strength
        self.data['Trend_Strength'] = self.calculate_trend_strength()

        # Price Action
        self.data['Price_Action'] = self.calculate_price_action()

        # Volume Price Trend
        self.data['Volume_Price_Trend'] = self.calculate_volume_price_trend()

        print("Indicators initialized. Data columns:")
        print(self.data.columns)

    def calculate_trend_strength(self):
        price = self.data['Kapanış(TL)']
        ma50 = self.data['SMA_50']
        ma200 = self.data['SMA_200']
        atr = self.data['ATR']
        
        trend_strength = ((price - ma200) / ma200) * (ma50 / ma200) * (atr / price)
        return trend_strength

    def calculate_price_action(self):
        high_prices = self.data['Max(TL)']
        low_prices = self.data['Min(TL)']
        close_prices = self.data['Kapanış(TL)']
        
        body = close_prices - close_prices.shift(1)
        range_ = high_prices - low_prices
        
        price_action = body / range_
        return price_action

    def calculate_volume_price_trend(self):
        close = self.data['Kapanış(TL)']
        volume = self.data['Hacim(TL)']
        
        vpt = (close - close.shift(1)) / close.shift(1) * volume
        return vpt.cumsum()

    def initialize_trend_predictor(self):
        print("\nInitializing Trend Predictor...")
        print(f"Total rows in dataset: {len(self.data)}")

        features = ['MACD', 'RSI', 'Stoch_K', 'Stoch_D', 'ATR', 'Williams_R', 'Trend_Strength', 'Price_Action', 'Volume_Price_Trend']
        
        # Eksik değerleri kontrol et ve raporla
        missing_data = self.data[features + ['Kapanış(TL)']].isnull().sum()
        print("\nMissing data in features:")
        print(missing_data)

        # Eksik değerleri doldur
        self.data[features] = self.data[features].ffill().bfill()

        X = self.data[features]
        y = self.data['Kapanış(TL)'].pct_change().shift(-1)  # Bir sonraki günün getirisini tahmin et

        # Son satırı kaldır çünkü y'de karşılığı yok
        X = X.iloc[:-1]
        y = y.iloc[:-1]

        # NaN değerleri olan satırları kaldır
        valid_data = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_data]
        y = y[valid_data]

        print(f"\nShape of X after preprocessing: {X.shape}")
        print(f"Shape of y after preprocessing: {y.shape}")

        if len(X) == 0 or len(y) == 0:
            print("Error: No valid data left after preprocessing!")
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        try:
            model.fit(X_scaled, y)
            print("\nModel fitted successfully!")
        except Exception as e:
            print(f"\nError fitting the model: {e}")
            return None

        # Model performansını değerlendir
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")

        return {'model': model, 'scaler': scaler, 'features': features}

    def predict_trend(self, current_step):
        features = self.data[['MACD', 'RSI', 'Stoch_K', 'Stoch_D', 'ATR', 'Williams_R', 'Trend_Strength', 'Price_Action', 'Volume_Price_Trend']].iloc[current_step]
        scaled_features = self.trend_predictor['scaler'].transform(features.values.reshape(1, -1))
        prediction = self.trend_predictor['model'].predict(scaled_features)[0]
        return prediction
    def generate_signal(self, indicator, current_step):
        if indicator == 'MACD':
            macd = self.data['MACD'].iloc[current_step]
            signal = self.data['MACD_Signal'].iloc[current_step]
            hist = self.data['MACD_Hist'].iloc[current_step]
            prev_hist = self.data['MACD_Hist'].iloc[current_step - 1] if current_step > 0 else 0
            
            if macd > signal and hist > 0 and hist > prev_hist:
                return 1  # Güçlü alım sinyali
            elif macd < signal and hist < 0 and hist < prev_hist:
                return -1  # Güçlü satım sinyali
            elif macd > signal and hist > 0:
                return 0.5  # Zayıf alım sinyali
            elif macd < signal and hist < 0:
                return -0.5  # Zayıf satım sinyali
            else:
                return 0  # Nötr

        elif indicator == 'RSI':
            rsi = self.data['RSI'].iloc[current_step]
            prev_rsi = self.data['RSI'].iloc[current_step - 1] if current_step > 0 else 50
            
            if rsi < 30 and rsi > prev_rsi:
                return 1  # Güçlü aşırı satım, alım fırsatı
            elif rsi > 70 and rsi < prev_rsi:
                return -1  # Güçlü aşırı alım, satım fırsatı
            elif rsi < 40 and rsi > prev_rsi:
                return 0.5  # Zayıf aşırı satım
            elif rsi > 60 and rsi < prev_rsi:
                return -0.5  # Zayıf aşırı alım
            else:
                return 0  # Nötr bölge

        elif indicator == 'SMA_Crossover':
            sma50 = self.data['SMA_50'].iloc[current_step]
            sma200 = self.data['SMA_200'].iloc[current_step]
            prev_sma50 = self.data['SMA_50'].iloc[current_step - 1] if current_step > 0 else sma50
            prev_sma200 = self.data['SMA_200'].iloc[current_step - 1] if current_step > 0 else sma200
            
            if sma50 > sma200 and prev_sma50 <= prev_sma200:
                return 1  # Altın çapraz, güçlü alım sinyali
            elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                return -1  # Ölüm çaprazı, güçlü satım sinyali
            elif sma50 > sma200 and (sma50 - sma200) > (prev_sma50 - prev_sma200):
                return 0.5  # Yükselen trend devam ediyor
            elif sma50 < sma200 and (sma200 - sma50) > (prev_sma200 - prev_sma50):
                return -0.5  # Düşen trend devam ediyor
            else:
                return 0  # Nötr

        elif indicator == 'Stochastic':
            k = self.data['Stoch_K'].iloc[current_step]
            d = self.data['Stoch_D'].iloc[current_step]
            prev_k = self.data['Stoch_K'].iloc[current_step - 1] if current_step > 0 else k
            prev_d = self.data['Stoch_D'].iloc[current_step - 1] if current_step > 0 else d
            
            if k > d and k < 20 and prev_k <= prev_d:
                return 1  # Aşırı satım bölgesinde alttan kesişim, güçlü alım sinyali
            elif k < d and k > 80 and prev_k >= prev_d:
                return -1  # Aşırı alım bölgesinde üstten kesişim, güçlü satım sinyali
            elif k > d and k < 50:
                return 0.5  # Zayıf alım sinyali
            elif k < d and k > 50:
                return -0.5  # Zayıf satım sinyali
            else:
                return 0  # Nötr

        elif indicator == 'Bollinger':
            price = self.data['Kapanış(TL)'].iloc[current_step]
            upper = self.data['BB_Upper'].iloc[current_step]
            lower = self.data['BB_Lower'].iloc[current_step]
            middle = self.data['BB_Mid'].iloc[current_step]
            
            bandwidth = (upper - lower) / middle
            
            if price < lower and bandwidth > 0.1:
                return 1  # Fiyat alt bandın altında ve bantlar geniş, güçlü alım sinyali
            elif price > upper and bandwidth > 0.1:
                return -1  # Fiyat üst bandın üstünde ve bantlar geniş, güçlü satım sinyali
            elif price < middle and price > lower:
                return 0.5  # Fiyat orta ile alt bant arasında, zayıf alım sinyali
            elif price > middle and price < upper:
                return -0.5  # Fiyat orta ile üst bant arasında, zayıf satım sinyali
            else:
                return 0  # Nötr

        elif indicator == 'Williams_R':
            williams = self.data['Williams_R'].iloc[current_step]
            prev_williams = self.data['Williams_R'].iloc[current_step - 1] if current_step > 0 else williams
            
            if williams < -80 and williams > prev_williams:
                return 1  # Aşırı satım bölgesinden çıkış, güçlü alım sinyali
            elif williams > -20 and williams < prev_williams:
                return -1  # Aşırı alım bölgesinden çıkış, güçlü satım sinyali
            elif williams < -50 and williams > prev_williams:
                return 0.5  # Satım bölgesinde yükseliş, zayıf alım sinyali
            elif williams > -50 and williams < prev_williams:
                return -0.5  # Alım bölgesinde düşüş, zayıf satım sinyali
            else:
                return 0  # Nötr

        elif indicator == 'Trend_Strength':
            strength = self.data['Trend_Strength'].iloc[current_step]
            prev_strength = self.data['Trend_Strength'].iloc[current_step - 1] if current_step > 0 else 0
            
            if strength > 0.05 and strength > prev_strength:
                return 1  # Güçlenen pozitif trend, güçlü alım sinyali
            elif strength < -0.05 and strength < prev_strength:
                return -1  # Güçlenen negatif trend, güçlü satım sinyali
            elif strength > 0:
                return 0.5  # Zayıf pozitif trend, zayıf alım sinyali
            elif strength < 0:
                return -0.5  # Zayıf negatif trend, zayıf satım sinyali
            else:
                return 0  # Nötr veya belirsiz trend

        elif indicator == 'Price_Action':
            pa = self.data['Price_Action'].iloc[current_step]
            price = self.data['Kapanış(TL)'].iloc[current_step]
            prev_price = self.data['Kapanış(TL)'].iloc[current_step - 1] if current_step > 0 else price
            
            if pa > 0.6 and price > prev_price:
                return 1  # Güçlü alıcı baskısı ve yükselen fiyat, güçlü alım sinyali
            elif pa < -0.6 and price < prev_price:
                return -1  # Güçlü satıcı baskısı ve düşen fiyat, güçlü satım sinyali
            elif pa > 0.3 and price > prev_price:
                return 0.5  # Orta seviye alıcı baskısı, zayıf alım sinyali
            elif pa < -0.3 and price < prev_price:
                return -0.5  # Orta seviye satıcı baskısı, zayıf satım sinyali
            else:
                return 0  # Nötr veya kararsız piyasa

        elif indicator == 'Volume_Price_Trend':
            vpt = self.data['Volume_Price_Trend'].iloc[current_step]
            prev_vpt = self.data['Volume_Price_Trend'].iloc[current_step - 1] if current_step > 0 else vpt
            
            if vpt > prev_vpt * 1.05:
                return 1  # Hacim ve fiyat trendi güçlü bir şekilde yükseliyor
            elif vpt < prev_vpt * 0.95:
                return -1  # Hacim ve fiyat trendi güçlü bir şekilde düşüyor
            elif vpt > prev_vpt:
                return 0.5  # Hacim ve fiyat trendi hafif yükseliyor
            elif vpt < prev_vpt:
                return -0.5  # Hacim ve fiyat trendi hafif düşüyor
            else:
                return 0  # Nötr

        elif indicator == 'Volatility':
            atr = self.data['ATR'].iloc[current_step]
            avg_atr = self.data['ATR'].rolling(window=14).mean().iloc[current_step]
            
            if atr < avg_atr * 0.5:
                return 1  # Düşük volatilite, potansiyel breakout fırsatı
            elif atr > avg_atr * 2:
                return -1  # Çok yüksek volatilite, riskli piyasa
            elif atr < avg_atr * 0.75:
                return 0.5  # Orta-düşük volatilite
            elif atr > avg_atr * 1.5:
                return -0.5  # Orta-yüksek volatilite
            else:
                return 0  # Normal volatilite

        else:
            return 0  # Tanımlanmamış gösterge için nötr sinyal

    def generate_combined_signal(self, current_step):
        self.current_step = current_step  # current_step'i güncelle
        signals = {}
        for indicator in self.signals:
            signals[indicator] = self.generate_signal(indicator, current_step)
        
        weighted_signals = [signals[s] * self.weights[s] for s in self.signals]
        combined_signal = sum(weighted_signals)
        
        trend_prediction = self.predict_trend(current_step)
        
        # Trend tahmini ile birleştirilmiş sinyal
        combined_signal = combined_signal * 0.7 + trend_prediction * 0.3
        
        signal_strength = abs(combined_signal)
        
        if combined_signal > 0:
            if signal_strength > 0.8:
                return "Strong Buy", 0.9, signals
            elif signal_strength > 0.5:
                return "Buy", 0.6, signals
            elif signal_strength > 0.3:
                return "Weak Buy", 0.3, signals
            else:
                return "Hold", 0, signals
        else:
            if signal_strength > 0.8:
                return "Strong Sell", -0.9, signals
            elif signal_strength > 0.5:
                return "Sell", -0.6, signals
            elif signal_strength > 0.3:
                return "Weak Sell", -0.3, signals
            else:
                return "Hold", 0, signals

    def update_weights_with_softmax(self):
        print("\n🔄 Updating Weights Using Softmax:")
        
        performance_scores = np.array([self.performance_score[signal] for signal in self.signals])
        
        # Performans skorlarını pozitif yapmak için min-max normalizasyonu uygulayalım
        min_score = np.min(performance_scores)
        max_score = np.max(performance_scores)
        
        if max_score > min_score:
            normalized_scores = (performance_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(performance_scores) / len(performance_scores)
        
        # Normalize edilmiş skorlara softmax uygulayalım
        softmax_weights = softmax(normalized_scores)
        
        for idx, signal in enumerate(self.signals):
            old_weight = self.weights[signal]
            new_weight = softmax_weights[idx]
            self.weights[signal] = new_weight
            print(f"{signal}: Old Weight = {old_weight:.4f}, New Weight = {new_weight:.4f}")
        
        # Ağırlıkların toplamının 1 olduğundan emin olalım
        total_weight = sum(self.weights.values())
        for signal in self.weights:
            self.weights[signal] /= total_weight

    def update_performance_scores(self, rewards):
        print("\n🎯 Updating Performance Scores:")
        
        self.evaluate_signal_performance(self.current_step)

        for signal in self.signals:
            if signal in rewards:
                reward = rewards[signal]
                self.reward_deque[signal].append(reward)
                avg_reward = np.mean(self.reward_deque[signal])
                
                old_score = self.performance_score[signal]
                self.performance_score[signal] = old_score * 0.95 + (0.05 * avg_reward)
                
                correct = self.signal_performance[signal]['correct']
                incorrect = self.signal_performance[signal]['incorrect']
                total = correct + incorrect
                accuracy = correct / total if total > 0 else 0
                
                print(f"{signal}: Old Score = {old_score:.4f}, New Score = {self.performance_score[signal]:.4f}")
                print(f"  Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:.2%}")
        
        self.update_weights_with_softmax()


    def print_summary(self, current_step):
        general_signal, strength, individual_signals = self.generate_combined_signal(current_step)
        trend_prediction = self.predict_trend(current_step)
        
        print(f"\n{Fore.CYAN}═══ Trading System Summary ═══{Style.RESET_ALL}")
        
        # Genel Sinyal Tablosu
        general_table = tabulate([
            ["General Signal", self.colorize_value(general_signal)],
            ["Signal Strength", f"{strength:.2f}"],
            ["Trend Prediction", f"{trend_prediction:.2%}"]
        ], tablefmt="fancy_grid")
        
        # Bireysel Sinyal Tablosu
        individual_table = tabulate([
            [signal, self.colorize_value(value), f"{self.weights[signal]:.4f}"] 
            for signal, value in individual_signals.items()
        ], headers=["Indicator", "Signal", "Weight"], tablefmt="fancy_grid")
        
        # Performans Tablosu
        performance_table = tabulate([
            [signal, f"{self.correct_signals[signal]}", f"{self.incorrect_signals[signal]}", 
             f"{self.correct_signals[signal] / (self.correct_signals[signal] + self.incorrect_signals[signal] + 1e-10):.2f}",
             f"{self.performance_score[signal]:.4f}"]
            for signal in self.signals
        ], headers=["Indicator", "Correct", "Incorrect", "Accuracy", "Score"], tablefmt="fancy_grid")
        
        # Tabloları yan yana yazdır
        self.print_side_by_side([general_table, individual_table, performance_table])

    def colorize_value(self, value):
        if isinstance(value, str):
            if "Buy" in value or value == 1:
                return f"{Fore.GREEN}{value}{Style.RESET_ALL}"
            elif "Sell" in value or value == -1:
                return f"{Fore.RED}{value}{Style.RESET_ALL}"
            else:
                return f"{Fore.YELLOW}{value}{Style.RESET_ALL}"
        elif isinstance(value, (int, float)):
            if value > 0:
                return f"{Fore.GREEN}{value:.2f}{Style.RESET_ALL}"
            elif value < 0:
                return f"{Fore.RED}{value:.2f}{Style.RESET_ALL}"
            else:
                return f"{Fore.YELLOW}{value:.2f}{Style.RESET_ALL}"
        else:
            return str(value)

    def print_side_by_side(self, tables):
        lines = [table.split('\n') for table in tables]
        max_lines = max(len(table) for table in lines)
        padded_lines = [table + [''] * (max_lines - len(table)) for table in lines]
        for row in zip(*padded_lines):
            print('    '.join(row))

    def analyze_market(self, current_step):
        general_signal, strength, individual_signals = self.generate_combined_signal(current_step)
        trend_prediction = self.predict_trend(current_step)
        
        current_price = self.data['Kapanış(TL)'].iloc[current_step]
        prev_price = self.data['Kapanış(TL)'].iloc[current_step - 1] if current_step > 0 else current_price
        price_change = (current_price - prev_price) / prev_price

        market_analysis = {
            "General Signal": general_signal,
            "Signal Strength": strength,
            "Trend Prediction": trend_prediction,
            "Current Price": current_price,
            "Price Change": f"{price_change:.2%}",
            "Individual Signals": individual_signals
        }

        return market_analysis

    def should_open_position(self, current_step, balance):
        general_signal, strength, _ = self.generate_combined_signal(current_step)
        
        if general_signal in ["Strong Buy", "Buy"] and strength > 0.5:
            return True, min(balance, balance * strength)  # Pozisyon büyüklüğü sinyal gücüne bağlı
        return False, 0

    def should_close_position(self, current_step, entry_price):
        general_signal, strength, _ = self.generate_combined_signal(current_step)
        current_price = self.data['Kapanış(TL)'].iloc[current_step]
        
        if general_signal in ["Strong Sell", "Sell"] and strength > 0.5:
            return True
        
        # Stop-loss ve take-profit kontrolü
        profit_percentage = (current_price - entry_price) / entry_price
        if profit_percentage <= -0.05 or profit_percentage >= 0.1:
            return True
        
        return False

    def should_adjust_position(self, current_step, position, entry_price):
        general_signal, strength, _ = self.generate_combined_signal(current_step)
        current_price = self.data['Kapanış(TL)'].iloc[current_step]
        
        if general_signal in ["Strong Buy", "Buy"] and strength > 0.7:
            return "increase", position * 0.2  # Pozisyonu %20 artır
        elif general_signal in ["Strong Sell", "Sell"] and strength > 0.7:
            return "decrease", position * 0.2  # Pozisyonu %20 azalt
        
        # Trailing stop-loss kontrolü
        profit_percentage = (current_price - entry_price) / entry_price
        if profit_percentage > 0.05:
            new_stop_loss = current_price * 0.95
            if new_stop_loss > entry_price:
                return "update_stop_loss", new_stop_loss
        
        return "hold", 0

    def set_dynamic_stop_loss_take_profit(self, current_step, entry_price):
        atr = self.data['ATR'].iloc[current_step]
        
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        
        return stop_loss, take_profit