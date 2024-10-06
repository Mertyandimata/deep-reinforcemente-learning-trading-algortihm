import numpy as np
import pandas as pd
from tabulate import tabulate
import logging
from colorama import Fore, Style, init
from collections import deque
from RewardCalculator import RewardCalculator
from graph_visualizer import GraphVisualizer
from advanced_trading_system import AdvancedTradingSystem
from gym import spaces

init(autoreset=True)

class StockTradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.advanced_system = AdvancedTradingSystem(data)
        self.reward_calculator = RewardCalculator(self)
        self.graph_visualizer = GraphVisualizer()
        self.ascii_charts = [] 
        self.trailing_stop = 0
        self.trailing_take_profit = 0
        self.position_duration = 0
        self.max_position_value = 0
        self.price_history = deque(maxlen=100)
        self.volatility_factor = 1.0
        self.trades = []
        self.observation_window = 50  # Son 50 günlük veriyi gözlemle
        self.action_space = spaces.Discrete(21)  # 0: Sat, 1-10: Tut, 11-20: Al (farklı miktarlarda)



        logging.basicConfig(filename='trading_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.total_profit = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.highest_price = 0
        self.trailing_stop = 0
        self.trailing_take_profit = 0
        self.position_duration = 0
        self.max_position_value = 0
        self.price_history.clear()
        self.volatility_factor = 1.0
        self.update_graph()
        return self._next_observation()

    def _next_observation(self):
        end_idx = self.current_step + 1
        start_idx = max(0, end_idx - self.observation_window)
        
        features = ['Kapanış(TL)', 'USDTRY', 'BIST 100', 'PiyasaDeğeri(mn TL)']
        technical_indicators = ['RSI', 'MACD', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower', 'BB_Mid', 'Stoch_K', 'Stoch_D', 'Williams_R', 'ATR', 'Trend_Strength', 'Price_Action', 'Volume_Price_Trend']
        
        all_features = features + [ti for ti in technical_indicators if ti in self.data.columns]
        
        obs = self.data[all_features].iloc[start_idx:end_idx].values
        
        current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
        position_value = self.position * current_price
        total_value = self.balance + position_value
        
        account_info = np.array([
            self.balance / total_value,
            position_value / total_value,
            self.position,
            self.entry_price,
            (current_price - self.entry_price) / self.entry_price if self.position > 0 else 0,
            (self.stop_loss - current_price) / current_price if self.stop_loss > 0 else -1,
            (self.take_profit - current_price) / current_price if self.take_profit > 0 else -1,
        ])
        
        account_info_reshaped = np.tile(account_info, (end_idx - start_idx, 1))
        
        obs = np.hstack([obs, account_info_reshaped])
        
        print(f"Observation shape: {obs.shape}")
        
        return obs

    def step(self, action):
        # Adımdan önceki portföy değerini hesapla
        prev_value = self.balance + self.position * self.data['Kapanış(TL)'].iloc[self.current_step]

        try:
            # Eylemi gerçekleştir (al, sat veya tut)
            self._take_action(action)
        except Exception as e:
            print(f"Action step error: {e}")

        # Mevcut adımı bir artır
        self.current_step += 1

        # Eğer veri setinin sonuna geldiysek, done'ı True yap
        done = self.current_step >= len(self.data) - 1

        # Yeni gözlemi al
        obs = self._next_observation()

        # Adımdan sonraki portföy değerini hesapla
        current_value = self.balance + self.position * self.data['Kapanış(TL)'].iloc[self.current_step]

        # Ödülü hesapla
        reward = self._calculate_reward(prev_value, current_value)

        # Farklı göstergeler için ödülleri hazırla
        rewards = {"MACD": reward, "RSI": reward, "SMA_50_200": reward}

        # AdvancedTradingSystem'in mevcut adımını güncelle ve sinyalleri oluştur
        self.advanced_system.generate_combined_signal(self.current_step)

        # Performans skorlarını güncelle
        self.advanced_system.update_performance_scores(rewards)

        # Ağırlıkları güncelle
        self.advanced_system.update_weights_with_softmax()

        # Toplam kârı güncelle
        self.total_profit += current_value - prev_value

        # Bu adımla ilgili bilgileri topla
        info = {
            'step': self.current_step,
            'balance': current_value,
            'profit': self.total_profit,
            'reward': reward,
            'position': self.position,
            'action': action,
            'price': self.data['Kapanış(TL)'].iloc[self.current_step],
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }

        # Adım bilgilerini yazdır
        self._print_step_info(info)

        # Adım bilgilerini logla
        self._log_step_info(info)


        # OpenAI Gym standartlarına uygun olarak gözlem, ödül, bitti mi bilgisi ve ek bilgileri döndür
        return obs, reward, done, info
    def _apply_trend_following_strategy(self, current_price):
        if self.position > 0 and current_price > self.entry_price:
            stop_loss_pct, take_profit_pct, position_increase_pct = self._calculate_dynamic_parameters()
            
            new_stop_loss = max(self.stop_loss, current_price * (1 - stop_loss_pct))
            new_take_profit = max(self.take_profit, current_price * (1 + take_profit_pct))
            
            if current_price >= self.entry_price * (1 + position_increase_pct):
                self.stop_loss = new_stop_loss
                self.take_profit = new_take_profit
                print(f"\n{Fore.YELLOW}📈 Trend Takibi: Stop-Loss ve Take-Profit güncellendi{Style.RESET_ALL}")
                print(tabulate([
                    ["Yeni Stop-Loss", f"{self.stop_loss:.2f} TL"],
                    ["Yeni Take-Profit", f"{self.take_profit:.2f} TL"]
                ], tablefmt="fancy_grid"))
                
            # Eğer fiyat take-profit seviyesine ulaştıysa, pozisyonun yarısını kapat
            if current_price >= self.take_profit:
                shares_to_sell = self.position // 2
                if shares_to_sell > 0:
                    self._decrease_position(current_price, shares_to_sell)
                    print(f"\n{Fore.GREEN}💰 Trend Takibi: Take-Profit'e ulaşıldı, pozisyonun yarısı kapatıldı{Style.RESET_ALL}")
            
            # Eğer fiyat stop-loss seviyesine düştüyse, tüm pozisyonu kapat
            elif current_price <= self.stop_loss:
                self._close_position(current_price)
                print(f"\n{Fore.RED}🛑 Trend Takibi: Stop-Loss'a ulaşıldı, tüm pozisyon kapatıldı{Style.RESET_ALL}")


    def _take_action(self, action):
        try:
            current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
            self.price_history.append(current_price)

            # Mevcut karar mekanizması
            normalized_action = (action - 10) / 10
            
            if self.position == 0 and normalized_action > 0:
                shares_to_buy = int((self.balance * abs(normalized_action)) / current_price)
                if shares_to_buy > 0:
                    self._open_position(current_price, shares_to_buy)
            elif self.position > 0:
                if normalized_action < -0.5:
                    self._close_position(current_price)
                elif normalized_action < 0:
                    shares_to_sell = int(self.position * abs(normalized_action))
                    if shares_to_sell > 0:
                        self._decrease_position(current_price, shares_to_sell)
                elif normalized_action > 0:
                    additional_shares = int((self.balance * normalized_action) / current_price)
                    if additional_shares > 0:
                        self._increase_position(current_price, additional_shares)
            
            # Trend takip stratejisini uygula
            self._apply_trend_following_strategy(current_price)
            
            self._update_portfolio(current_price)

            market_analysis = self.advanced_system.analyze_market(self.current_step)
            self._log_action(action, market_analysis)
            
            self.update_graph()
        except Exception as e:
            print(f"Hata oluştu: {e}")
    def _calculate_dynamic_parameters(self):
        volatility = self._calculate_volatility()
        
        # Volatiliteye göre stop-loss ve take-profit yüzdelerini ayarla
        stop_loss_percentage = max(0.02, min(0.05, volatility * 2))
        take_profit_percentage = max(0.03, min(0.07, volatility * 3))
        
        # Trend gücüne göre pozisyon artırma yüzdesini ayarla
        trend_strength = abs(self._calculate_market_trend())
        position_increase_percentage = max(0.1, min(0.3, trend_strength * 0.2))
        
        return stop_loss_percentage, take_profit_percentage, position_increase_percentage


    def _update_trailing_stop_and_profit(self, current_price):
        if self.position <= 0:
            return

        profit_percentage = (current_price - self.entry_price) / self.entry_price
        volatility = self._calculate_volatility()

        # Dinamik trailing stop
        stop_loss_percentage = max(0.01, min(0.05, volatility * 2))  # %1 ile %5 arasında
        self.trailing_stop = max(self.trailing_stop, current_price * (1 - stop_loss_percentage))

        # Dinamik take-profit
        if profit_percentage >= 0.01:  # %1'den fazla kâr varsa
            take_profit_percentage = self._calculate_dynamic_take_profit(profit_percentage, volatility)
            self.take_profit = max(self.take_profit, current_price * (1 + take_profit_percentage))

        # Kısmi pozisyon kapatma stratejisi
        self._apply_partial_closing_strategy(current_price, profit_percentage)

        # Stop-loss ve take-profit kontrolü
        if current_price <= self.trailing_stop:
            self._close_position(current_price)
            print(f"\n{Fore.YELLOW}🛑 Trailing Stop Tetiklendi ({stop_loss_percentage:.2%}){Style.RESET_ALL}")
        elif current_price >= self.take_profit:
            self._partial_close_and_update(current_price)
            print(f"\n{Fore.GREEN}🎯 Take-Profit Tetiklendi ({take_profit_percentage:.2%}){Style.RESET_ALL}")

    def _calculate_volatility(self):
        # Son 20 günlük volatilite hesapla
        returns = np.diff(self.data['Kapanış(TL)'].iloc[max(0, self.current_step-20):self.current_step+1]) / \
                self.data['Kapanış(TL)'].iloc[max(0, self.current_step-20):self.current_step]
        return np.std(returns)
    def _calculate_dynamic_take_profit(self, profit_percentage, volatility):
        base_take_profit = 0.01  # Başlangıç %1
        
        # Kâr arttıkça take-profit yüzdesini artır
        if profit_percentage >= 0.05:
            base_take_profit = 0.02
        elif profit_percentage >= 0.02:
            base_take_profit = 0.015

        # Volatiliteye göre ayarla
        take_profit = base_take_profit * (1 + volatility)
        
        return min(0.05, take_profit)  # Maksimum %5 ile sınırla

    def _apply_partial_closing_strategy(self, current_price, profit_percentage):
        if profit_percentage >= 0.05:  # %5 ve üzeri kârda
            shares_to_sell = int(self.position * 0.25)  # %25'ini sat
            if shares_to_sell > 0:
                self._decrease_position(current_price, shares_to_sell)
                print(f"\n{Fore.GREEN}💰 Kısmi Kâr Realizasyonu (%25){Style.RESET_ALL}")
        elif profit_percentage >= 0.03:  # %3 ve üzeri kârda
            shares_to_sell = int(self.position * 0.15)  # %15'ini sat
            if shares_to_sell > 0:
                self._decrease_position(current_price, shares_to_sell)
                print(f"\n{Fore.GREEN}💰 Kısmi Kâr Realizasyonu (%15){Style.RESET_ALL}")

    def _partial_close_and_update(self, current_price):
        shares_to_sell = int(self.position * 0.5)  # %50'sini sat
        self._decrease_position(current_price, shares_to_sell)
        
        # Yeni entry price, take-profit ve trailing stop belirle
        self.entry_price = current_price
        self.take_profit = current_price * 1.02  # Yeni %2'lik hedef
        self.trailing_stop = current_price * 0.99  # Yeni %1'lik trailing stop

        print(f"\n{Fore.GREEN}🔄 Pozisyon Güncellendi{Style.RESET_ALL}")
        print(f"Yeni Entry Price: {self.entry_price:.2f}")
        print(f"Yeni Take-Profit: {self.take_profit:.2f}")
        print(f"Yeni Trailing Stop: {self.trailing_stop:.2f}")

    def _log_action(self, action, market_analysis):
        print(f"Adım: {self.current_step}, Aksiyon: {action}")
        print(f"Piyasa Analizi: {market_analysis}")

    def _increase_position(self, price, additional_shares):
        cost = additional_shares * price
        if cost <= self.balance:
            self.position += additional_shares
            self.balance -= cost
            self.entry_price = ((self.entry_price * self.position) + (price * additional_shares)) / (self.position + additional_shares)
            self._set_dynamic_stop_loss_take_profit(price)

            print(f"\n{Fore.GREEN}🔼 POZİSYON ARTIRILDI{Style.RESET_ALL}")
            print(tabulate([
                ["Eklenen Hisse Adedi", f"{additional_shares:,}"],
                ["Alış Fiyatı", f"{price:,.2f} TL"],
                ["Toplam Maliyet", f"{cost:,.2f} TL"],
                ["Yeni Pozisyon", f"{self.position:,}"],
                ["Yeni Ortalama Giriş Fiyatı", f"{self.entry_price:,.2f} TL"],
                ["Yeni Nakit Bakiye", f"{self.balance:,.2f} TL"]
            ], tablefmt="fancy_grid"))

    def _decrease_position(self, price, shares_to_sell):
        if shares_to_sell <= self.position:
            sell_value = shares_to_sell * price
            self.balance += sell_value
            self.position -= shares_to_sell
            
            print(f"\n{Fore.YELLOW}🔽 POZİSYON AZALTILDI{Style.RESET_ALL}")
            print(tabulate([
                ["Satılan Hisse Adedi", f"{shares_to_sell:,}"],
                ["Satış Fiyatı", f"{price:,.2f} TL"],
                ["Toplam Değer", f"{sell_value:,.2f} TL"],
                ["Kalan Pozisyon", f"{self.position:,}"],
                ["Yeni Nakit Bakiye", f"{self.balance:,.2f} TL"]
            ], tablefmt="fancy_grid"))
            
            if self.position == 0:
                self.entry_price = 0
                self.stop_loss = 0
                self.take_profit = 0
                self.highest_price = 0
            else:
                self._set_dynamic_stop_loss_take_profit(price)

    def _update_volatility_factor(self):
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            volatility = np.std(returns)
            self.volatility_factor = max(1.0, min(2.0, volatility * 10))

    def _open_position(self, price, shares_to_buy):
        cost = shares_to_buy * price
        if cost <= self.balance:
            self.position += shares_to_buy
            self.balance -= cost
            self.entry_price = price
            self.highest_price = price

            self._set_dynamic_stop_loss_take_profit(price)
            self.trailing_stop = price * 0.95
            self.trailing_take_profit = price * 1.05
            self.position_duration = 0
            self.max_position_value = price * shares_to_buy
            self._update_volatility_factor()

            self.trades.append({'type': 'buy', 'price': price, 'shares': shares_to_buy, 'step': self.current_step})
            
            print(f"\n{Fore.GREEN}🛒 ALIŞ İŞLEMİ GERÇEKLEŞTI{Style.RESET_ALL}")
            print(tabulate([
                ["Alınan Hisse Adedi", f"{shares_to_buy:,}"],
                ["Alış Fiyatı", f"{price:,.2f} TL"],
                ["Toplam Maliyet", f"{cost:,.2f} TL"],
                ["Yeni Nakit Bakiye", f"{self.balance:,.2f} TL"],
                ["Stop-Loss", f"{self.stop_loss:,.2f} TL"],
                ["Take-Profit", f"{self.take_profit:,.2f} TL"]
            ], tablefmt="fancy_grid"))

    def _close_position(self, price):
        if self.position > 0:
            sell_value = self.position * price
            profit = sell_value - (self.position * self.entry_price)
            self.balance += sell_value

            self.trades.append({'type': 'sell', 'price': price, 'shares': self.position, 'step': self.current_step})

            print(f"\n{Fore.RED}💰 SATIŞ İŞLEMİ GERÇEKLEŞTI{Style.RESET_ALL}")
            print(tabulate([
                ["Satılan Hisse Adedi", f"{self.position:,}"],
                ["Satış Fiyatı", f"{price:,.2f} TL"],
                ["Toplam Değer", f"{sell_value:,.2f} TL"],
                ["Kâr/Zarar", f"{profit:,.2f} TL"],
                ["Yeni Nakit Bakiye", f"{self.balance:,.2f} TL"]
            ], tablefmt="fancy_grid"))

            self.position = 0
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            self.highest_price = 0
            self.trailing_stop = 0
            self.trailing_take_profit = 0
            self.position_duration = 0
            self.max_position_value = 0

    def _calculate_market_trend(self):
        ma_50 = self.data['Kapanış(TL)'].rolling(window=20).mean().iloc[self.current_step]
        ma_100 = self.data['Kapanış(TL)'].rolling(window=50).mean().iloc[self.current_step]
        current_price = self.data['Kapanış(TL)'].iloc[self.current_step]

        if np.isnan(ma_50) or np.isnan(ma_100) or np.isnan(current_price):
            return 0

        if current_price > ma_50 and ma_50 > ma_100:
            return 1
        elif current_price < ma_50 and ma_50 < ma_100:
            return -1
        else:
            return 0
    def _update_trailing_stop_and_profit(self, current_price):
        high = self.data['Max(TL)'].iloc[self.current_step]
        low = self.data['Min(TL)'].iloc[self.current_step]
        
        if self.current_step == 0:
            prev_close = self.data['Kapanış(TL)'].iloc[self.current_step]
        else:
            prev_close = self.data['Kapanış(TL)'].iloc[self.current_step - 1]

        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atr = true_range
        atr_multiplier = 2
        new_stop_loss = current_price - (atr * atr_multiplier)

        position_value = self.position * current_price

        if position_value > self.max_position_value:
            self.max_position_value = position_value

        new_trailing_stop = current_price * (1 - 0.05 * self.volatility_factor)
        if new_trailing_stop > self.trailing_stop:
            self.trailing_stop = new_trailing_stop

        new_trailing_take_profit = current_price * (1 + 0.05 * self.volatility_factor)
        if new_trailing_take_profit > self.trailing_take_profit:
            self.trailing_take_profit = new_trailing_take_profit

        profit_percentage = (position_value - (self.position * self.entry_price)) / (self.position * self.entry_price)
        if profit_percentage > 0.1:
            self.trailing_stop = max(self.trailing_stop, current_price * 0.98)
        elif profit_percentage > 0.2:
            self.trailing_stop = max(self.trailing_stop, current_price * 0.99)

        if current_price > self.highest_price:
            self.highest_price = current_price
            self.stop_loss = max(self.stop_loss, new_stop_loss)
            print(f"\n{Fore.YELLOW}📈 YENİ EN YÜKSEK FİYAT VE STOP-LOSS{Style.RESET_ALL}")
            print(tabulate([
                ["Yeni En Yüksek Fiyat", f"{current_price:,.2f} TL"],
                ["Yeni Stop-Loss", f"{self.stop_loss:,.2f} TL"]
            ], tablefmt="fancy_grid"))
        elif current_price < self.highest_price:
            print(f"\n{Fore.YELLOW}📉 FİYAT DÜŞÜŞÜ{Style.RESET_ALL}")
            print(tabulate([
                ["Mevcut Fiyat", f"{current_price:,.2f} TL"],
                ["Mevcut Stop-Loss", f"{self.stop_loss:,.2f} TL"]
            ], tablefmt="fancy_grid"))

        if current_price <= self.trailing_stop:
            self._close_position(current_price)
        elif current_price >= self.trailing_take_profit:
            self._close_position(current_price)

    def _get_robot_decision(self, action):
        if action < 5:
            return "🔴 Sat"
        elif action > 15:
            return "🟢 Al"
        else:
            return "⚪ Tut"

    def _print_step_info(self, info):
        try:
            robot_decision = self._get_robot_decision(info['action'])
            
            print(f"\n{'='*150}")
            print(f"{Fore.CYAN}📊 ADIM BİLGİLERİ - Adım {info.get('step', 'N/A')}{Style.RESET_ALL}")
            
            step_table = tabulate([
                ["Mevcut Fiyat", f"{info.get('price', 'N/A'):,.2f} TL" if isinstance(info.get('price'), (int, float)) else "N/A"],
                ["Portföy Değeri", f"{info.get('balance', 'N/A'):,.2f} TL" if isinstance(info.get('balance'), (int, float)) else "N/A"],
                ["Nakit Bakiye", f"{self.balance:,.2f} TL" if hasattr(self, 'balance') else "N/A"],
                ["Pozisyon", f"{info.get('position', 'N/A'):,} adet" if isinstance(info.get('position'), (int, float)) else "N/A"],
                ["Toplam Kâr/Zarar", f"{Fore.GREEN if info.get('profit', 0) > 0 else Fore.RED}{info.get('profit', 'N/A'):,.2f} TL{Style.RESET_ALL}" if isinstance(info.get('profit'), (int, float)) else "N/A"],
                ["Stop-Loss", f"{info.get('stop_loss', 'N/A'):,.2f} TL" if isinstance(info.get('stop_loss'), (int, float)) else "N/A"],
                ["Take-Profit", f"{info.get('take_profit', 'N/A'):,.2f} TL" if isinstance(info.get('take_profit'), (int, float)) else "N/A"],
                ["Aksiyon", f"{info.get('action', 'N/A')}"],
                ["Ödül", f"{Fore.GREEN if info.get('reward', 0) > 0 else Fore.RED}{info.get('reward', 'N/A'):,.2f}{Style.RESET_ALL}" if isinstance(info.get('reward'), (int, float)) else "N/A"],
                ["Robot Kararı", f"{robot_decision}"]
            ], tablefmt="fancy_grid")

            metrics = self.get_performance_metrics()
            metrics_table = tabulate([(k, f"{v:.2f}" if isinstance(v, (int, float)) and not np.isnan(v) else "N/A") for k, v in metrics.items()], tablefmt="fancy_grid")

            signal_table = self.print_signal_summary_table(info.get('step', 0))

            # Tabloları yan yana yerleştir
            self._print_tables_side_by_side([step_table, metrics_table, signal_table])

            market_trend = self._calculate_market_trend()
            if market_trend == 1:
                trend_desc = f"{Fore.GREEN}📈 Yükseliş{Style.RESET_ALL}"
            elif market_trend == -1:
                trend_desc = f"{Fore.RED}📉 Düşüş{Style.RESET_ALL}"
            else:
                trend_desc = "➖ Yatay"

            print(f"\n📊 Market Trend: {trend_desc}")
            print(f"{'='*150}\n")
        except Exception as e:
            print(f"Error in _print_step_info: {e}")

    def print_signal_summary_table(self, step):
        try:
            # Genel sinyal, güç ve bireysel sinyalleri al
            general_signal, strength, individual_signals = self.advanced_system.generate_combined_signal(step)
            
            # Trend tahmini al
            trend_prediction = self.advanced_system.predict_trend(step)
            
            # Tablo verilerini oluştur
            table_data = [
                # Genel sinyal bilgisi
                ["General Signal", self.advanced_system.colorize_value(general_signal), "N/A", "N/A", "N/A"],
                # Sinyal gücü
                ["Signal Strength", f"{strength:.2f}", "N/A", "N/A", "N/A"],
                # Trend tahmini
                ["Trend Prediction", f"{trend_prediction:.2%}", "N/A", "N/A", "N/A"]
            ]
            
            # Her bir sinyal için bilgileri ekle
            for signal, value in individual_signals.items():
                # Sinyalin ağırlığını al
                weight = self.advanced_system.weights.get(signal, 'N/A')
                
                # Doğru ve yanlış tahmin sayılarını al
                correct = self.advanced_system.signal_performance[signal]['correct']
                incorrect = self.advanced_system.signal_performance[signal]['incorrect']
                
                # Toplam tahmin sayısını hesapla
                total = correct + incorrect
                
                # Doğruluk oranını hesapla (sıfıra bölme hatasını önle)
                accuracy = correct / total if total > 0 else 0
                
                # Sinyal bilgilerini tabloya ekle
                table_data.append([
                    signal,  # Sinyal adı
                    self.advanced_system.colorize_value(value),  # Renklendirilen sinyal değeri
                    f"{weight:.4f}" if isinstance(weight, (int, float)) else weight,  # Ağırlık
                    f"{correct}/{incorrect}",  # Doğru/Yanlış tahmin sayıları
                    f"{accuracy:.2%}"  # Doğruluk oranı
                ])
            
            # Tablo başlıklarını tanımla
            headers = ["Indicator", "Signal", "Weight", "Correct/Incorrect", "Accuracy"]
            
            # Tabulate kullanarak güzel bir tablo oluştur
            table = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
            
            return table

        except Exception as e:
            # Hata durumunda kullanıcıyı bilgilendir
            print(f"Error in print_signal_summary_table: {e}")
            return "Error generating signal summary table"
        
    def _print_tables_side_by_side(self, tables):
        try:
            lines = [table.split('\n') for table in tables]
            max_lines = max(len(table) for table in lines)
            padded_lines = [table + [''] * (max_lines - len(table)) for table in lines]
            for row in zip(*padded_lines):
                print('    '.join(row))
        except Exception as e:
            print(f"Error in _print_tables_side_by_side: {e}")

    def _log_step_info(self, info):
        log_message = (
            f"Adım: {info['step']} | "
            f"Fiyat: {info['price']:.2f} TL | "
            f"Portföy: {info['balance']:.2f} TL | "
            f"Pozisyon: {info['position']} | "
            f"Kâr/Zarar: {info['profit']:.2f} TL | "
            f"Stop-Loss: {info['stop_loss']:.2f} TL | "
            f"Take-Profit: {info['take_profit']:.2f} TL | "
            f"Aksiyon: {info['action']} | "
            f"Ödül: {info['reward']:.2f}"
        )
        logging.info(log_message)

    def _is_in_downtrend(self):
        current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
        ma_50 = self.data['Kapanış(TL)'].rolling(window=50).mean().iloc[self.current_step]
        return current_price < ma_50

    def _calculate_buy_amount(self):
        price_change = self.data['Kapanış(TL)'].pct_change(periods=10).iloc[self.current_step]
        ma_50 = self.data['Kapanış(TL)'].rolling(window=50).mean().iloc[self.current_step]
        current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
        volatility = self.data['Kapanış(TL)'].pct_change().rolling(window=20).std().iloc[self.current_step]
        trend = 1 if current_price > ma_50 else (-1 if current_price < ma_50 else 0)
        portfolio_value = self.balance + (self.position * current_price)
        
        base_buy_percentage = 0.9

        if trend == 1:
            base_buy_percentage *= 1.3
        elif trend == -1:
            base_buy_percentage *= 0.9

        if price_change < -0.05:
            base_buy_percentage *= 1.5
        elif price_change < -0.03:
            base_buy_percentage *= 1.2

        if volatility > 0.03:
            base_buy_percentage *= 1.1

        max_buy_percentage = 0.9
        buy_percentage = min(base_buy_percentage, max_buy_percentage)
        
        max_risk_amount = portfolio_value * 0.9
        buy_amount = min(self.balance * buy_percentage, max_risk_amount)

        min_alim_miktari = 5000
        buy_amount = max(buy_amount, min_alim_miktari)
        
        return buy_amount / self.balance

    def _set_dynamic_stop_loss_take_profit(self, price):
        atr_value = self._calculate_atr()
        volatility = self.data['Kapanış(TL)'].pct_change().rolling(window=20).std().iloc[self.current_step]
        
        if volatility > 0.03:
            self.stop_loss = price - (4 * atr_value)
            self.take_profit = price + (5 * atr_value)
        else:
            self.stop_loss = price - (3 * atr_value)
            self.take_profit = price + (4 * atr_value)
        
        if price > self.highest_price:
            self.highest_price = price
            print(f"\n{Fore.GREEN}📈 Yeni En Yüksek Fiyat: {price:,.2f} TL. Stop-loss güncellendi.{Style.RESET_ALL}")
        
        if price <= self.highest_price * 0.95 and self.position > 0:
            print(f"\n{Fore.RED}⚠️ Fiyat en yüksek seviyeden %5 düştü. Pozisyonun %50'sini satıyoruz.{Style.RESET_ALL}")
            self._close_partial_position(price, sell_percentage=0.5)
        
        if price <= self.highest_price * 0.90 and self.position > 0:
            print(f"\n{Fore.RED}⚠️ Fiyat en yüksek seviyeden %10 düştü. Kalan tüm pozisyonu kapatıyoruz.{Style.RESET_ALL}")
            self._close_position(price)
        
        print(f"\n{Fore.YELLOW}📊 Güncel Stop-Loss ve Take-Profit Seviyeleri{Style.RESET_ALL}")
        print(tabulate([
            ["En Yüksek Fiyat", f"{self.highest_price:,.2f} TL"],
            ["Yeni Stop-Loss", f"{self.stop_loss:,.2f} TL"],
            ["Yeni Take-Profit", f"{self.take_profit:,.2f} TL"]
        ], tablefmt="fancy_grid"))

    def _close_partial_position(self, price, sell_percentage=0.5):
        shares_to_sell = int(self.position * sell_percentage)
        sell_value = shares_to_sell * price
        self.balance += sell_value
        self.position -= shares_to_sell
        
        profit = sell_value - (shares_to_sell * self.entry_price)
        
        print(f"\n{Fore.RED}💰 KISMİ SATIŞ GERÇEKLEŞTİ{Style.RESET_ALL}")
        print(tabulate([
            ["Satılan Hisse Adedi", f"{shares_to_sell:,}"],
            ["Satış Fiyatı", f"{price:,.2f} TL"],
            ["Toplam Değer", f"{sell_value:,.2f} TL"],
            ["Kâr/Zarar", f"{profit:,.2f} TL"],
            ["Yeni Nakit Bakiye", f"{self.balance:,.2f} TL"]
        ], tablefmt="fancy_grid"))

    def _calculate_atr(self, window=14):
        high = self.data['Max(TL)']
        low = self.data['Min(TL)']
        close = self.data['Kapanış(TL)']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr.iloc[self.current_step]

    def calculate_win_loss_ratio(self):
        wins = sum(1 for t in self.trades if t['type'] == 'sell' and t['price'] > t['price'])
        losses = sum(1 for t in self.trades if t['type'] == 'sell' and t['price'] <= t['price'])
        return wins / losses if losses > 0 else float('inf')

    def calculate_profit_factor(self):
        gross_profit = sum(t['price'] * t['shares'] - t['price'] * t['shares'] for t in self.trades if t['type'] == 'sell' and t['price'] > t['price'])
        gross_loss = sum(t['price'] * t['shares'] - t['price'] * t['shares'] for t in self.trades if t['type'] == 'sell' and t['price'] <= t['price'])
        return gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

    def get_performance_metrics(self):
        return {
            "Win/Loss Ratio": self.calculate_win_loss_ratio(),
            "Profit Factor": self.calculate_profit_factor()
        }

    def print_performance_summary(self):
        metrics = self.get_performance_metrics()
        print(f"\n{Fore.CYAN}📈 PERFORMANS ÖZETI{Style.RESET_ALL}")
        print(tabulate([(k, f"{v:.2f}") for k, v in metrics.items()], tablefmt="fancy_grid"))

    def get_state_size(self):
        return self._next_observation().shape[1]

    def get_action_size(self):
        return self.action_space
    
    def update_graph(self):
        """
        GraphVisualizer sınıfını kullanarak grafikleri günceller.
        """
        self.graph_visualizer.update_graphs(self.data, self.current_step, self.balance, self.position)

    def print_graphs(self):
        """
        GraphVisualizer sınıfını kullanarak grafikleri yan yana yazdırır.
        """
        self.graph_visualizer.print_charts_side_by_side()

    def _calculate_reward(self, prev_value, current_value):
        base_reward = (current_value - prev_value) / prev_value
        
        trend = self._calculate_market_trend()
        if (trend > 0 and self.position > 0) or (trend < 0 and self.position == 0):
            base_reward *= 1.2
        elif (trend < 0 and self.position > 0) or (trend > 0 and self.position == 0):
            base_reward *= 0.8
        
        if self.position > 0:
            current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
            if current_price <= self.stop_loss:
                base_reward -= 0.1
            elif current_price >= self.take_profit:
                base_reward += 0.1

        return self.reward_calculator.calculate_total_reward(prev_value, current_value)


    def _check_stop_loss_take_profit(self, current_price):
        if current_price <= self.stop_loss:
            self._close_position(current_price)
            print(f"\n{Fore.RED}🛑 Stop-Loss Tetiklendi{Style.RESET_ALL}")
        elif current_price >= self.take_profit:
            self._close_position(current_price)
            print(f"\n{Fore.GREEN}🎯 Take-Profit Tetiklendi{Style.RESET_ALL}")

    def render(self, mode='human'):
        """
        Ortamın mevcut durumunu görselleştirir.
        """
        if mode == 'human':
            self._print_step_info({
                'step': self.current_step,
                'price': self.data['Kapanış(TL)'].iloc[self.current_step],
                'balance': self.balance + self.position * self.data['Kapanış(TL)'].iloc[self.current_step],
                'position': self.position,
                'profit': self.total_profit,
                'action': None,  # Bu bilgi render sırasında mevcut değil
                'reward': None,  # Bu bilgi render sırasında mevcut değil
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            })
            self.print_graphs()
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented.")

    def close(self):
        """
        Ortamı kapatır ve gerekli temizlik işlemlerini yapar.
        """
        # Gerekli temizlik işlemleri burada yapılabilir
        pass