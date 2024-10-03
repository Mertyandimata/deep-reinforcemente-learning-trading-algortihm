import numpy as np
from signal_generator import SignalGenerator
class StockTradingEnv:
    def __init__(self, data, initial_balance=10000, bullish_obs=None, bearish_obs=None):
        self.data = data
        self.initial_balance = initial_balance
        self.signal_generator = SignalGenerator(data)
        self.bullish_obs = bullish_obs or []
        self.bearish_obs = bearish_obs or []
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        window = self.data.iloc[max(0, self.current_step-30):self.current_step+1]
        signals = self.signal_generator.get_signals(self.current_step)
        
        obs = np.array([
            self.data['Kapanış(TL)'].iloc[self.current_step],
            self.data['USDTRY'].iloc[self.current_step],
            self.data['BIST 100'].iloc[self.current_step],
            self.data['PiyasaDeğeri(mn TL)'].iloc[self.current_step],
            window['Kapanış(TL)'].pct_change().mean(),
            window['Kapanış(TL)'].pct_change().std(),
            (self.data['Kapanış(TL)'].iloc[self.current_step] - window['Kapanış(TL)'].mean()) / window['Kapanış(TL)'].std(),
            self._get_order_block_feature(),
        ] + list(signals.values()))
        
        return obs

    def _get_order_block_feature(self):
        current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
        
        in_bullish_ob = any(ob.bottom <= current_price <= ob.top for ob in self.bullish_obs)
        in_bearish_ob = any(ob.bottom <= current_price <= ob.top for ob in self.bearish_obs)
        
        if in_bullish_ob:
            return 1
        elif in_bearish_ob:
            return -1
        else:
            return 0

    def step(self, action):
        prev_value = self.balance + self.position * self.data['Kapanış(TL)'].iloc[self.current_step]
        
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._next_observation()
        
        current_value = self.balance + self.position * self.data['Kapanış(TL)'].iloc[self.current_step]
        reward = (current_value - prev_value) / prev_value
        self.total_profit += current_value - prev_value
        
        info = {
            'step': self.current_step,
            'balance': current_value,
            'profit': self.total_profit,
            'reward': reward
        }
        
        print(f"Adım: {self.current_step}, Bakiye: {current_value:.2f} TL, Kâr/Zarar: {self.total_profit:.2f} TL, Ödül: {reward:.4f}")
        
        return obs, reward, done, info

    def _take_action(self, action):
        current_price = self.data['Kapanış(TL)'].iloc[self.current_step]
        if action == 0:  # Sat
            self.balance += self.position * current_price
            self.position = 0
            print(f"SATIŞ: {self.position} adet hisse satıldı, yeni bakiye: {self.balance:.2f} TL")
        elif action == 2:  # Al
            shares_to_buy = self.balance // current_price
            self.position += shares_to_buy
            self.balance -= shares_to_buy * current_price
            print(f"ALIŞ: {shares_to_buy} adet hisse alındı, yeni bakiye: {self.balance:.2f} TL")
