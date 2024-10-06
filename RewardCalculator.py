import numpy as np

class RewardCalculator:
    def __init__(self, trading_env):
        self.env = trading_env
        self.previous_portfolio_value = None
        self.max_portfolio_value = 0
        self.drawdown_threshold = 0.1
        self.volatility_window = 20
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

    def calculate_total_reward(self, prev_value, current_value):
        base_reward = self.calculate_base_reward(prev_value, current_value)
        trend_reward = self.calculate_trend_reward(base_reward)
        risk_reward = self.calculate_risk_reward()
        volatility_reward = self.calculate_volatility_reward()
        drawdown_penalty = self.calculate_drawdown_penalty(current_value)
        position_reward = self.calculate_position_reward()
        sharpe_ratio_reward = self.calculate_sharpe_ratio_reward()
        
        # Yeni ödül bileşenleri
        consistency_reward = self.calculate_consistency_reward()
        recovery_speed_reward = self.calculate_recovery_speed_reward()
        market_impact_reward = self.calculate_market_impact_reward()

        total_reward = (
            base_reward * 2 +  # Temel ödülün ağırlığını artır
            trend_reward * 1.5 +  # Trend takibi önemli
            risk_reward * 2 +  # Risk yönetimine daha fazla önem ver
            volatility_reward +
            drawdown_penalty * 2 +  # Drawdown'a daha fazla ceza
            position_reward * 1.5 +  # Pozisyon yönetimine daha fazla önem ver
            sharpe_ratio_reward * 2 +  # Sharpe oranına daha fazla önem ver
            consistency_reward +
            recovery_speed_reward +
            market_impact_reward
        )

        self.log_rewards(base_reward, trend_reward, risk_reward, volatility_reward, 
                         drawdown_penalty, position_reward, sharpe_ratio_reward, 
                         consistency_reward, recovery_speed_reward, market_impact_reward, total_reward)

        return total_reward

    def calculate_base_reward(self, prev_value, current_value):
        if prev_value == 0:
            return 0
        return (current_value - prev_value) / prev_value

    def calculate_trend_reward(self, base_reward):
        market_trend = self.env._calculate_market_trend()
        if (market_trend > 0 and base_reward > 0) or (market_trend < 0 and base_reward < 0):
            return base_reward * 1.5  # Trend ile uyumlu hareket için daha büyük bonus
        else:
            return base_reward * 0.5  # Trend'e karşı hareket için daha büyük ceza

    def calculate_risk_reward(self):
        if self.env.position == 0:
            return 0

        current_price = self.env.data['Kapanış(TL)'].iloc[self.env.current_step]
        entry_price = self.env.entry_price
        stop_loss = self.env.stop_loss
        take_profit = self.env.take_profit

        if stop_loss == 0 or take_profit == 0:
            return 0

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk != 0 else 0

        if current_price >= take_profit:
            return 15 * risk_reward_ratio  # Take-profit'e ulaşıldığında daha büyük ödül
        elif current_price <= stop_loss:
            return -10 * (1 / risk_reward_ratio)  # Stop-loss'a ulaşıldığında daha büyük ceza
        else:
            return 3 * risk_reward_ratio * (current_price - entry_price) / (take_profit - entry_price)

    def calculate_volatility_reward(self):
        returns = self.env.data['Kapanış(TL)'].pct_change().iloc[max(0, self.env.current_step - self.volatility_window + 1):self.env.current_step + 1]
        volatility = returns.std()
        
        if volatility == 0:
            return 0
        
        return 5 * (1 / volatility)  # Düşük volatilite için ödül, yüksek volatilite için ceza

    def calculate_drawdown_penalty(self, current_value):
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        if drawdown > self.drawdown_threshold:
            return -20 * drawdown  # Drawdown için daha sert ceza
        return 0

    def calculate_position_reward(self):
        if self.env.position == 0:
            return 0
        
        current_price = self.env.data['Kapanış(TL)'].iloc[self.env.current_step]
        position_value = self.env.position * current_price
        portfolio_value = self.env.balance + position_value
        position_ratio = position_value / portfolio_value
        
        if position_ratio > 0.5:
            return -5 * (position_ratio - 0.5)  # Aşırı konsantrasyon için ceza
        return 0

    def calculate_sharpe_ratio_reward(self):
        if self.previous_portfolio_value is None:
            self.previous_portfolio_value = self.env.initial_balance
            return 0

        current_price = self.env.data['Kapanış(TL)'].iloc[self.env.current_step]
        current_portfolio_value = self.env.balance + (self.env.position * current_price)
        returns = (current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        
        risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate, converted to daily
        excess_return = returns - risk_free_rate
        
        if self.env.current_step < 30:  # Not enough data for reliable Sharpe ratio
            return 0
        
        historical_returns = [
            (self.env.balance + (self.env.position * self.env.data['Kapanış(TL)'].iloc[i])) /
            (self.env.balance + (self.env.position * self.env.data['Kapanış(TL)'].iloc[i-1])) - 1
            for i in range(max(0, self.env.current_step - 29), self.env.current_step + 1)
        ]
        
        returns_std = np.std(historical_returns)
        if returns_std == 0:
            return 0
        
        sharpe_ratio = (excess_return / returns_std) * np.sqrt(252)  # Annualized Sharpe ratio
        
        self.previous_portfolio_value = current_portfolio_value
        
        return 10 * sharpe_ratio  # Positive Sharpe ratio is rewarded, negative is penalized

    def calculate_consistency_reward(self):
        if self.env.trades:
            last_trade = self.env.trades[-1]
            if last_trade['type'] == 'sell':
                profit = last_trade['price'] - self.env.entry_price
                if profit > 0:
                    self.consecutive_losses = 0
                    return 5  # Tutarlı kazanç için ödül
                else:
                    self.consecutive_losses += 1
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                    return -2 * self.consecutive_losses  # Ardışık kayıplar için artan ceza
        return 0

    def calculate_recovery_speed_reward(self):
        if self.max_drawdown > 0:
            current_drawdown = self.calculate_current_drawdown()
            recovery_speed = (self.max_drawdown - current_drawdown) / self.max_drawdown
            return 10 * recovery_speed  # Hızlı toparlanma için ödül
        return 0

    def calculate_market_impact_reward(self):
        if self.env.position != 0:
            position_value = abs(self.env.position) * self.env.data['Kapanış(TL)'].iloc[self.env.current_step]
            daily_volume = self.env.data['Hacim(TL)'].iloc[self.env.current_step]
            market_impact = position_value / daily_volume
            return -10 * market_impact  # Yüksek piyasa etkisi için ceza
        return 0

    def calculate_current_drawdown(self):
        current_value = self.env.balance + self.env.position * self.env.data['Kapanış(TL)'].iloc[self.env.current_step]
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        return drawdown

    def log_rewards(self, base_reward, trend_reward, risk_reward, volatility_reward, 
                    drawdown_penalty, position_reward, sharpe_ratio_reward, 
                    consistency_reward, recovery_speed_reward, market_impact_reward, total_reward):
        print(f"📊 Ödül Bileşenleri:")
        print(f"🔹 Temel Ödül: {base_reward:.2f}")
        print(f"📈 Trend Ödülü: {trend_reward:.2f}")
        print(f"🛡️ Risk Yönetimi Ödülü: {risk_reward:.2f}")
        print(f"📊 Volatilite Ödülü: {volatility_reward:.2f}")
        print(f"📉 Drawdown Cezası: {drawdown_penalty:.2f}")
        print(f"💼 Pozisyon Ödülü: {position_reward:.2f}")
        print(f"📈 Sharpe Oranı Ödülü: {sharpe_ratio_reward:.2f}")
        print(f"🔄 Tutarlılık Ödülü: {consistency_reward:.2f}")
        print(f"🚀 Toparlanma Hızı Ödülü: {recovery_speed_reward:.2f}")
        print(f"🌊 Piyasa Etkisi Ödülü: {market_impact_reward:.2f}")
        print(f"💰 Toplam Ödül: {total_reward:.2f}")