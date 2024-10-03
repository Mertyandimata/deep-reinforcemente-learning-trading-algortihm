import pandas as pd
import numpy as np

class OrderBlock:
    def __init__(self, top, bottom, location, is_bullish):
        self.top = top
        self.bottom = bottom
        self.location = location
        self.is_bullish = is_bullish
        self.is_breaker = False
        self.break_location = None

class OrderBlockDetector:
    def __init__(self, data, lookback=10, use_body=False):
        self.data = data
        self.lookback = lookback
        self.use_body = use_body
        self.bullish_obs = []
        self.bearish_obs = []

    def detect_swings(self):
        highs = self.data['Max(TL)'].rolling(window=self.lookback).max()
        lows = self.data['Min(TL)'].rolling(window=self.lookback).min()
        
        swing_highs = (self.data['Max(TL)'] == highs) & (self.data['Max(TL)'].shift(1) < highs)
        swing_lows = (self.data['Min(TL)'] == lows) & (self.data['Min(TL)'].shift(1) > lows)
        
        return swing_highs, swing_lows

    def detect_order_blocks(self):
        swing_highs, swing_lows = self.detect_swings()
        
        for i in range(len(self.data)):
            if swing_highs.iloc[i]:
                self.detect_bearish_ob(i)
            elif swing_lows.iloc[i]:
                self.detect_bullish_ob(i)

        self.update_breaker_blocks()

    def detect_bullish_ob(self, index):
        for j in range(index-1, max(0, index-self.lookback), -1):
            if self.use_body:
                low = min(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
                high = max(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
            else:
                low = self.data['Min(TL)'].iloc[j]
                high = self.data['Max(TL)'].iloc[j]
            
            if low < self.data['Min(TL)'].iloc[index]:
                ob = OrderBlock(high, low, j, True)
                self.bullish_obs.append(ob)
                break

    def detect_bearish_ob(self, index):
        for j in range(index-1, max(0, index-self.lookback), -1):
            if self.use_body:
                low = min(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
                high = max(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
            else:
                low = self.data['Min(TL)'].iloc[j]
                high = self.data['Max(TL)'].iloc[j]
            
            if high > self.data['Max(TL)'].iloc[index]:
                ob = OrderBlock(high, low, j, False)
                self.bearish_obs.append(ob)
                break

    def update_breaker_blocks(self):
        current_price = self.data['Kapanış(TL)'].iloc[-1]
        
        for ob in self.bullish_obs:
            if not ob.is_breaker and current_price < ob.bottom:
                ob.is_breaker = True
                ob.break_location = len(self.data) - 1
            elif ob.is_breaker and current_price > ob.top:
                self.bullish_obs.remove(ob)
        
        for ob in self.bearish_obs:
            if not ob.is_breaker and current_price > ob.top:
                ob.is_breaker = True
                ob.break_location = len(self.data) - 1
            elif ob.is_breaker and current_price < ob.bottom:
                self.bearish_obs.remove(ob)

    def get_active_order_blocks(self, num_bull=3, num_bear=3):
        return self.bullish_obs[-num_bull:], self.bearish_obs[-num_bear:]

def analyze_order_blocks(data, lookback=10, use_body=False, num_bull=3, num_bear=3):
    detector = OrderBlockDetector(data, lookback, use_body)
    detector.detect_order_blocks()
    return detector.get_active_order_blocks(num_bull, num_bear)