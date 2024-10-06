import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class OrderBlockType(Enum):
    BULLISH = 1
    BEARISH = -1

@dataclass
class OrderBlock:
    top: float
    bottom: float
    location: int
    ob_type: OrderBlockType
    strength: float = 0.0
    is_breaker: bool = False
    break_location: Optional[int] = None
    volume: float = 0.0
    atr: float = 0.0

    def calculate_size(self) -> float:
        return self.top - self.bottom

    def is_active(self, current_price: float) -> bool:
        if self.ob_type == OrderBlockType.BULLISH:
            return current_price >= self.bottom
        return current_price <= self.top

class OrderBlockDetector:
    def __init__(self, data: pd.DataFrame, lookback: int = 10, use_body: bool = False, atr_period: int = 14):
        self.data = data
        self.lookback = lookback
        self.use_body = use_body
        self.atr_period = atr_period
        self.order_blocks: List[OrderBlock] = []
        self.calculate_atr()

    def calculate_atr(self):
        high = self.data['Max(TL)']
        low = self.data['Min(TL)']
        close = self.data['Kapanış(TL)']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(window=self.atr_period).mean()

    def detect_swings(self) -> Tuple[pd.Series, pd.Series]:
        highs = self.data['Max(TL)'].rolling(window=self.lookback).max()
        lows = self.data['Min(TL)'].rolling(window=self.lookback).min()
        
        swing_highs = (self.data['Max(TL)'] == highs) & (self.data['Max(TL)'].shift(1) < highs)
        swing_lows = (self.data['Min(TL)'] == lows) & (self.data['Min(TL)'].shift(1) > lows)
        
        return swing_highs, swing_lows

    def detect_order_blocks(self):
        swing_highs, swing_lows = self.detect_swings()
        self.order_blocks.clear()
        
        for i in range(len(self.data)):
            if swing_highs.iloc[i]:
                self.detect_bearish_ob(i)
            elif swing_lows.iloc[i]:
                self.detect_bullish_ob(i)
        
        self.update_order_blocks()

    def detect_bullish_ob(self, index: int):
        for j in range(index-1, max(0, index-self.lookback), -1):
            if self.use_body:
                low = min(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
                high = max(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
            else:
                low = self.data['Min(TL)'].iloc[j]
                high = self.data['Max(TL)'].iloc[j]
            
            if low < self.data['Min(TL)'].iloc[index]:
                ob = OrderBlock(
                    top=high,
                    bottom=low,
                    location=j,
                    ob_type=OrderBlockType.BULLISH,
                    volume=self.data['Hacim(TL)'].iloc[j],
                    atr=self.data['ATR'].iloc[j]
                )
                ob.strength = self.calculate_ob_strength(ob, index)
                self.order_blocks.append(ob)
                break

    def detect_bearish_ob(self, index: int):
        for j in range(index-1, max(0, index-self.lookback), -1):
            if self.use_body:
                low = min(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
                high = max(self.data['AOF(TL)'].iloc[j], self.data['Kapanış(TL)'].iloc[j])
            else:
                low = self.data['Min(TL)'].iloc[j]
                high = self.data['Max(TL)'].iloc[j]
            
            if high > self.data['Max(TL)'].iloc[index]:
                ob = OrderBlock(
                    top=high,
                    bottom=low,
                    location=j,
                    ob_type=OrderBlockType.BEARISH,
                    volume=self.data['Hacim(TL)'].iloc[j],
                    atr=self.data['ATR'].iloc[j]
                )
                ob.strength = self.calculate_ob_strength(ob, index)
                self.order_blocks.append(ob)
                break

    def calculate_ob_strength(self, ob: OrderBlock, swing_index: int) -> float:
        ob_size = ob.calculate_size()
        price_move = abs(self.data['Kapanış(TL)'].iloc[swing_index] - self.data['Kapanış(TL)'].iloc[ob.location])
        volume_change = self.data['Hacim(TL)'].iloc[swing_index] / ob.volume
        
        size_factor = ob_size / ob.atr
        move_factor = price_move / ob.atr
        volume_factor = min(volume_change, 2.0)  # Volume factor capped at 2.0
        
        strength = (size_factor * 0.3 + move_factor * 0.5 + volume_factor * 0.2)
        return min(strength, 1.0)  # Normalize strength to be between 0 and 1

    def update_order_blocks(self):
        current_price = self.data['Kapanış(TL)'].iloc[-1]
        
        for ob in self.order_blocks:
            if not ob.is_breaker:
                if (ob.ob_type == OrderBlockType.BULLISH and current_price < ob.bottom) or \
                (ob.ob_type == OrderBlockType.BEARISH and current_price > ob.top):
                    ob.is_breaker = True
                    ob.break_location = len(self.data) - 1
                    ob.strength *= 0.5  # Kırıldığında sinyal gücünü azalt
            elif not ob.is_active(current_price):
                self.order_blocks.remove(ob)  # Aktif olmayan OB'yi kaldır


    def get_active_order_blocks(self, num_blocks: int = 3) -> Tuple[List[OrderBlock], List[OrderBlock]]:
        active_obs = [ob for ob in self.order_blocks if not ob.is_breaker]
        active_obs.sort(key=lambda x: x.strength, reverse=True)
        bullish_obs = [ob for ob in active_obs if ob.ob_type == OrderBlockType.BULLISH]
        bearish_obs = [ob for ob in active_obs if ob.ob_type == OrderBlockType.BEARISH]
        return bullish_obs[:num_blocks], bearish_obs[:num_blocks]

    def generate_ob_signal(self) -> pd.Series:
        signal = pd.Series(0, index=self.data.index)
        
        # Tüm lokasyonları tarayıp aktif order block olup olmadığını kontrol et
        for i in range(len(self.data)):
            current_price = self.data['Kapanış(TL)'].iloc[i]
            
            for ob in self.order_blocks:
                if ob.is_active(current_price):
                    signal.iloc[i] = ob.strength * ob.ob_type.value
                    
        return signal
