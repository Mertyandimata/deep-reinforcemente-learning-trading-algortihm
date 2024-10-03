class NBARReversalDetector:
    def __init__(self, data, num_bars=7, min_bars_percent=50, pattern_type="All"):
        self.data = data
        self.num_bars = num_bars
        self.min_bars_percent = min_bars_percent / 100
        self.pattern_type = pattern_type

    def is_bullish_reversal(self):
        bull_low = self.data['Min(TL)'].iloc[-self.num_bars]
        bear_count = 0
        bull_reversal = True
        for i in range(1, self.num_bars):
            if self.data['Max(TL)'].iloc[-i] > self.data['Max(TL)'].iloc[-self.num_bars]:
                bull_reversal = False
                break
            bull_low = min(bull_low, self.data['Min(TL)'].iloc[-i])
            if self.data['AOF(TL)'].iloc[-i] > self.data['Kapanış(TL)'].iloc[-i]:
                bear_count += 1
        if bear_count / (self.num_bars - 1) >= self.min_bars_percent:
            bull_reversal = True
        else:
            bull_reversal = False
        bull_low = min(bull_low, self.data['Min(TL)'].iloc[-1])
        return bull_low, bull_reversal and self.data['Max(TL)'].iloc[-1] > self.data['Max(TL)'].iloc[-self.num_bars]

    def is_bearish_reversal(self):
        bear_high = self.data['Max(TL)'].iloc[-self.num_bars]
        bull_count = 0
        bear_reversal = True
        for i in range(1, self.num_bars):
            if self.data['Min(TL)'].iloc[-i] < self.data['Min(TL)'].iloc[-self.num_bars]:
                bear_reversal = False
                break
            bear_high = max(bear_high, self.data['Max(TL)'].iloc[-i])
            if self.data['AOF(TL)'].iloc[-i] < self.data['Kapanış(TL)'].iloc[-i]:
                bull_count += 1
        if bull_count / (self.num_bars - 1) >= self.min_bars_percent:
            bear_reversal = True
        else:
            bear_reversal = False
        bear_high = max(bear_high, self.data['Max(TL)'].iloc[-1])
        return bear_high, bear_reversal and self.data['Min(TL)'].iloc[-1] < self.data['Min(TL)'].iloc[-self.num_bars]

    def detect_reversals(self):
        reversals = []
        for i in range(self.num_bars, len(self.data)):
            window = self.data.iloc[i-self.num_bars:i+1]
            detector = NBARReversalDetector(window, self.num_bars, self.min_bars_percent * 100, self.pattern_type)
            
            bull_low, is_bullish = detector.is_bullish_reversal()
            bear_high, is_bearish = detector.is_bearish_reversal()
            
            if is_bullish and self.pattern_type in ["All", "Normal"] and window['Kapanış(TL)'].iloc[-1] < window['Max(TL)'].iloc[0]:
                reversals.append(("Bullish", i, "Normal"))
            elif is_bullish and self.pattern_type in ["All", "Enhanced"] and window['Kapanış(TL)'].iloc[-1] > window['Max(TL)'].iloc[0]:
                reversals.append(("Bullish", i, "Enhanced"))
            elif is_bearish and self.pattern_type in ["All", "Normal"] and window['Kapanış(TL)'].iloc[-1] > window['Min(TL)'].iloc[0]:
                reversals.append(("Bearish", i, "Normal"))
            elif is_bearish and self.pattern_type in ["All", "Enhanced"] and window['Kapanış(TL)'].iloc[-1] < window['Min(TL)'].iloc[0]:
                reversals.append(("Bearish", i, "Enhanced"))
        
        return reversals