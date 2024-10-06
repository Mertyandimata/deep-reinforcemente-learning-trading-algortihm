import numpy as np
import pandas as pd
from tabulate import tabulate

class GraphVisualizer:
    def __init__(self):
        self.ascii_charts = []

    def create_ascii_chart(self, data, width, height, title):
        """
        ASCII grafiğini oluşturur ve başlık, eksen ekler.
        """
        # NaN değerlerini kaldır
        data = [x for x in data if not np.isnan(x)]

        if not data:
            return [f"Insufficient data for {title}"] * (height + 3)

        # Veriyi normalize et
        min_val, max_val = min(data), max(data)
        if min_val == max_val:
            normalized = [0.5 for _ in data]
        else:
            normalized = [(x - min_val) / (max_val - min_val) for x in data]

        # Grafik oluştur
        chart = [['┈' for _ in range(width)] for _ in range(height)]
        for i, val in enumerate(normalized):
            chart_height = int(val * (height - 1))
            for j in range(chart_height + 1):
                chart[height - 1 - j][i] = '█'

        # Başlık ve eksenleri ekle
        result = [f"{title:^{width}}"]
        result.append('─' * width)
        result.extend(''.join(row) for row in chart)
        result.append('─' * width)
        result.append(f"{min_val:.2f}{' ' * (width - 20)}{max_val:.2f}")

        return result

    def update_graphs(self, data, current_step, balance, position, price_column='Kapanış(TL)'):
        """
        Grafiklerin güncellenmesini ve ascii tablolara eklenmesini sağlar.
        """
        start_step = max(0, current_step - 49)
        data_slice = data.iloc[start_step:current_step + 1]

        # Kapanış fiyatları ve portföy değerleri
        prices = data_slice[price_column].values
        portfolio_values = [self.calculate_portfolio_value(step, balance, position, data, price_column) 
                            for step in range(start_step, current_step + 1)]

        # Grafik boyutları
        width = 50
        height = 20

        # Fiyat ve portföy değeri grafiklerini oluştur
        price_chart = self.create_ascii_chart(prices, width, height, "Hisse Senedi Fiyatı")
        portfolio_chart = self.create_ascii_chart(portfolio_values, width, height, "Portföy Değeri")

        # Grafikleri liste olarak sakla
        self.ascii_charts = [price_chart, portfolio_chart]

    def calculate_portfolio_value(self, step, balance, position, data, price_column='Kapanış(TL)'):
        """
        Portföy değerini hesaplar.
        """
        price = data[price_column].iloc[step]
        if np.isnan(price):
            return balance
        return balance + (position * price)

    def print_charts_side_by_side(self):
        """
        Grafiklerin ASCII formatında yan yana yazdırılmasını sağlar.
        """
        if len(self.ascii_charts) < 2:
            return

        price_chart, portfolio_chart = self.ascii_charts
