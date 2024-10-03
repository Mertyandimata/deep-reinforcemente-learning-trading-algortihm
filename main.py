import pandas as pd
import numpy as np
from stock_env import StockTradingEnv
from dqn_agent import DQNAgent
from visualization import train_model_with_visualization
from order_blocks import analyze_order_blocks
def main():
    # Veriyi yükle
    data = pd.read_excel('data.xlsx')
    print("Veri sütunları:", data.columns)
    
    # Tarihi indeks olarak ayarla
    data['Tarih'] = pd.to_datetime(data['Tarih'], format='%d-%m-%Y')
    data.set_index('Tarih', inplace=True)
    
    # Veriyi kronolojik sıraya koy
    data = data.sort_index()
    
    # Order Blocks analizi yap
    bullish_obs, bearish_obs = [], []  # Bu kısmı Order Blocks analizinize göre güncelleyin
    
    # Order Blocks bilgilerini StockTradingEnv'e ekle
    initial_balance = 100000  # Başlangıç bakiyesi
    env = StockTradingEnv(data, initial_balance=initial_balance, bullish_obs=bullish_obs, bearish_obs=bearish_obs)
    
    # Modeli eğit ve görselleştir
    trained_agent, scores = train_model_with_visualization(data, env=env)
    
    # Test et
    print("\nTest aşaması başlıyor...")
    state = env.reset()
    state = np.reshape(state, [1, -1])
    done = False
    while not done:
        action = trained_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, -1])
    
    final_balance = env.balance + env.position * data['Kapanış(TL)'].iloc[-1]
    print(f"Final balance: {final_balance:.2f}")
    print(f"Return: {(final_balance - env.initial_balance) / env.initial_balance * 100:.2f}%")
if __name__ == "__main__":
    main()