import pandas as pd
import numpy as np
from stock_env import StockTradingEnv
from dqn_agent import DQNAgent
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import logging
import tempfile

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """
    Excel dosyasından veriyi yükler, tarih sütununu işleyip indeks olarak ayarlar ve boş alanları 0 ile doldurur.
    """
    try:
        data = pd.read_excel(file_path)
        logger.info(f"Veri sütunları: {data.columns}")
        
        data['Tarih'] = pd.to_datetime(data['Tarih'], format='%d-%m-%Y')
        data.set_index('Tarih', inplace=True)
        data = data.sort_index()
        
        # Boş alanları 0 ile doldur
        data = data.fillna(0)
        
        return data
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        return None

def log_profit(epoch, total_reward, initial_balance, session_dir):
    """
    Her epoch sonunda kazanılan miktarı bir txt dosyasına yazar.
    """
    profit = total_reward - initial_balance
    log_file = os.path.join(session_dir, 'profit_log.txt')
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch+1}: Profit = {profit:.2f}\n")
    logger.info(f"Profit logged to {log_file}")

def train_model(train_env, agent, epochs, batch_size, initial_balance, session_dir):
    """
    Modeli eğitir ve eğitim sürecini yönetir.
    """
    for epoch in range(agent.current_epoch, epochs):
        state = train_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        logger.info(f"Epoch: {epoch+1}/{epochs}, Total Reward: {total_reward:.2f}")
        
        agent.save_best_model(total_reward, epoch)
        
        if (epoch + 1) % 2 == 0:  # Her 2 epoch'ta bir checkpoint kaydet
            agent.save_checkpoint()
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Her epoch sonunda kazanılan miktarı log dosyasına yaz
        log_profit(epoch, total_reward, initial_balance, session_dir)
        
        # Epsilon değerini güncelle
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

def test_model(test_env, agent):
    """
    Eğitilmiş modeli test eder ve sonuçları döndürür.
    """
    state = test_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)
        state = next_state
        total_reward += reward
    
    return total_reward

def main():
    # Benzersiz bir oturum kimliği oluştur
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Kullanıcının ev dizininde bir klasör oluştur
    base_dir = os.path.expanduser("~/stock_trading_rl")
    session_dir = os.path.join(base_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Loglama ayarlarını güncelle
    log_file = os.path.join(session_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting main function. Session ID: {session_id}")
    logger.info(f"Logs will be saved to: {log_file}")
    
    # Veriyi yükle ve ön işle
    data = load_and_preprocess_data('data_aselsan.xlsx')
    if data is None:
        logger.error("Veri yüklenemedi. Program sonlandırılıyor.")
        return
    logger.info("Data loaded successfully")
    
    # Veriyi eğitim ve test setlerine ayır
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    
    # Başlangıç bakiyesi
    initial_balance = 100000
    
    # Eğitim ortamını oluştur
    train_env = StockTradingEnv(train_data, initial_balance=initial_balance)
    logger.info("Training environment created")

    # DQN ajanını oluştur
    state_size = train_env.get_state_size()
    action_size = 21  # 0: Çok Sat, 1-10: Tut, 11-20: Al (farklı miktarlarda)
    agent = DQNAgent(state_size, action_size, session_dir)
    logger.info("DQN agent created")
    
    # En son checkpointten devam et
    agent.load_model()
    logger.info(f"Resuming from epoch {agent.current_epoch}")
    
    # Eğitim parametreleri
    epochs = 100
    batch_size = 32
    
    # Modeli eğit
    train_model(train_env, agent, epochs, batch_size, initial_balance, session_dir)
    logger.info("Training completed")
    
    # Eğitim sonunda son modeli kaydet
    final_model_path = os.path.join(session_dir, "final_model.h5")
    agent.model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Performans metriklerini hesapla ve yazdır
    train_env.print_performance_summary()
    
    # Test aşaması
    test_env = StockTradingEnv(test_data, initial_balance=initial_balance)
    test_reward = test_model(test_env, agent)
    logger.info(f"Test Total Reward: {test_reward:.2f}")
    
    # Test performans metriklerini hesapla ve yazdır
    test_env.print_performance_summary()
    
    logger.info("Main function completed successfully")
    logger.info(f"All session data saved in: {session_dir}")

if __name__ == "__main__":
    main()