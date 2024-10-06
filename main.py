import pandas as pd
import numpy as np
from stock_env import StockTradingEnv
from improved_dqn_agent import ImprovedDQNAgent
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import logging
import tempfile
import matplotlib.pyplot as plt

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path)
        logger.info(f"Veri sütunları: {data.columns}")
        
        data['Tarih'] = pd.to_datetime(data['Tarih'], format='%d-%m-%Y')
        data.set_index('Tarih', inplace=True)
        data = data.sort_index()
        
        data = data.fillna(0)
        
        return data
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        return None

def train_model(train_env, agent, episodes, initial_balance, session_dir):
    for episode in range(agent.total_steps // train_env.data.shape[0], episodes):
        state = train_env.reset()
        state = np.reshape(state, [1, agent.state_size])
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.total_steps += 1

            agent.replay()

            if agent.total_steps % agent.update_target_frequency == 0:
                agent.update_target_model()

        agent.episode_rewards.append(episode_reward)
        avg_reward = np.mean(agent.episode_rewards[-100:])
        agent.write_summary(episode, episode_reward, avg_reward)
        agent.save_checkpoint()
        agent.save_best_model(episode_reward)

        if episode % 10 == 0:
            logger.info(f"Episode: {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

        # Epsilon değerini güncelle
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    logger.info("Training completed")

def test_model(test_env, agent):
    state = test_env.reset()
    state = np.reshape(state, [1, agent.state_size])
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)
        state = np.reshape(next_state, [1, agent.state_size])
        total_reward += reward
    
    return total_reward

def plot_results(agent, session_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(agent.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(session_dir, 'episode_rewards.png'))
    plt.close()

    # Plot moving average
    window_size = 100
    moving_avg = np.convolve(agent.episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg)
    plt.title(f'Moving Average of Episode Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig(os.path.join(session_dir, 'moving_average_rewards.png'))
    plt.close()

def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.expanduser("~/stock_trading_rl")
    session_dir = os.path.join(base_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    log_file = os.path.join(session_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting main function. Session ID: {session_id}")
    logger.info(f"Logs will be saved to: {log_file}")
    
    data = load_and_preprocess_data('data_aselsan.xlsx')
    if data is None:
        logger.error("Veri yüklenemedi. Program sonlandırılıyor.")
        return
    logger.info("Data loaded successfully")
    
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    
    initial_balance = 100000
    
    train_env = StockTradingEnv(train_data, initial_balance=initial_balance)
    logger.info("Training environment created")

    state_size = train_env.get_state_size()
    action_size = 21
    agent = ImprovedDQNAgent(state_size, action_size, session_dir)
    logger.info("ImprovedDQNAgent created")
    
    agent.load_checkpoint()
    logger.info(f"Resuming from step {agent.total_steps}")
    
    episodes = 1000
    
    train_model(train_env, agent, episodes, initial_balance, session_dir)
    
    final_model_path = os.path.join(session_dir, "final_model.h5")
    agent.model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    plot_results(agent, session_dir)
    
    train_env.print_performance_summary()
    
    test_env = StockTradingEnv(test_data, initial_balance=initial_balance)
    test_reward = test_model(test_env, agent)
    logger.info(f"Test Total Reward: {test_reward:.2f}")
    
    test_env.print_performance_summary()
    
    logger.info("Main function completed successfully")
    logger.info(f"All session data saved in: {session_dir}")

if __name__ == "__main__":
    main()