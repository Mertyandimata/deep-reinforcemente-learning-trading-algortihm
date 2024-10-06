import datetime
import numpy as np
import random
from collections import deque
from tensorflow import keras
from keras.layers import Input, Dense, Lambda, Add, Subtract
import tensorflow as tf
import os
import json
import logging
from tqdm import tqdm

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DQNAgent:
    def __init__(self, state_size, action_size, session_dir):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.update_target_frequency = 10
        self.model = self._build_dueling_model()
        self.target_model = self._build_dueling_model()
        self.update_target_model()
        self.best_reward = float('-inf')
        self.session_dir = session_dir
        self.best_model_path = os.path.join(session_dir, 'best_model.weights.h5')
        self.checkpoint_path = os.path.join(session_dir, 'checkpoint.weights.h5')
        self.metadata_path = os.path.join(session_dir, 'training_metadata.json')
        self.current_epoch = 0
        self.total_steps = 0
        self.episode_rewards = []


    def _build_dueling_model(self):
        input_layer = Input(shape=(self.state_size,))
        
        main_features = Dense(128, activation='relu')(input_layer)
        main_features = Dense(128, activation='relu')(main_features)
        
        value_fc = Dense(64, activation='relu')(main_features)
        value = Dense(1, activation='linear')(value_fc)

        advantage_fc = Dense(64, activation='relu')(main_features)
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)

        mean_advantage = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        q_values = Add()([value, Subtract()([advantage, mean_advantage])])

        model = keras.models.Model(inputs=input_layer, outputs=q_values)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)

    def save_best_model(self, total_reward, epoch):
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            try:
                self.model.save_weights(self.best_model_path)
                print(f"New best model saved at epoch {epoch} with total reward: {total_reward}")
                return True
            except Exception as e:
                print(f"Error saving best model: {e}")
        return False

    def save_checkpoint(self):
        try:
            self.model.save_weights(self.checkpoint_path)
            metadata = {
                'current_epoch': self.current_epoch,
                'epsilon': self.epsilon,
                'best_reward': self.best_reward,
                'total_steps': self.total_steps,
                'episode_rewards': self.episode_rewards
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"Checkpoint saved at epoch {self.current_epoch}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
   
    def load_model(self):
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.current_epoch = metadata['current_epoch']
                self.epsilon = metadata['epsilon']
                self.best_reward = metadata['best_reward']
                self.total_steps = metadata.get('total_steps', 0)
                self.episode_rewards = metadata.get('episode_rewards', [])
                
            if os.path.exists(self.best_model_path):
                self.model.load_weights(self.best_model_path)
                logger.info(f"Loaded best model weights. Continuing from epoch {self.current_epoch + 1}")
            elif os.path.exists(self.checkpoint_path):
                self.model.load_weights(self.checkpoint_path)
                logger.info(f"Loaded checkpoint weights. Continuing from epoch {self.current_epoch + 1}")
            else:
                logger.info("No existing model weights found, starting from scratch")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Starting from scratch")

def train_agent(env, agent, epochs, batch_size, eval_frequency=10):
    best_eval_reward = float('-inf')
    
    for epoch in tqdm(range(agent.current_epoch, epochs), desc="Training Progress"):
        agent.current_epoch = epoch
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.total_steps += 1

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        agent.episode_rewards.append(total_reward)
        
        if epoch % eval_frequency == 0:
            eval_reward = evaluate_agent(env, agent)
            logger.info(f"Epoch: {epoch+1}/{epochs}, Train Reward: {total_reward:.2f}, Eval Reward: {eval_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_best_model(eval_reward, epoch)
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (epoch + 1) % agent.update_target_frequency == 0:
            agent.update_target_model()
            logger.info("Target model updated.")

        agent.save_checkpoint()

    logger.info("Training completed!")

    def evaluate_agent(env, agent, n_episodes=5):
        total_rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])
            done = False
            episode_reward = 0
            while not done:
                action = np.argmax(agent.model.predict(state, verbose=0)[0])
                next_state, reward, done, _ = env.step(action)
                state = np.reshape(next_state, [1, agent.state_size])
                episode_reward += reward
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)
    
    def plot_training_results(agent):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(agent.episode_rewards)
        plt.title('Training Rewards over Time')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(os.path.join(agent.session_dir, 'training_rewards.png'))
        plt.close()

        # Plot moving average
        window_size = 100
        moving_avg = np.convolve(agent.episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.figure(figsize=(12, 6))
        plt.plot(moving_avg)
        plt.title(f'Moving Average of Training Rewards (Window Size: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.savefig(os.path.join(agent.session_dir, 'training_rewards_moving_avg.png'))
        plt.close()


if __name__ == "__main__":
    from stock_env import StockTradingEnv
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Veri yükleme ve ön işleme
    data = pd.read_excel('data_aselsan.xlsx')
    data['Tarih'] = pd.to_datetime(data['Tarih'], format='%d-%m-%Y')
    data.set_index('Tarih', inplace=True)
    data = data.sort_index()

    # Veriyi eğitim ve test setlerine ayırma
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    # Eğitim ortamını oluşturma
    env = StockTradingEnv(train_data, initial_balance=100000)

    # Oturum dizini oluşturma
    session_dir = os.path.join(os.path.expanduser("~"), "stock_trading_rl", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    # Loglama için dosya handler'ı ekle
    file_handler = logging.FileHandler(os.path.join(session_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # DQN ajanını oluşturma
    state_size = env.get_state_size()
    action_size = 21
    agent = DQNAgent(state_size, action_size, session_dir)

    # Mevcut modeli yükleme (varsa)
    agent.load_model()

    # Eğitim parametreleri
    epochs = 1000
    batch_size = 32

    # Ajanı eğitme
    train_agent(env, agent, epochs, batch_size)

    # Eğitim sonuçlarını görselleştir
    plot_training_results(agent)

    # Test aşaması
    test_env = StockTradingEnv(test_data, initial_balance=100000)
    test_reward = evaluate_agent(test_env, agent, n_episodes=1)
    logger.info(f"Test Reward: {test_reward:.2f}")

    logger.info(f"Training completed. All session data saved in: {session_dir}")