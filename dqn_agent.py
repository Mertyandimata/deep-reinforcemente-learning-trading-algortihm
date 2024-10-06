import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os
import json
import logging
from tqdm import tqdm

class ImprovedDQNAgent:
    def __init__(self, state_size, action_size, session_dir):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 1000
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.session_dir = session_dir
        self.checkpoint_path = os.path.join(session_dir, 'checkpoint.h5')
        self.best_model_path = os.path.join(session_dir, 'best_model.h5')
        self.metadata_path = os.path.join(session_dir, 'metadata.json')
        self.summary_path = os.path.join(session_dir, 'training_summary.txt')
        
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
        
        self.load_checkpoint()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
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

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_checkpoint(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            self.total_steps = metadata['total_steps']
            self.epsilon = metadata['epsilon']
            self.best_reward = metadata['best_reward']
            self.episode_rewards = metadata['episode_rewards']
            
            if os.path.exists(self.checkpoint_path):
                self.model.load_weights(self.checkpoint_path)
                print("Checkpoint loaded. Continuing from previous session.")
            else:
                print("No checkpoint found. Starting from scratch.")
        else:
            print("No metadata found. Starting from scratch.")

    def save_checkpoint(self):
        self.model.save_weights(self.checkpoint_path)
        metadata = {
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

    def save_best_model(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            self.model.save_weights(self.best_model_path)
            print(f"New best model saved with reward: {reward}")

    def write_summary(self, episode, reward, avg_reward):
        with open(self.summary_path, 'a') as f:
            f.write(f"Episode: {episode}, Reward: {reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}\n")

def train_agent(env, agent, episodes):
    for episode in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
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
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    print("Training finished.")

# Kullanım örneği
if __name__ == "__main__":
    from stock_env import StockTradingEnv
    import pandas as pd

    # Veri yükleme ve ön işleme
    data = pd.read_excel('data_aselsan.xlsx')
    data['Tarih'] = pd.to_datetime(data['Tarih'], format='%d-%m-%Y')
    data.set_index('Tarih', inplace=True)
    data = data.sort_index()

    # Eğitim ortamını oluşturma
    env = StockTradingEnv(data, initial_balance=100000)

    # Oturum dizini oluşturma
    session_dir = os.path.join(os.path.expanduser("~"), "stock_trading_rl", "current_session")
    os.makedirs(session_dir, exist_ok=True)

    # DQN ajanını oluşturma
    state_size = env.get_state_size()
    action_size = 21
    agent = ImprovedDQNAgent(state_size, action_size, session_dir)

    # Eğitim
    episodes = 1000
    train_agent(env, agent, episodes)