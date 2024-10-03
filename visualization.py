import matplotlib.pyplot as plt
import numpy as np
from dqn_agent import DQNAgent

class VisualizationTool:
    def __init__(self, initial_balance):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 18))
        self.balance_history = [initial_balance]
        self.profit_loss_history = [0]
        self.price_history = []
        self.buy_points = []
        self.sell_points = []

        self.line1, = self.ax1.plot([], [], 'b-', label='Balance')
        self.line2, = self.ax2.plot([], [], 'g-', label='Profit/Loss')
        self.line3, = self.ax3.plot([], [], 'r-', label='Stock Price')

        self.ax1.set_title('Account Balance')
        self.ax2.set_title('Profit/Loss')
        self.ax3.set_title('Stock Price and Trading Actions')

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.legend()

    def update(self):
        self.line1.set_data(range(len(self.balance_history)), self.balance_history)
        self.line2.set_data(range(len(self.profit_loss_history)), self.profit_loss_history)
        self.line3.set_data(range(len(self.price_history)), self.price_history)

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()

        buy_x, buy_y = zip(*self.buy_points) if self.buy_points else ([], [])
        sell_x, sell_y = zip(*self.sell_points) if self.sell_points else ([], [])

        self.ax3.clear()
        self.ax3.plot(self.price_history, 'r-', label='Stock Price')
        self.ax3.plot(buy_x, buy_y, 'g^', label='Buy')
        self.ax3.plot(sell_x, sell_y, 'rv', label='Sell')
        self.ax3.legend()
        self.ax3.set_title('Stock Price and Trading Actions')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def train_model_with_visualization(data, episodes=100, batch_size=32, env=None):
    if env is None:
        env = StockTradingEnv(data)
    state_size = env._next_observation().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    vis_tool = VisualizationTool(env.initial_balance)
    plt.ion()  # Turn on interactive mode
    
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        for time in range(len(data) - 1):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            
            vis_tool.balance_history.append(info['balance'])
            vis_tool.profit_loss_history.append(info['profit'])
            vis_tool.price_history.append(data['Kapanış(TL)'].iloc[env.current_step])

            if action == 2:  # Buy
                vis_tool.buy_points.append((time, data['Kapanış(TL)'].iloc[env.current_step]))
            elif action == 0:  # Sell
                vis_tool.sell_points.append((time, data['Kapanış(TL)'].iloc[env.current_step]))
            
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 10 == 0:  # Update every 10 episodes to reduce computational load
            vis_tool.update()
            plt.pause(0.001)

    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    return agent, vis_tool.balance_history
