import pandas as pd
import numpy as np
import random

# Load data
data = pd.read_csv('spy.csv')
data['Close'] = data['Close'].fillna(method='ffill')  # Forward fill NaN if any

# Hyperparameters
n_episodes = 1000        # Number of training episodes
learning_rate = 0.1      # Learning rate for Q-learning
discount_factor = 0.95   # Discount factor for future rewards
epsilon = 1.0            # Exploration rate
epsilon_decay = 0.995    # Decay rate for exploration
min_epsilon = 0.01       # Minimum exploration rate

# Define actions
actions = ['hold', 'buy', 'sell']

# Initialize Q-table with zeros
state_space_size = len(data) - 1
q_table = np.zeros((state_space_size, len(actions)))

# Define the reward function
def get_reward(action, state, next_state):
    price_now = data['Close'].iloc[state]
    price_next = data['Close'].iloc[next_state]
    if action == 'buy':
        return price_next - price_now
    elif action == 'sell':
        return price_now - price_next
    return 0  # For hold

# Train the agent
for episode in range(n_episodes):
    state = 0  # Start each episode at the beginning
    total_reward = 0
    while state < state_space_size - 1:
        # Choose action using epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action_index = random.choice([0, 1, 2])  # Explore
        else:
            action_index = np.argmax(q_table[state])  # Exploit best action
        
        # Take action and observe reward and next state
        next_state = state + 1
        reward = get_reward(actions[action_index], state, next_state)
        
        # Update Q-table
        best_future_q = np.max(q_table[next_state])
        q_table[state, action_index] = q_table[state, action_index] + \
            learning_rate * (reward + discount_factor * best_future_q - q_table[state, action_index])
        
        # Move to next state
        state = next_state
        total_reward += reward
    
    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    
    if episode % 100 == 0:
        print(f'Episode {episode} - Total Reward: {total_reward}')

# Evaluation
state = 0
balance = 1000  # Starting balance
positions = 0
for t in range(state_space_size - 1):
    action = np.argmax(q_table[t])
    current_price = data['Close'].iloc[t]
    
    if action == 1 and balance >= current_price:  # Buy
        positions += 1
        balance -= current_price
    elif action == 2 and positions > 0:           # Sell
        positions -= 1
        balance += current_price

final_balance = balance + positions * data['Close'].iloc[-1]
print(f"Final Balance: ${final_balance:.2f}")

# Buy-and-hold strategy
initial_balance = 1000
start_price = data['Close'].iloc[0]
end_price = data['Close'].iloc[-1]

# Buy 1 unit at the start price
units_held = initial_balance // start_price
final_balance_buy_and_hold = units_held * end_price

print(f"Buy-and-Hold Final Balance: ${final_balance_buy_and_hold:.2f}")
