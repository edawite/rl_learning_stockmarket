import pandas as pd
import numpy as np
import random

# Load and preprocess stock data
data = pd.read_csv('spy.csv')
data['Close'] = data['Close'].fillna(method='ffill')  # Fill missing closing prices

# Feature Engineering: Add Moving Averages, RSI, and Price Change
data['MA5'] = data['Close'].rolling(window=5).mean()   # 5-day moving average
data['MA20'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).apply(
    lambda x: (x[x > 0].mean() / -x[x < 0].mean()) if x[x < 0].mean() != 0 else 1)))  # Relative Strength Index
data['Price_Change'] = data['Close'].pct_change()  # Daily price change percentage
data.fillna(0, inplace=True)  # Fill NaNs from technical indicators

# Define actions and Q-Table initialization
actions = ['hold', 'buy', 'sell']  # Possible actions
state_space_size = 100  # Reduced state space size for efficiency
q_table = np.zeros((state_space_size, len(actions)))  # Initialize Q-table

# Hyperparameters for SARSA
n_episodes = 500
alpha = 0.1  # Learning rate
discount_factor = 0.95  # Discount future rewards
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
min_epsilon = 0.01  # Minimum exploration rate

# Define Reward Function with Penalties
def get_reward(action, positions, current_price, next_price):
    """
    Reward function that penalizes unprofitable trades and risky behaviors.
    - Reward for buy: Positive if price increases
    - Reward for sell: Positive if price decreases
    - Penalty for over-trading or high-risk actions.
    """
    if action == 'buy' and positions >= 3:
        return -2  # Penalty for over-buying
    elif action == 'sell' and positions <= 0:
        return -2  # Penalty for selling with no positions
    elif action == 'buy':
        return max(0, next_price - current_price)  # Reward if price increases after buying
    elif action == 'sell':
        return max(0, current_price - next_price)  # Reward if price decreases after selling
    return -0.1  # Small penalty for holding

# State Discretization Function
def get_state(row):
    """
    Convert continuous features into discrete states.
    - Aggregate technical indicators into bins for simpler state representation.
    - Helps reduce state space dimensionality for efficient learning.
    """
    price_bin = int(row['Close'] // 10)  # Price range bin
    ma_diff = int((row['MA5'] - row['MA20']) * 10)  # Moving average difference bin
    rsi_bin = int(row['RSI'] // 10)  # RSI bin
    
    # Combine binned values to represent a unique state (reduce to fit within state space)
    return min(state_space_size - 1, abs(price_bin + ma_diff + rsi_bin))

# Training the Agent with SARSA
for episode in range(n_episodes):
    state = get_state(data.iloc[0])  # Initial state
    total_reward = 0
    positions = 0
    balance = 1000  # Initial cash balance

    # Epsilon decay for exploration-exploitation balance
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    for t in range(1, len(data) - 1):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action_index = np.random.choice([0, 1, 2])  # Random action (explore)
        else:
            action_index = np.argmax(q_table[state])  # Best action based on learned Q-values (exploit)
        
        # Take action and observe next state
        action = actions[action_index]
        current_price = data['Close'].iloc[t]
        next_price = data['Close'].iloc[t + 1]
        next_state = get_state(data.iloc[t + 1])  # Get next state discretized
        reward = get_reward(action, positions, current_price, next_price)

        # Update balance and positions based on action
        if action == 'buy' and balance >= current_price:
            positions += 1
            balance -= current_price
        elif action == 'sell' and positions > 0:
            positions -= 1
            balance += current_price

        # SARSA update for Q-value with next action's Q-value
        if np.random.rand() < epsilon:
            next_action_index = np.random.choice([0, 1, 2])  # Random action for next state (explore)
        else:
            next_action_index = np.argmax(q_table[next_state])  # Best action for next state (exploit)
        
        # Calculate target using SARSA formula
        q_target = reward + discount_factor * q_table[next_state, next_action_index]
        q_table[state, action_index] = (1 - alpha) * q_table[state, action_index] + alpha * q_target

        # Move to the next state
        state = next_state
        total_reward += reward

    # Print progress every 50 episodes for tracking performance
    if episode % 50 == 0:
        print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# Evaluation after training
state = get_state(data.iloc[0])  # Reset to initial state
balance = 1000  # Reset balance for evaluation
positions = 0
for t in range(1, len(data) - 1):
    action_index = np.argmax(q_table[state])  # Choose best action from trained Q-table
    action = actions[action_index]
    current_price = data['Close'].iloc[t]

    # Execute action for balance and positions
    if action == 'buy' and balance >= current_price:
        positions += 1
        balance -= current_price
    elif action == 'sell' and positions > 0:
        positions -= 1
        balance += current_price
    
    # Update state for next step
    state = get_state(data.iloc[t + 1])

final_balance = balance + positions * data['Close'].iloc[-1]
print(f"Final Balance after Training: ${final_balance:.2f}")
