import pandas as pd
import numpy as np
import random

# Load and preprocess data
data = pd.read_csv('spy.csv')
data['Close'] = data['Close'].fillna(method='ffill')  # Forward fill missing values in 'Close' column

# Feature Engineering: Add Moving Averages (e.g., 5-day and 20-day)
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data.fillna(0, inplace=True)  # Fill NaNs from moving averages

# Hyperparameters for Q-learning
n_episodes = 1000          # Number of training episodes
initial_learning_rate = 0.1  # Starting learning rate
discount_factor = 0.95     # Discount future rewards
initial_epsilon = 1.0      # Starting exploration rate
epsilon_decay = 0.995      # Epsilon decay rate per episode
min_epsilon = 0.01         # Minimum epsilon to ensure some exploration

# Define Actions and Q-Table Initialization
actions = ['hold', 'buy', 'sell']  # Actions available to the agent
state_space_size = len(data) - 1   # Number of states, minus the last row
q_table = np.zeros((state_space_size, len(actions)))  # Initialize Q-table with zeros

# Reward Function with Dynamic Rewards
def get_reward(action, state, next_state, positions):
    """
    Computes the reward for the agent based on action taken and resulting state.
    - `positions` keeps track of the stock units held by the agent.
    - Rewards are higher for profitable trades and lower for holding during losses.
    """
    price_now = data['Close'].iloc[state]
    price_next = data['Close'].iloc[next_state]
    
    if action == 'buy':
        # Reward is potential profit if price increases
        return (price_next - price_now) * (1 + positions * 0.05)  # Higher reward with more positions
    elif action == 'sell':
        # Reward is potential profit if price drops after selling
        return (price_now - price_next) * (1 + positions * 0.05)
    return -0.1  # Small penalty to discourage holding for long periods

# Train the Q-Learning Agent
for episode in range(n_episodes):
    state = 0  # Start at the beginning of the dataset each episode
    total_reward = 0  # Track total reward for this episode
    learning_rate = max(0.01, initial_learning_rate * (0.99 ** episode))  # Decay learning rate per episode
    epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))  # Decay epsilon per episode
    positions = 0  # Track the number of stock units held by the agent

    while state < state_space_size - 1:
        # Decide on action using epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action_index = np.random.choice([0, 1, 2])  # Random action (explore)
        else:
            action_index = np.argmax(q_table[state])  # Best action based on learned Q-values (exploit)

        # Take action and observe the next state
        next_state = state + 1  # Move to the next day
        reward = get_reward(actions[action_index], state, next_state, positions)  # Get reward for the action

        # Limit consecutive buys/sells (to avoid unrealistic trading)
        if action_index == 1 and positions >= 3:  # Max 3 buy actions without selling
            action_index = 0  # Override to 'hold' if buy limit reached
        elif action_index == 2 and positions <= 0:  # No selling without any positions
            action_index = 0  # Override to 'hold' if no positions to sell

        # Update positions based on action
        if actions[action_index] == 'buy':
            positions += 1
        elif actions[action_index] == 'sell' and positions > 0:
            positions -= 1

        # Update Q-value based on the Q-learning formula
        best_future_q = np.max(q_table[next_state])  # Max Q-value for next state
        q_table[state, action_index] = q_table[state, action_index] + \
            learning_rate * (reward + discount_factor * best_future_q - q_table[state, action_index])

        # Move to the next state
        state = next_state
        total_reward += reward  # Accumulate total reward for this episode

    # Print progress every 100 episodes
    if episode % 100 == 0:
        print(f'Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.3f}, Learning Rate = {learning_rate:.3f}')

# Evaluate Agent after Training
state = 0
balance = 1000  # Start with an initial balance
positions = 0  # No stock positions initially
for t in range(state_space_size - 1):
    # Choose best action from Q-table (without exploration)
    action_index = np.argmax(q_table[t])
    current_price = data['Close'].iloc[t]
    
    # Execute action
    if actions[action_index] == 'buy' and balance >= current_price:
        positions += 1
        balance -= current_price
    elif actions[action_index] == 'sell' and positions > 0:
        positions -= 1
        balance += current_price

# Calculate final balance by adding cash and value of held stocks
final_balance = balance + positions * data['Close'].iloc[-1]
print(f"Final Balance from RL Agent: ${final_balance:.2f}")

# Buy-and-Hold Strategy for Comparison
start_price = data['Close'].iloc[0]
end_price = data['Close'].iloc[-1]
units_held = balance // start_price
final_balance_buy_and_hold = units_held * end_price
print(f"Buy-and-Hold Final Balance: ${final_balance_buy_and_hold:.2f}")
