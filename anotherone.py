# Import necessary libraries
import gym  # OpenAI's toolkit for developing and comparing RL algorithms
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation
from gym import spaces  # Provides action and observation spaces for RL
from stable_baselines3 import DQN  # DQN algorithm for training our agent
from stable_baselines3.common.vec_env import DummyVecEnv  # Wrapper for environment to work with stable-baselines3
import gymnasium as gym

from gymnasium import spaces



# Define the StockTradingEnv class as a custom Gym environment
class StockTradingEnv(gym.Env):
    def __init__(self, df):
        """
        Initialize the stock trading environment.
        
        Parameters:
        df (DataFrame): Data containing historical stock prices.
        """
        super(StockTradingEnv, self).__init__()
        
        # Define trading parameters
        self.initial_balance = 10000  # Starting cash balance for the agent
        self.df = df  # Historical stock data DataFrame
        self.current_step = 0  # Tracks the current step in the data (index in df)
        self.balance = self.initial_balance  # Current cash balance
        self.net_worth = self.initial_balance  # Total value of portfolio (balance + stock value)
        self.shares_held = 0  # Number of shares the agent currently holds

        # Action space: 3 discrete actions (0: Buy, 1: Sell, 2: Hold)
        self.action_space = spaces.Discrete(3)

        # Observation space: a vector containing stock data (e.g., OHLC, volume) + balance + shares_held
        # `shape=(len(df.columns) + 2,)` means number of columns in df + 2 additional features (balance, shares held)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(df.columns) + 2,), dtype=np.float32
        )

    def reset(self, seed=None):
        """
        Reset the environment to an initial state and return the first observation and an empty info dictionary.
        """
        # Optional seeding
        if seed is not None:
            np.random.seed(seed)

        # Reset all variables to start a new episode
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = 0  # Start from the beginning of the dataset
        return self._next_observation(), {}  # Return the initial observation and an empty info dictionary

    def _next_observation(self):
        """
        Get the next observation, which includes stock data and current balance and shares held.
        """
        # Combine stock data (current row in df) with balance and shares_held into a single observation
        obs = np.array(
            list(self.df.iloc[self.current_step]) + [self.balance, self.shares_held]
        )
        return obs  # Return the observation as a numpy array

    def step(self, action):
        """
        Take an action (0: Buy, 1: Sell, 2: Hold), update the environment, and return the new state, reward, done flags, and info.
        """
        # Ensure we don't exceed the dataset bounds
        if self.current_step >= len(self.df) - 1:
            terminated = True
            truncated = False
            return self._next_observation(), 0, terminated, truncated, {}

        current_price = self.df.iloc[self.current_step]["Close"]
        reward = 0  # Initialize reward to 0

        # Execute the chosen action
        if action == 0:  # Buy action
            if self.balance > current_price:  # Check if there's enough balance to buy at least one share
                shares_bought = self.balance // current_price  # Number of shares to buy
                self.balance -= shares_bought * current_price  # Decrease balance by the total price
                self.shares_held += shares_bought  # Increase the number of shares held by agent

        elif action == 1:  # Sell action
            if self.shares_held > 0:  # Only sell if there are shares to sell
                self.balance += self.shares_held * current_price  # Increase balance by the total sell price
                self.shares_held = 0  # Reset shares held to 0 after selling

        # Calculate the agent's net worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Calculate the reward as the profit or loss from the initial balance
        reward = self.net_worth - self.initial_balance  # Positive if agent made money, negative otherwise

        # Check if the episode is done (we've reached the end of the dataset)
        terminated = self.current_step >= len(self.df) - 1  # Episode ends when we reach the last time step
        truncated = False  # Can be set to True if you want to add conditions for early termination

        # Move to the next time step
        self.current_step += 1
        obs = self._next_observation() if not terminated else self.reset()[0]  # Get the next observation or reset if terminated

        # Return the observation, reward, termination flags, and info dict
        return obs, reward, terminated, truncated, {}


    def render(self, mode='human'):
        """
        Print the current step, balance, net worth, and shares held.
        """
        print(f'Step: {self.current_step}, Balance: {self.balance}, Net Worth: {self.net_worth}, Shares Held: {self.shares_held}')

# Load stock data (e.g., from spy.csv file) and initialize the environment
# Load stock data (e.g., from spy.csv file) and preprocess to remove non-numeric columns
df = pd.read_csv('spy.csv')

# Drop non-numeric columns, assuming 'Date' is the column name
df = df.select_dtypes(include=[np.number])  # Keeps only numeric columns

# Initialize environment with the numeric-only data

 # Historical stock data; make sure it includes columns 'Open', 'High', 'Low', 'Close', 'Volume'
env = DummyVecEnv([lambda: StockTradingEnv(df)])  # Vectorize the environment to work with stable-baselines3

# Create and train the DQN agent on the environment
model = DQN("MlpPolicy", env, verbose=1)  # Use MLP (Multi-Layer Perceptron) Policy
model.learn(total_timesteps=10000)  # Train the model for 10,000 timesteps
model.save("stock_trading_dqn")  # Save the trained model for future use

# Test the trained model on the environment
test_env = StockTradingEnv(df)  # Create a new instance of the environment for testing
obs = test_env.reset()  # Reset the environment to the initial state

# Run a loop through the environment steps to test the model
for _ in range(len(df)):  # Loop over each time step in the dataset
    action, _states = model.predict(obs)  # Predict the best action to take based on current observation
    obs, rewards, done, info = test_env.step(action)  # Take the action and get the next state and reward
    test_env.render()  # Print out the environment state at each step
    if done:  # Stop the loop if we've reached the end of the dataset
        break
