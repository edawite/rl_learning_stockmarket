import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import List
import math


class StockEnv(gym.Env):
    """Custom stock trading environment with discrete actions: hold, buy, sell."""
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, features: List[str], initial_cash: float = 100000.0,
                 transaction_cost: float = 0.001, max_steps: int = 1000, lookback: int = 50):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.features = features
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.lookback = lookback
        # Precompute technical indicators
        self._prepare_data()
        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        obs_dim = len(self.features) * self.lookback + 2  # plus cash and shares
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def _prepare_data(self):
        """Compute feature columns required by the environment."""
        df = pd.DataFrame(index=self.data.index)
        close = self.data['Close']
        for feat in self.features:
            if feat == 'close':
                df['close'] = close
            elif feat.startswith('sma_'):
                window = int(feat.split('_')[1])
                df[feat] = close.rolling(window).mean()
            elif feat.startswith('ema_'):
                window = int(feat.split('_')[1])
                df[feat] = close.ewm(span=window, adjust=False).mean()
            elif feat.startswith('rsi_'):
                period = int(feat.split('_')[1])
                delta = close.diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.rolling(period).mean()
                roll_down = down.rolling(period).mean()
                rs = roll_up / (roll_down + 1e-8)
                df[feat] = 100.0 - (100.0 / (1.0 + rs))
            else:
                raise ValueError(f"Unknown feature {feat}")
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        self.features_data = df

    def reset(self):
        """Reset the environment state."""
        self.cash = self.initial_cash
        self.shares = 0
        # start step at lookback to allow for lookback window
        self.current_step = self.lookback
        self.steps_left = self.max_steps
        self.trades = []
        return self._get_observation()

    def _get_observation(self):
        """Return the current observation."""
        start = self.current_step - self.lookback
        end = self.current_step
        window = self.features_data.iloc[start:end][self.features]
        obs = window.values.flatten()
        obs = np.append(obs, [self.cash, self.shares])
        return obs.astype(np.float32)

    def _get_portfolio_value(self):
        current_price = self.data.loc[self.current_step, 'Close']
        return self.cash + self.shares * current_price

    def step(self, action: int):
        assert self.action_space.contains(action)
        prev_value = self._get_portfolio_value()
        price = self.data.loc[self.current_step, 'Close']
        done = False
        # Perform action
        if action == 1:  # buy
            cost = price * (1.0 + self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += 1
                self.trades.append(('buy', self.current_step, price))
        elif action == 2:  # sell
            if self.shares > 0:
                proceeds = price * (1.0 - self.transaction_cost)
                self.cash += proceeds
                self.shares -= 1
                self.trades.append(('sell', self.current_step, price))
        # else action == 0 hold

        self.current_step += 1
        self.steps_left -= 1

        # compute reward
        current_value = self._get_portfolio_value()
        reward = current_value - prev_value

        # check if done
        if self.current_step >= len(self.data) or self.steps_left <= 0:
            done = True

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {'portfolio_value': current_value}
        return obs, reward, done, info

    def render(self, mode='human'):
        """Print the current state."""
        value = self._get_portfolio_value()
        print(f"Step: {self.current_step} | Price: {self.data.loc[self.current_step, 'Close']:.2f} | Cash: {self.cash:.2f} | Shares: {self.shares} | Portfolio Value: {value:.2f}")
