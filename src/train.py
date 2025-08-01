import argparse
import yaml
import pandas as pd
import numpy as np
import os
import torch

from src.envs.stock_env import StockEnv
from src.models.dqn import DQNAgent
from src.utils.logger import CSVLogger


def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std()
    if std < 1e-8:
        return 0.0
    return mean / std * np.sqrt(len(returns))


def train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env_cfg = config['environment']
    dqn_cfg = config['dqn']
    train_cfg = config['train']

    # Load historical data
    data_path = config.get('data_path', 'spy.csv')
    data = pd.read_csv(data_path)

    env = StockEnv(
        data=data,
        features=env_cfg['features'],
        initial_cash=env_cfg['initial_cash'],
        transaction_cost=env_cfg['transaction_cost'],
        max_steps=env_cfg['max_steps'],
        lookback=env_cfg['lookback']
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, dqn_cfg)

    # Set up logger
    log_path = train_cfg['log_path']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = CSVLogger(log_path, ['episode', 'final_value', 'total_reward', 'sharpe'])

    best_value = -float('inf')
    num_episodes = dqn_cfg['num_episodes']
    eval_interval = train_cfg.get('eval_interval', 10)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        returns = []
        prev_value = env._get_portfolio_value()

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state

            total_reward += reward
            current_value = info['portfolio_value']
            ret = (current_value - prev_value) / (prev_value + 1e-8)
            returns.append(ret)
            prev_value = current_value

        final_value = env._get_portfolio_value()
        sharpe = compute_sharpe(returns)
        logger.log({
            'episode': episode,
            'final_value': final_value,
            'total_reward': total_reward,
            'sharpe': sharpe
        })

        if final_value > best_value:
            best_value = final_value
            save_path = train_cfg['save_path']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.policy_net.state_dict(), save_path)

        if (episode + 1) % eval_interval == 0:
            print(f'Episode {episode+1}/{num_episodes} | Value: {final_value:.2f} | Sharpe: {sharpe:.2f}')

    print('Training completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    train(args.config)
