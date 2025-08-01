import argparse
import yaml
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from src.envs.stock_env import StockEnv
from src.models.dqn import QNetwork


def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std()
    if std < 1e-8:
        return 0.0
    return mean / std * np.sqrt(len(returns))


def max_drawdown(values):
    arr = np.array(values)
    cummax = np.maximum.accumulate(arr)
    drawdowns = (arr - cummax) / cummax
    return drawdowns.min()


def evaluate(config_path: str, model_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    env_cfg = config['environment']
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

    # Load trained network
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    state = env.reset()
    done = False
    portfolio_values = [env._get_portfolio_value()]

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = int(torch.argmax(q_values).item())
        next_state, _, done, info = env.step(action)
        state = next_state
        portfolio_values.append(info['portfolio_value'])

    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    sharpe = compute_sharpe(returns)
    max_dd = max_drawdown(values)
    win_rate = float((returns > 0).mean())
    final_value = float(values[-1])

    # Save metrics
    results_dir = os.path.join(os.path.dirname(model_path), 'eval_results')
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('metric,value\n')
        f.write(f'Final Portfolio Value,{final_value}\n')
        f.write(f'Sharpe Ratio,{sharpe}\n')
        f.write(f'Max Drawdown,{max_dd}\n')
        f.write(f'Win Rate,{win_rate}\n')

    # Plot equity curve
    plt.figure()
    plt.plot(values)
    plt.title('Equity Curve')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.tight_layout()
    equity_path = os.path.join(results_dir, 'equity_curve.png')
    plt.savefig(equity_path)
    print(f'Evaluation completed. Metrics saved to {metrics_file}, equity curve saved to {equity_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth', help='Path to trained model file')
    args = parser.parse_args()
    evaluate(args.config, args.model_path)
