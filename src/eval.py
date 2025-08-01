import argparse
import yaml
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from src.envs.stock_env import StockEnv
from src.models.dqn import QNetwork


def compute_sharpe(returns: np.ndarray) -> float:
    """Compute the Sharpe ratio given an array of returns."""
    returns = np.asarray(returns)
    if returns.size == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std()
    if std == 0:
        return 0.0
    # Annualize assuming returns are per step; multiply by sqrt(N)
    return mean / std * np.sqrt(len(returns))


def max_drawdown(values: np.ndarray) -> float:
    """Compute maximum drawdown from a sequence of portfolio values."""
    arr = np.asarray(values)
    cummax = np.maximum.accumulate(arr)
    drawdowns = (arr - cummax) / cummax
    return drawdowns.min()


def evaluate(config_path: str, model_path: str) -> None:
    """Evaluate a trained DQN model on the stock trading environment.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_path (str): Path to the saved PyTorch model weights (.pth).
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    env_config = config.get("environment", {})

    # Load market data
    # Expect the SPY CSV to be at project root; adjust path if needed
    data_path = env_config.get("data_path", "spy.csv")
    df = pd.read_csv(data_path)

    # Create environment
    env = StockEnv(df, **env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Initialize Q-network and load weights
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    obs = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        portfolio_values.append(env.portfolio_value)

    # Compute metrics
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    sharpe = compute_sharpe(returns)
    mdd = max_drawdown(values)
    win_rate = float((returns > 0).mean())
    final_value = float(values[-1])

    # Persist results
    results_dir = config.get("train", {}).get("save_path", "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_df = pd.DataFrame([
        {
            "final_value": final_value,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "win_rate": win_rate,
        }
    ])
    metrics_path = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Plot equity curve
    plt.figure(figsize=(10, 4))
    plt.plot(values, label="Equity Curve")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt_path = os.path.join(results_dir, "equity_curve.png")
    plt.savefig(plt_path)
    plt.close()

    print(f"Evaluation completed. Final portfolio value: {final_value:.2f}")
    print(f"Sharpe ratio: {sharpe:.4f}, Max drawdown: {mdd:.4f}, Win rate: {win_rate:.4f}")
    print(f"Metrics saved to {metrics_path} and equity curve saved to {plt_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN trading agent.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model weights (.pth)."
    )
    args = parser.parse_args()
    evaluate(args.config, args.model_path)
