# 📈 RL-Driven Stock Trading Agent using Deep Q-Learning

A production-grade reinforcement learning system applying **Deep Q-Networks (DQN)** to historical SPY (S&P 500 ETF) data. This project showcases a custom OpenAI Gym environment, an end-to-end training pipeline, and thoughtful metrics tracking.

## 💼 Purpose

This project demonstrates:

- Custom environment engineering for financial markets.
- Design, training, and evaluation of a Deep Q-Learning agent.
- Realistic reward shaping using profit and Sharpe ratio.
- Modular, reusable code structure with configuration via YAML.
- A reproducible pipeline ready for experimentation and extension.

## 🤓 Problem Overview

The agent learns to **buy, sell, or hold** SPY based on historical price features to maximise long-term return while controlling drawdown and volatility.

- **Action space**: buy, sell, or hold.
- **State space**: price indicators (close, SMA, RSI, etc.).
- **Reward**: profit and loss (PnL) penalised by volatility and over-trading.

## 🪒 Repository Structure

```
rl_learning_stockmarket/
├── config/
│   └── config.yaml          # Hyper-parameters for training
├── data/
│   └── spy.csv              # Historical SPY data (provided separately)
├── src/
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation and back-testing
│   ├── envs/
│   │   └── stock_env.py     # Custom Gym environment
│   ├── models/
│   │   └── dqn.py           # DQN agent definition
│   │
│   └── utils/
│       └── logger.py        # Logging utilities
├── results/
│   ├── equity_curve.png     # Saved equity curve
│   └── metrics.csv          # Logged evaluation metrics
├── Dockerfile               # Container for reproducibility
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)
```

## ⚙️ Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/edawite/rl_learning_stockmarket.git
cd rl_learning_stockmarket
pip install -r requirements.txt
```

## 🚀 Quickstart

Train the agent using the configuration file:

```bash
python src/train.py --config config/config.yaml
```

Evaluate a trained policy:

```bash
python src/eval.py --model_path results/best_model.pth
```


## 🔍 Future Improvements

- Add PPO, SAC, and DDPG agents for comparison.
- Incorporate LSTM or transformer models for richer state encoding.
- Integrate a broker API for live paper-trading.
- Extend the feature set with additional technical indicators.

## 🤓 Credits

Developed by Edjutawee Dawit. This repository is designed for internship-ready demonstration of reinforcement learning applied to algorithmic trading.

## 🔐 Legal

**Disclaimer**: This project is for research and educational purposes only and does not constitute financial advice.
