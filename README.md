# 🚀 Advanced PPO Trading Agent PPOScalper for ETH-USD Scalping on AMM ETH mainnet.

A sophisticated Reinforcement Learning trading agent using Proximal Policy Optimization (PPO) for ETH-USD scalping on 5-minute timeframes. This system features advanced neural networks, curriculum learning, meta-learning, and comprehensive risk management.

## 🎯 Key Features

### 🤖 Advanced PPO Implementation
- **LSTM-based Neural Networks**: Captures temporal dependencies in price action
- **Multi-layered Architecture**: Deep networks with dropout for regularization
- **Position Sizing Network**: Intelligent capital allocation based on confidence
- **Meta-learning**: Self-tuning hyperparameters and learning rates

### 📊 40+ Technical Indicators
- **Trend Indicators**: SMA, EMA, MACD, ADX, Parabolic SAR, Ichimoku
- **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI, ROC
- **Volatility Indicators**: Bollinger Bands, ATR, Volatility ratios
- **Volume Indicators**: OBV, VWAP, Volume ratios
- **Support/Resistance**: Dynamic levels and Fibonacci retracements

### 🎓 Curriculum Learning (5 Stages)
1. **Basic Trading**: Fundamental buy/sell/hold actions
2. **Risk Management**: Stop-loss, position sizing, drawdown control
3. **Multi-timeframe**: Integration of 5m (primary), 15m, 1h, 4h, 1d data
4. **Market Adaptation**: Dynamic strategy adjustment
5. **Performance Optimization**: Advanced risk assessment and execution

### 💰 Realistic Trading Environment
- **Trading Costs**: Dynamic fees ($3-$5 per trade) + slippage (0.05%)
- **Risk Management**: Max 1% loss per trade, 3-5% max drawdown
- **Position Sizing**: 5-25% of capital per trade (adaptive)
- **Auto-sell**: 4-hour maximum holding time

### 📈 Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside deviation
- **Profit Expectancy**: (WinRate × AvgWin) - (LossRate × AvgLoss)
- **Win Rate**: Secondary to overall expectancy
- **Maximum Drawdown**: Portfolio protection

## 🏗️ System Architecture

```
ai-agent-trading-ppo/
├── environment/
│   ├── trading_env.py      # Advanced trading environment
│   └── ppo_agent.py        # PPO agent with LSTM networks
├── dashboard/
│   ├── advanced_dashboard.py  # Real-time Streamlit dashboard
│   └── dashboard.py          # Legacy dashboard
├── logs/                   # Training logs and plots
├── models/                 # Saved model checkpoints
├── data/                   # Market data storage
├── trainer.py              # Main training loop
├── main.py                 # Command-line interface
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/soradaisuki/ppo-scalper.git
cd ai-agent-trading-ppo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Check dependencies
python main.py check-deps
```

### 2. Training the Agent

```bash
# Basic training
python main.py train --data-path ohlcv_5m.csv --max-episodes 1000

# Advanced training with custom parameters
python main.py train \
    --data-path ohlcv_5m.csv \
    --initial-balance 100000 \
    --max-episodes 5000 \
    --device cuda \
    --save-frequency 100 \
    --eval-frequency 50
```

### 3. Running the Dashboard

```bash
# Start the real-time dashboard
python main.py dashboard --port 8501

# Access at: http://localhost:8501
```

### 4. Evaluating a Model

```bash
# Evaluate a trained model
python main.py evaluate \
    --model-path models/best_model.pth \
    --eval-episodes 50 \
    --data-path ohlcv_5m.csv
```

## 📊 Dashboard Features

The advanced dashboard provides real-time monitoring of:

- **Training Progress**: Episode rewards, returns, and learning curves
- **Portfolio Performance**: Equity curve with drawdown analysis
- **Trade Analysis**: P&L distribution, win/loss ratios, holding times
- **Technical Indicators**: Price charts with 40+ indicators
- **System Status**: GPU usage, memory, training status
- **Model Management**: Save/load models, performance tracking

## 🧠 Advanced Features

### Meta-Learning
- **Adaptive Learning Rate**: Automatically adjusts based on performance
- **Entropy Coefficient Tuning**: Maintains optimal exploration
- **Hyperparameter Optimization**: Self-evolving parameters

### Curriculum Learning
- **Progressive Difficulty**: 5 stages of increasing complexity
- **Performance Thresholds**: Automatic stage advancement
- **Skill Building**: From basic trading to advanced strategies

### Risk Management
- **Dynamic Stop-Loss**: 1% maximum loss per trade
- **Drawdown Protection**: 3-5% maximum portfolio drawdown
- **Position Sizing**: Intelligent capital allocation
- **Auto-sell Logic**: Prevents excessive holding times

### Technical Analysis
- **40+ Indicators**: Comprehensive market analysis
- **Multi-timeframe**: 5m (primary), 15m, 1h, 4h, 1d integration
- **Pattern Recognition**: Support/resistance, Fibonacci levels
- **Volume Analysis**: OBV, VWAP, volume ratios

## 📈 Performance Expectations

### Training Metrics
- **Episode Length**: 500-1000 steps per episode (41.67-83.33 hours at 5-minute intervals)
- **Minimum Trades**: 10-50 trades per episode
- **Win Rate**: 40-60% (secondary to profit expectancy)
- **Sharpe Ratio**: Target > 1.0
- **Maximum Drawdown**: < 5%

### Curriculum Progression
- **Stage 1**: Basic trading patterns (episodes 1-500)
- **Stage 2**: Risk management integration (episodes 500-1000)
- **Stage 3**: Multi-timeframe analysis (episodes 1000-2000)
- **Stage 4**: Market adaptation + Trading strategy guidance (episodes 2000-3000)
- **Stage 5**: Performance optimization + Advanced strategy guidance (episodes 3000+)

## 🔧 Configuration

### Environment Parameters
```python
# Trading Environment
initial_balance = 100000.0
max_position_size = 0.25  # 25% max per trade
trading_fee = 4.0         # $4 per trade
slippage = 0.0005         # 0.05%
max_loss_per_trade = 0.01 # 1%
max_drawdown = 0.05       # 5%
episode_length = 1000     # steps per episode
```

### PPO Hyperparameters
```python
# PPO Agent
learning_rate = 3e-4
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # GAE parameter
clip_epsilon = 0.2        # PPO clip parameter
value_coef = 0.5          # Value loss coefficient
entropy_coef = 0.01       # Entropy coefficient
max_grad_norm = 0.5       # Gradient clipping
```

### Neural Network Architecture
```python
# LSTM Policy Network
hidden_size = 128         # LSTM hidden size
num_layers = 2            # LSTM layers
dropout = 0.2             # Dropout rate
input_size = 58           # Features (OHLCV + indicators + state)
action_size = 4           # buy, sell, hold + position_size
```

## 📁 Data Format

The system expects OHLCV data in CSV format:
```csv
timestamp,open,high,low,close,volume
2024-12-23 19:15:00,92918.68,93166.65,92918.68,93127.68,1263155.62
2024-12-23 19:45:00,93127.68,93127.68,92897.77,92897.77,647425.42
...
```

## 🎯 Trading Strategy Features

### Trading Strategy Reference Features (Stage 4+)
Starting from Stage 4, the agent receives optional guidance from 8 proven trading strategies:

1. **Momentum Scalping**: Buy/sell signals based on short-term price momentum
2. **RSI Crossover**: Oversold/overbought signals when RSI crosses 30/70 levels
3. **VWAP Reversion**: Mean reversion signals relative to Volume Weighted Average Price
4. **EMA Crossover**: Golden cross/death cross signals from fast/slow EMA crossovers
5. **MACD Histogram Scalping**: Momentum signals from MACD histogram changes
6. **Breakout Scalping**: Breakout signals when price breaks Bollinger Band levels
7. **Volume Spike Scalping**: Volume-based signals combined with price movement
8. **Fibonacci Levels**: Dynamic support/resistance levels calculated from recent swing points

### Strategy Guidance System
- **Optional Guidance**: Agent is encouraged but not forced to follow strategies
- **Consensus Signals**: Combined signals from multiple strategies for stronger guidance
- **Confidence Scoring**: Only provides guidance when multiple strategies agree
- **Small Rewards**: Minimal bonuses for following high-confidence signals
- **Learning Focus**: Helps agent discover effective trading logic through reinforcement learning

### Strategy Features
- **Strategy Consensus**: Average signal from all 8 strategies (-1 to +1)
- **Strategy Confidence**: Percentage of strategies agreeing on direction (0 to 1)
- **Individual Signals**: Each strategy provides buy/sell/hold signals
- **Dynamic Calculation**: All levels and signals calculated in real-time

### Risk Controls
- **Capital Management**: Adaptive position sizing
- **Stop-Loss**: Dynamic 1% maximum loss
- **Take-Profit**: Intelligent profit-taking
- **Holding Limits**: 4-hour maximum position time

## 🔍 Monitoring and Logging

### Training Logs
- **Episode Data**: Rewards, returns, trade counts
- **Performance Metrics**: Sharpe, Sortino, drawdown
- **Model Checkpoints**: Automatic saving of best models
- **Training Plots**: Real-time visualization

### Dashboard Metrics
- **Real-time Updates**: Live training progress
- **Performance Charts**: Equity curves, trade analysis
- **System Monitoring**: GPU, memory, training status
- **Model Management**: Save/load functionality

## 🚨 Important Notes

### Risk Disclaimer
- This is a research project for educational purposes
- Paper trading only - no real money involved
- Past performance does not guarantee future results
- Always test thoroughly before live trading

### System Requirements
- **Python**: 3.8+
- **GPU**: Recommended for faster training (CUDA compatible)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 1GB+ for models and logs

### Best Practices
- **Start Small**: Begin with fewer episodes to test
- **Monitor Closely**: Use dashboard for real-time monitoring
- **Save Models**: Regular checkpoints prevent loss of progress
- **Evaluate Thoroughly**: Test on unseen data before deployment

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI Gym**: Reinforcement learning framework
- **PyTorch**: Deep learning framework
- **Technical Analysis Library**: Technical indicators
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

---

**Happy Trading! 🚀📈** 
