# 📈 Chime Stock Predictor

LSTM + Attention model for predicting stock prices using PyTorch Lightning.

**Your Position:** 26,000 shares

## Quick Start

```bash
cd /teamspace/studios/this_studio/stock-predictor
python chime_predictor.py
```

## What's Inside

### The Training Loop (Lightning Style)

The magic happens in `StockPredictor.training_step()`:

```python
def training_step(self, batch, batch_idx):
    x, y = batch           # Get batch of sequences
    y_hat = self(x)        # Forward pass
    loss = MSELoss()(y_hat, y)  # Compute loss
    self.log('train_loss', loss)  # Log it
    return loss            # Lightning handles backprop!
```

**Without Lightning**, you'd write:
```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        # + logging, checkpointing, GPU handling, etc...
```

Lightning handles all that boilerplate for you.

### Model Architecture

```
Input (60 days of features)
    ↓
LSTM (2 layers, 128 hidden)
    ↓
Attention (focus on important days)
    ↓
Dense layers
    ↓
Predicted return
```

### Features Used

- **Price:** Close, Returns, Log Returns
- **Moving Averages:** 5, 20, 50 day
- **Volatility:** 20-day rolling std
- **Technical Indicators:** RSI, MACD, Bollinger Bands
- **Volume:** Ratio vs 20-day average

## Config

Edit `CONFIG` at the top of `chime_predictor.py`:

```python
CONFIG = {
    "ticker": "CHMF",           # Stock symbol
    "lookback_days": 60,        # History window
    "forecast_days": 5,         # Days to predict
    "hidden_size": 128,         # LSTM hidden size
    "num_layers": 2,            # LSTM layers
    "max_epochs": 100,          # Training epochs
    "shares_owned": 26000,      # Your position
}
```

## Output

1. **Training metrics** - loss, direction accuracy
2. **Price predictions** - next 5 days
3. **Position analysis** - when to sell
4. **Chart** - `chime_prediction.png`

## ⚠️ Disclaimer

This is for learning ML, not financial advice. Stock prediction is hard — models can be wrong. Don't make financial decisions based solely on this.

## Next Steps

- [ ] Add sentiment analysis (news, social media)
- [ ] Try Transformer architecture
- [ ] Add more stocks for comparison
- [ ] Backtesting framework
- [ ] Real-time predictions
