"""
Chime Stock Price Predictor using PyTorch Lightning
====================================================
Predicts $CHMF stock price movements using LSTM + Attention.

Cameron's position: 26,000 shares
Goal: Find optimal sell timing based on predicted price movements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# ============================================================
# CONFIG - Easy to tweak
# ============================================================
CONFIG = {
    "ticker": "CHYM",           # Chime Financial
    "lookback_days": 20,        # How many days of history to use for prediction
    "forecast_days": 5,         # How many days ahead to predict
    "train_split": 0.8,         # 80% train, 20% validation
    "batch_size": 32,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "max_epochs": 100,
    "shares_owned": 26000,      # Your position
}


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    print(f"📈 Fetching {ticker} data...")
    
    # Try multiple tickers in case some fail
    tickers_to_try = [ticker, "PYPL", "AAPL", "SPY"]
    
    for t in tickers_to_try:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period=period)
            
            if not df.empty and len(df) > 100:
                if t != ticker:
                    print(f"⚠️  {ticker} not found, using {t} as proxy...")
                print(f"✅ Loaded {len(df)} days of data from {df.index[0].date()} to {df.index[-1].date()}")
                return df
        except Exception as e:
            print(f"⚠️  Error fetching {t}: {e}")
            continue
    
    # If all else fails, generate synthetic data for demo
    print("⚠️  Using synthetic data for demonstration...")
    dates = pd.date_range(end=datetime.now(), periods=500, freq='B')
    np.random.seed(42)
    
    # Generate realistic-looking stock data
    returns = np.random.normal(0.0005, 0.02, 500)
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.randint(1000000, 10000000, 500)
    
    df = pd.DataFrame({
        'Open': prices * (1 - np.random.uniform(0, 0.01, 500)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 500)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 500)),
        'Close': prices,
        'Volume': volumes,
    }, index=dates)
    
    print(f"✅ Generated {len(df)} days of synthetic data")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the model."""
    data = df.copy()
    
    # Price features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Volatility
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Volume features
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # Drop NaN rows
    data = data.dropna()
    
    return data


# ============================================================
# DATASET
# ============================================================
class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, lookback: int):
        self.data = data
        self.targets = targets
        self.lookback = lookback
    
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ============================================================
# MODEL - LSTM with Attention
# ============================================================
class StockPredictor(pl.LightningModule):
    """
    LSTM + Attention model for stock prediction.
    
    This is where Lightning shines - clean separation of:
    - Model architecture (__init__, forward)
    - Training logic (training_step, validation_step)
    - Optimization (configure_optimizers)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()  # Lightning magic - logs all params
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self.learning_rate = learning_rate
        
        # Track predictions for analysis
        self.validation_outputs = []
    
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        
        # Prediction
        out = self.fc(context)
        return out
    
    def training_step(self, batch, batch_idx):
        """
        🔥 THE TRAINING LOOP - This is what you wanted to see!
        
        Lightning calls this for each batch. It:
        1. Gets a batch of (sequences, targets)
        2. Forward pass through the model
        3. Computes loss
        4. Returns loss (Lightning handles backprop, optimizer step, etc.)
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Loss computation
        loss = nn.MSELoss()(y_hat, y)
        
        # Logging (Lightning handles TensorBoard, etc.)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation logic - same as training but no gradients."""
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        
        # Store for epoch-end analysis
        self.validation_outputs.append({
            'loss': loss,
            'y_true': y,
            'y_pred': y_hat,
        })
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Called at end of validation epoch - compute metrics."""
        if not self.validation_outputs:
            return
            
        # Aggregate predictions
        y_true = torch.cat([x['y_true'] for x in self.validation_outputs])
        y_pred = torch.cat([x['y_pred'] for x in self.validation_outputs])
        
        # Direction accuracy (did we predict up/down correctly?)
        direction_correct = ((y_pred > 0) == (y_true > 0)).float().mean()
        self.log('val_direction_acc', direction_correct, prog_bar=True)
        
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Optimizer setup - Lightning handles the training loop."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }


# ============================================================
# DATA MODULE - Lightning way to handle data
# ============================================================
class StockDataModule(pl.LightningDataModule):
    """
    Encapsulates all data loading logic.
    Lightning calls prepare_data() once, setup() per GPU.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        
    def prepare_data(self):
        """Download data (called once)."""
        fetch_stock_data(self.config['ticker'])
    
    def setup(self, stage=None):
        """Prepare datasets (called on each GPU)."""
        # Fetch and process data
        df = fetch_stock_data(self.config['ticker'])
        df = create_features(df)
        
        # Features to use
        feature_cols = [
            'Close', 'Volume', 'Returns', 'MA_5', 'MA_20', 'MA_50',
            'Volatility_20', 'RSI', 'MACD', 'BB_Position', 'Volume_Ratio'
        ]
        
        # Target: Next day's return
        df['Target'] = df['Returns'].shift(-1)
        df = df.dropna()
        
        # Scale features
        features = df[feature_cols].values
        targets = df['Target'].values
        
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Store for later (prediction)
        self.raw_df = df
        self.feature_cols = feature_cols
        
        # Train/val split
        split_idx = int(len(features_scaled) * self.config['train_split'])
        
        self.train_dataset = StockDataset(
            features_scaled[:split_idx],
            targets[:split_idx],
            self.config['lookback_days']
        )
        
        self.val_dataset = StockDataset(
            features_scaled[split_idx:],
            targets[split_idx:],
            self.config['lookback_days']
        )
        
        print(f"📊 Training samples: {len(self.train_dataset)}")
        print(f"📊 Validation samples: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
        )


# ============================================================
# PREDICTION & ANALYSIS
# ============================================================
def predict_future(model, data_module, days_ahead=5):
    """Generate predictions for the next N days."""
    model.eval()
    
    # Get the last sequence
    df = data_module.raw_df
    features = df[data_module.feature_cols].values
    features_scaled = data_module.scaler.transform(features)
    
    last_sequence = features_scaled[-CONFIG['lookback_days']:]
    last_price = df['Close'].iloc[-1]
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    with torch.no_grad():
        for i in range(days_ahead):
            x = torch.FloatTensor(current_sequence).unsqueeze(0)
            pred_return = model(x).item()
            predictions.append(pred_return)
            
            # Shift sequence (simplified - just use predicted return)
            current_sequence = np.roll(current_sequence, -1, axis=0)
    
    # Convert returns to prices
    predicted_prices = [last_price]
    for ret in predictions:
        predicted_prices.append(predicted_prices[-1] * (1 + ret))
    
    return predictions, predicted_prices[1:]


def analyze_position(current_price, predicted_prices, shares=26000):
    """Analyze the trading position based on predictions."""
    print("\n" + "="*60)
    print("📊 POSITION ANALYSIS - 26,000 Shares")
    print("="*60)
    
    current_value = current_price * shares
    print(f"\n💰 Current Position Value: ${current_value:,.2f}")
    print(f"📈 Current Price: ${current_price:.2f}")
    
    print("\n📅 Price Predictions:")
    for i, price in enumerate(predicted_prices, 1):
        change = ((price - current_price) / current_price) * 100
        value = price * shares
        profit = value - current_value
        emoji = "🟢" if change > 0 else "🔴"
        print(f"   Day {i}: ${price:.2f} ({emoji} {change:+.2f}%) → Value: ${value:,.2f} ({'+' if profit > 0 else ''}{profit:,.2f})")
    
    # Best sell day
    best_day = np.argmax(predicted_prices) + 1
    best_price = max(predicted_prices)
    best_value = best_price * shares
    best_profit = best_value - current_value
    
    print(f"\n🎯 RECOMMENDATION:")
    if best_price > current_price:
        print(f"   Best predicted sell: Day {best_day} @ ${best_price:.2f}")
        print(f"   Potential profit: ${best_profit:,.2f}")
    else:
        print(f"   ⚠️  Model predicts decline - consider selling soon or holding")
    
    return best_day, best_price


def plot_results(df, model, data_module, predictions, save_path="prediction_chart.png"):
    """Create visualization of predictions."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Price history with predictions
    ax1 = axes[0]
    recent_prices = df['Close'].tail(60)
    ax1.plot(recent_prices.index, recent_prices.values, label='Actual Price', color='blue')
    
    # Add prediction line
    last_date = recent_prices.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions), freq='B')
    ax1.plot(future_dates, predictions, label='Predicted Price', color='red', linestyle='--', marker='o')
    
    ax1.axhline(y=recent_prices.iloc[-1], color='gray', linestyle=':', alpha=0.5, label='Current Price')
    ax1.set_title('Stock Price: History & Predictions')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI
    ax2 = axes[1]
    rsi = df['RSI'].tail(60)
    ax2.plot(rsi.index, rsi.values, label='RSI', color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax2.set_title('RSI Indicator')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volume
    ax3 = axes[2]
    volume = df['Volume'].tail(60)
    ax3.bar(volume.index, volume.values, alpha=0.7, color='teal')
    ax3.set_title('Trading Volume')
    ax3.set_ylabel('Volume')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Chart saved to {save_path}")
    plt.close()


# ============================================================
# MAIN - Run everything
# ============================================================
def main():
    print("🚀 Chime Stock Predictor")
    print("="*60)
    print(f"Position: {CONFIG['shares_owned']:,} shares")
    print(f"Lookback: {CONFIG['lookback_days']} days")
    print(f"Forecast: {CONFIG['forecast_days']} days ahead")
    print("="*60)
    
    # Set up data
    data_module = StockDataModule(CONFIG)
    data_module.setup()
    
    # Get input size from data
    sample_x, _ = data_module.train_dataset[0]
    input_size = sample_x.shape[1]
    
    # Create model
    model = StockPredictor(
        input_size=input_size,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        learning_rate=CONFIG['learning_rate'],
    )
    
    print(f"\n🧠 Model Architecture:")
    print(model)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='best-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
        ),
    ]
    
    # Trainer - This is the Lightning magic ✨
    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        callbacks=callbacks,
        accelerator='auto',  # Uses GPU if available
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train! 🚂
    print("\n🏋️ Starting Training...")
    print("-"*60)
    trainer.fit(model, data_module)
    
    # Load best model
    best_model_path = callbacks[1].best_model_path
    if best_model_path:
        print(f"\n📂 Loading best model from {best_model_path}")
        model = StockPredictor.load_from_checkpoint(best_model_path)
    
    # Predictions
    print("\n🔮 Generating Predictions...")
    returns, predicted_prices = predict_future(model, data_module, CONFIG['forecast_days'])
    
    # Analysis
    current_price = data_module.raw_df['Close'].iloc[-1]
    best_day, best_price = analyze_position(current_price, predicted_prices, CONFIG['shares_owned'])
    
    # Plot
    plot_results(
        data_module.raw_df,
        model,
        data_module,
        predicted_prices,
        save_path="chime_prediction.png"
    )
    
    print("\n✅ Done!")
    return model, data_module, predicted_prices


if __name__ == "__main__":
    main()
