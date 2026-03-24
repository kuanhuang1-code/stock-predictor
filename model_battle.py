"""
Model Battle: Find the Best Stock Predictor
============================================
Tests multiple ML architectures and picks the winner.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import torch
import torch.nn as nn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🏆 MODEL BATTLE: Finding the Best CHYM Predictor")
print("="*70)

# ============================================================
# DATA
# ============================================================
print("\n📈 Fetching CHYM data...")
stock = yf.Ticker("CHYM")
df = stock.history(period="max")
print(f"✅ Loaded {len(df)} days of data")

def create_features(df):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        data[f'MA_{window}'] = data['Close'].rolling(window).mean()
        data[f'MA_ratio_{window}'] = data['Close'] / data[f'MA_{window}']
    
    # Volatility
    data['Volatility_10'] = data['Returns'].rolling(10).std()
    data['Volatility_20'] = data['Returns'].rolling(20).std()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # MACD
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Bollinger
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_upper'] = bb_mid + 2 * bb_std
    data['BB_lower'] = bb_mid - 2 * bb_std
    data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'] + 1e-10)
    
    # Momentum
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    
    # Volume
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_ratio'] = data['Volume'] / (data['Volume_MA'] + 1e-10)
    
    # Price position
    data['High_Low_ratio'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    
    return data.dropna()

df = create_features(df)
print(f"✅ Created {len(df.columns)} features, {len(df)} samples after processing")

# ============================================================
# PREPARE DATA
# ============================================================
feature_cols = [
    'Returns', 'MA_ratio_5', 'MA_ratio_10', 'MA_ratio_20', 
    'Volatility_10', 'Volatility_20', 'RSI', 'MACD_hist', 
    'BB_position', 'Momentum_5', 'Momentum_10', 'Volume_ratio', 'High_Low_ratio'
]

# Target: next day return
df['Target'] = df['Returns'].shift(-1)
df = df.dropna()

X = df[feature_cols].values
y = df['Target'].values

# Time-series split (no shuffle!)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\n📊 Data Split:")
print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODELS TO TEST
# ============================================================

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Direction accuracy (most important for trading!)
    direction_correct = np.mean((y_pred > 0) == (y_true > 0))
    
    # Profit simulation: if we predict up, we buy; if down, we sell/short
    # Simple: sum of returns where we predicted correctly
    profits = np.where((y_pred > 0) == (y_true > 0), np.abs(y_true), -np.abs(y_true))
    total_return = np.sum(profits)
    
    return {
        'name': name,
        'rmse': rmse,
        'mae': mae,
        'direction_acc': direction_correct,
        'total_return': total_return
    }

results = []

# ============================================================
# 1. BASELINE: Always predict mean
# ============================================================
print("\n🔬 Testing models...")
print("-"*70)

y_pred_baseline = np.full_like(y_test, y_train.mean())
results.append(evaluate_model("Baseline (Mean)", y_test, y_pred_baseline))
print(f"✓ Baseline (Mean)")

# ============================================================
# 2. LINEAR MODELS
# ============================================================
for name, model in [
    ("Ridge Regression", Ridge(alpha=1.0)),
    ("Lasso Regression", Lasso(alpha=0.001)),
    ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5)),
]:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results.append(evaluate_model(name, y_test, y_pred))
    print(f"✓ {name}")

# ============================================================
# 3. TREE-BASED MODELS
# ============================================================
for name, model in [
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
]:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results.append(evaluate_model(name, y_test, y_pred))
    print(f"✓ {name}")

# ============================================================
# 4. SVR
# ============================================================
svr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)
results.append(evaluate_model("SVR (RBF)", y_test, y_pred))
print(f"✓ SVR (RBF)")

# ============================================================
# 5. NEURAL NETWORKS (PyTorch)
# ============================================================

# Prepare sequence data for LSTM
def create_sequences(X, y, lookback=10):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

lookback = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq)
X_val_t = torch.FloatTensor(X_val_seq)
y_val_t = torch.FloatTensor(y_val_seq)
X_test_t = torch.FloatTensor(X_test_seq)
y_test_t = torch.FloatTensor(y_test_seq)

# 5a. Simple MLP
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

mlp = MLP(lookback * len(feature_cols))
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    mlp.train()
    optimizer.zero_grad()
    pred = mlp(X_train_t).squeeze()
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

mlp.eval()
with torch.no_grad():
    y_pred = mlp(X_test_t).squeeze().numpy()
results.append(evaluate_model("MLP (3-layer)", y_test_seq, y_pred))
print(f"✓ MLP (3-layer)")

# 5b. LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm = LSTM(len(feature_cols), hidden_size=64, num_layers=2)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

for epoch in range(100):
    lstm.train()
    optimizer.zero_grad()
    pred = lstm(X_train_t).squeeze()
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

lstm.eval()
with torch.no_grad():
    y_pred = lstm(X_test_t).squeeze().numpy()
results.append(evaluate_model("LSTM (2-layer)", y_test_seq, y_pred))
print(f"✓ LSTM (2-layer)")

# 5c. LSTM + Attention
class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)

lstm_attn = LSTMAttention(len(feature_cols), hidden_size=64, num_layers=2)
optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=0.001)

for epoch in range(100):
    lstm_attn.train()
    optimizer.zero_grad()
    pred = lstm_attn(X_train_t).squeeze()
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

lstm_attn.eval()
with torch.no_grad():
    y_pred = lstm_attn(X_test_t).squeeze().numpy()
results.append(evaluate_model("LSTM + Attention", y_test_seq, y_pred))
print(f"✓ LSTM + Attention")

# 5d. GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

gru = GRU(len(feature_cols), hidden_size=64, num_layers=2)
optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)

for epoch in range(100):
    gru.train()
    optimizer.zero_grad()
    pred = gru(X_train_t).squeeze()
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

gru.eval()
with torch.no_grad():
    y_pred = gru(X_test_t).squeeze().numpy()
results.append(evaluate_model("GRU (2-layer)", y_test_seq, y_pred))
print(f"✓ GRU (2-layer)")

# 5e. Transformer
class TransformerPredictor(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

transformer = TransformerPredictor(len(feature_cols), d_model=32, nhead=4, num_layers=2)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

for epoch in range(100):
    transformer.train()
    optimizer.zero_grad()
    pred = transformer(X_train_t).squeeze()
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

transformer.eval()
with torch.no_grad():
    y_pred = transformer(X_test_t).squeeze().numpy()
results.append(evaluate_model("Transformer", y_test_seq, y_pred))
print(f"✓ Transformer")

# ============================================================
# 6. ENSEMBLE: Average of top models
# ============================================================
# We'll create an ensemble after we see results

# ============================================================
# RESULTS
# ============================================================
print("\n" + "="*70)
print("🏆 RESULTS (sorted by Direction Accuracy)")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('direction_acc', ascending=False)

print(f"\n{'Model':<25} {'Dir Acc':>10} {'RMSE':>12} {'MAE':>12} {'Return':>12}")
print("-"*70)
for _, row in results_df.iterrows():
    print(f"{row['name']:<25} {row['direction_acc']*100:>9.1f}% {row['rmse']:>12.6f} {row['mae']:>12.6f} {row['total_return']*100:>11.2f}%")

# Winner
winner = results_df.iloc[0]
print("\n" + "="*70)
print(f"🥇 WINNER: {winner['name']}")
print(f"   Direction Accuracy: {winner['direction_acc']*100:.1f}%")
print(f"   Simulated Return: {winner['total_return']*100:.2f}%")
print("="*70)

# ============================================================
# GENERATE PREDICTIONS WITH BEST MODEL
# ============================================================
print("\n🔮 Generating 7-day predictions with winner...")

# Retrain winner on all data
best_model_name = winner['name']
X_all = scaler.fit_transform(df[feature_cols].values)
y_all = df['Target'].values

if 'LSTM' in best_model_name or 'GRU' in best_model_name or 'Transformer' in best_model_name or 'MLP' in best_model_name:
    # Use the already trained model for prediction
    X_seq, _ = create_sequences(X_all, y_all, lookback)
    last_seq = torch.FloatTensor(X_seq[-1:])
    
    # Select best model
    if 'Attention' in best_model_name:
        best_nn = lstm_attn
    elif 'LSTM' in best_model_name:
        best_nn = lstm
    elif 'GRU' in best_model_name:
        best_nn = gru
    elif 'Transformer' in best_model_name:
        best_nn = transformer
    else:
        best_nn = mlp
    
    best_nn.eval()
    predictions = []
    current_seq = X_all[-lookback:].copy()
    
    with torch.no_grad():
        for _ in range(7):
            x = torch.FloatTensor(current_seq).unsqueeze(0)
            pred = best_nn(x).item()
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1, axis=0)
else:
    # Sklearn model
    if 'Random Forest' in best_model_name:
        best_sk = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif 'Gradient' in best_model_name:
        best_sk = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    elif 'Ridge' in best_model_name:
        best_sk = Ridge(alpha=1.0)
    elif 'SVR' in best_model_name:
        best_sk = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    else:
        best_sk = Ridge(alpha=1.0)
    
    best_sk.fit(X_all[:-1], y_all[:-1])
    
    predictions = []
    current_features = X_all[-1:].copy()
    for _ in range(7):
        pred = best_sk.predict(current_features)[0]
        predictions.append(pred)

# Convert to prices
current_price = df['Close'].iloc[-1]
predicted_prices = [current_price]
for r in predictions:
    predicted_prices.append(predicted_prices[-1] * (1 + r))
predicted_prices = predicted_prices[1:]

shares = 26000
current_value = current_price * shares

print(f"\n📊 CHYM 7-Day Forecast ({best_model_name})")
print("-"*50)
print(f"Current Price: ${current_price:.2f}")
print(f"Position: {shares:,} shares = ${current_value:,.0f}")
print()

for i, (price, ret) in enumerate(zip(predicted_prices, predictions)):
    change = ((price - current_price) / current_price) * 100
    value = price * shares
    pl = value - current_value
    emoji = "🟢" if ret > 0 else "🔴"
    print(f"Day {i+1}: ${price:.2f} ({emoji} {change:+.2f}%) → ${value:,.0f} ({'+' if pl > 0 else ''}{pl:,.0f})")

best_day = np.argmax(predicted_prices) + 1
best_price = max(predicted_prices)
best_profit = (best_price - current_price) * shares

print()
if best_price > current_price:
    print(f"🎯 Best Sell Day: Day {best_day} @ ${best_price:.2f}")
    print(f"   Potential Profit: ${best_profit:+,.0f}")
else:
    worst_price = min(predicted_prices)
    worst_loss = (worst_price - current_price) * shares
    print(f"⚠️  Model predicts continued decline")
    print(f"   Worst predicted: ${worst_price:.2f} (${worst_loss:,.0f})")
    print(f"   Consider: Hold for recovery OR cut losses")

# Save best model info
with open('best_model.txt', 'w') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Direction Accuracy: {winner['direction_acc']*100:.1f}%\n")
    f.write(f"Test Period Return: {winner['total_return']*100:.2f}%\n")
    f.write(f"\nPredictions:\n")
    for i, (price, ret) in enumerate(zip(predicted_prices, predictions)):
        f.write(f"Day {i+1}: ${price:.2f} ({ret*100:+.2f}%)\n")

print("\n✅ Results saved to best_model.txt")
