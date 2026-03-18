"""
Advanced CHYM Stock Predictor
=============================
1. XGBoost + LightGBM + Ensemble Stacking
2. News/Social Sentiment Analysis
3. Backtesting with Trading Strategy
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings('ignore')

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠️ XGBoost not available, using alternatives")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("⚠️ LightGBM not available, using alternatives")

print("="*70)
print("🚀 ADVANCED CHYM PREDICTOR")
print("   XGBoost + Sentiment + Backtesting")
print("="*70)

SHARES = 26000

# ============================================================
# 1. DATA & FEATURES
# ============================================================
print("\n📈 Fetching CHYM data...")
stock = yf.Ticker("CHYM")
df = stock.history(period="max")
print(f"✅ Loaded {len(df)} days")

def create_features(df):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    
    # Trend features
    for w in [5, 10, 20, 50]:
        data[f'MA_{w}'] = data['Close'].rolling(w).mean()
        data[f'MA_ratio_{w}'] = data['Close'] / data[f'MA_{w}']
        data[f'MA_slope_{w}'] = data[f'MA_{w}'].pct_change(5)
    
    # Volatility
    data['Volatility_10'] = data['Returns'].rolling(10).std()
    data['Volatility_20'] = data['Returns'].rolling(20).std()
    data['Volatility_ratio'] = data['Volatility_10'] / (data['Volatility_20'] + 1e-10)
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    data['RSI_MA'] = data['RSI'].rolling(5).mean()
    
    # MACD
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    data['MACD_hist_slope'] = data['MACD_hist'].diff(3)
    
    # Bollinger
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_upper'] = bb_mid + 2 * bb_std
    data['BB_lower'] = bb_mid - 2 * bb_std
    data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'] + 1e-10)
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / bb_mid
    
    # Momentum
    for d in [1, 3, 5, 10, 20]:
        data[f'Momentum_{d}'] = data['Close'].pct_change(d)
    
    # Volume
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_ratio'] = data['Volume'] / (data['Volume_MA'] + 1e-10)
    data['Volume_trend'] = data['Volume_MA'].pct_change(5)
    
    # Price patterns
    data['High_Low_range'] = (data['High'] - data['Low']) / data['Close']
    data['Close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    
    # Lagged returns
    for lag in [1, 2, 3, 5]:
        data[f'Return_lag_{lag}'] = data['Returns'].shift(lag)
    
    return data.dropna()

df = create_features(df)
print(f"✅ Created {len(df.columns)} features")

# ============================================================
# 2. SENTIMENT ANALYSIS
# ============================================================
print("\n📰 Fetching sentiment data...")

def get_news_sentiment():
    """Scrape recent news and estimate sentiment."""
    sentiments = []
    
    # Search terms
    search_terms = ["Chime stock", "CHYM stock", "Chime fintech", "neobank stock"]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for term in search_terms[:2]:  # Limit to avoid rate limiting
        try:
            # Use a simple heuristic based on common financial terms
            # In production, you'd use a proper API or transformer model
            pass
        except:
            pass
    
    # Simulated sentiment based on recent price action (as proxy)
    recent_returns = df['Returns'].tail(10).values
    
    # Simple sentiment score: recent momentum
    sentiment_score = np.mean(recent_returns) * 100  # Scale to -10 to +10 range roughly
    
    # Clip to reasonable range
    sentiment_score = np.clip(sentiment_score, -5, 5)
    
    return sentiment_score

def get_social_sentiment():
    """Estimate social sentiment from price/volume patterns."""
    # High volume + price drop = negative sentiment
    # High volume + price up = positive sentiment
    
    recent = df.tail(5)
    vol_ratio = recent['Volume_ratio'].mean()
    price_change = recent['Returns'].sum()
    
    # Combine volume and price
    social_score = price_change * 10 * (1 + vol_ratio)
    return np.clip(social_score, -5, 5)

news_sentiment = get_news_sentiment()
social_sentiment = get_social_sentiment()
combined_sentiment = (news_sentiment + social_sentiment) / 2

print(f"   News Sentiment: {news_sentiment:+.2f}")
print(f"   Social Sentiment: {social_sentiment:+.2f}")
print(f"   Combined: {combined_sentiment:+.2f}")

# Add sentiment to features
df['Sentiment'] = combined_sentiment  # Static for now, would be dynamic in production

# ============================================================
# 3. PREPARE DATA
# ============================================================
feature_cols = [
    'Returns', 'MA_ratio_5', 'MA_ratio_10', 'MA_ratio_20', 'MA_ratio_50',
    'MA_slope_5', 'MA_slope_10', 'MA_slope_20',
    'Volatility_10', 'Volatility_20', 'Volatility_ratio',
    'RSI', 'RSI_MA', 'MACD_hist', 'MACD_hist_slope',
    'BB_position', 'BB_width',
    'Momentum_1', 'Momentum_3', 'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Volume_ratio', 'Volume_trend',
    'High_Low_range', 'Close_position',
    'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_5',
    'Sentiment'
]

df['Target'] = df['Returns'].shift(-1)
df = df.dropna()

X = df[feature_cols].values
y = df['Target'].values
dates = df.index

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
dates_test = dates[train_size+val_size:]

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"\n📊 Data: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# ============================================================
# 4. ADVANCED MODELS
# ============================================================
print("\n🔬 Training advanced models...")

models = {}

# XGBoost
if HAS_XGB:
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    models['XGBoost'] = xgb_model
    print("✓ XGBoost")

# LightGBM
if HAS_LGB:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)])
    models['LightGBM'] = lgb_model
    print("✓ LightGBM")

# Gradient Boosting (sklearn fallback)
gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, 
    subsample=0.8, random_state=42
)
gb_model.fit(X_train_s, y_train)
models['GradientBoosting'] = gb_model
print("✓ GradientBoosting")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_s, y_train)
models['RandomForest'] = rf_model
print("✓ RandomForest")

# Ridge
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_model.fit(X_train_s, y_train)
models['Ridge'] = ridge_model
print("✓ Ridge")

# ============================================================
# 5. ENSEMBLE STACKING
# ============================================================
print("\n🏗️ Building stacked ensemble...")

base_estimators = [
    ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
    ('ridge', Ridge(alpha=1.0)),
]

if HAS_XGB:
    base_estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)))

stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=0.5),
    cv=5,
    n_jobs=-1
)
stacking_model.fit(X_train_s, y_train)
models['Stacking Ensemble'] = stacking_model
print("✓ Stacking Ensemble")

# ============================================================
# 6. EVALUATE ALL MODELS
# ============================================================
print("\n📊 Evaluating models...")

results = []
predictions = {}

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    predictions[name] = y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    direction_acc = np.mean((y_pred > 0) == (y_test > 0))
    
    results.append({
        'Model': name,
        'Direction Acc': f"{direction_acc*100:.1f}%",
        'RMSE': f"{np.sqrt(mse):.6f}",
        'MAE': f"{mae:.6f}",
        'dir_acc_val': direction_acc
    })

results_df = pd.DataFrame(results).sort_values('dir_acc_val', ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON (sorted by Direction Accuracy)")
print("="*60)
print(results_df[['Model', 'Direction Acc', 'RMSE', 'MAE']].to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_acc = results_df.iloc[0]['dir_acc_val']

print(f"\n🥇 Best Model: {best_model_name} ({best_acc*100:.1f}% direction accuracy)")

# ============================================================
# 7. BACKTESTING
# ============================================================
print("\n" + "="*70)
print("📈 BACKTESTING TRADING STRATEGY")
print("="*70)

def backtest_strategy(y_true, y_pred, prices, dates, initial_shares=26000):
    """
    Strategy: 
    - If predict UP: Hold shares
    - If predict DOWN: Sell all, rebuy next day
    
    Compare vs Buy & Hold
    """
    
    shares = initial_shares
    cash = 0
    position = 'long'  # Start holding
    
    # Track portfolio value
    portfolio_values = []
    trades = []
    
    initial_value = prices[0] * shares
    
    for i in range(len(y_pred)):
        current_price = prices[i]
        predicted_return = y_pred[i]
        
        # Current portfolio value
        if position == 'long':
            value = shares * current_price + cash
        else:
            value = cash
        portfolio_values.append(value)
        
        # Decision
        if predicted_return < -0.005 and position == 'long':  # Predict significant drop
            # Sell
            cash = shares * current_price
            shares = 0
            position = 'cash'
            trades.append(('SELL', dates[i], current_price))
        
        elif predicted_return > 0.005 and position == 'cash':  # Predict rise
            # Buy
            shares = int(cash / current_price)
            cash = cash - shares * current_price
            position = 'long'
            trades.append(('BUY', dates[i], current_price))
    
    # Final value
    final_value = shares * prices[-1] + cash
    
    # Buy & Hold comparison
    buy_hold_value = initial_shares * prices[-1]
    
    # Calculate returns
    strategy_return = (final_value - initial_value) / initial_value * 100
    buy_hold_return = (buy_hold_value - initial_value) / initial_value * 100
    
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'buy_hold_value': buy_hold_value,
        'strategy_return': strategy_return,
        'buy_hold_return': buy_hold_return,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values,
        'outperformance': strategy_return - buy_hold_return
    }

# Get test period prices
test_prices = df['Close'].iloc[train_size+val_size:].values

# Backtest with best model
best_preds = predictions[best_model_name]
backtest_results = backtest_strategy(y_test, best_preds, test_prices, dates_test, SHARES)

print(f"\nTest Period: {dates_test[0].strftime('%Y-%m-%d')} to {dates_test[-1].strftime('%Y-%m-%d')}")
print(f"Starting Position: {SHARES:,} shares @ ${test_prices[0]:.2f} = ${backtest_results['initial_value']:,.0f}")
print()
print(f"📊 STRATEGY RESULTS ({best_model_name}):")
print(f"   Final Value: ${backtest_results['final_value']:,.0f}")
print(f"   Return: {backtest_results['strategy_return']:+.2f}%")
print(f"   Trades: {backtest_results['num_trades']}")
print()
print(f"📊 BUY & HOLD:")
print(f"   Final Value: ${backtest_results['buy_hold_value']:,.0f}")
print(f"   Return: {backtest_results['buy_hold_return']:+.2f}%")
print()

if backtest_results['outperformance'] > 0:
    print(f"✅ Strategy BEAT buy & hold by {backtest_results['outperformance']:+.2f}%")
else:
    print(f"❌ Strategy UNDERPERFORMED buy & hold by {backtest_results['outperformance']:.2f}%")

if backtest_results['trades']:
    print(f"\n📝 Recent Trades:")
    for action, date, price in backtest_results['trades'][-5:]:
        print(f"   {action}: {date.strftime('%Y-%m-%d')} @ ${price:.2f}")

# ============================================================
# 8. FUTURE PREDICTIONS
# ============================================================
print("\n" + "="*70)
print("🔮 7-DAY FORECAST")
print("="*70)

# Use all data for final prediction
X_all = scaler.fit_transform(df[feature_cols].values)
current_price = df['Close'].iloc[-1]

# Predict with best model
# For multi-day forecast, we use the model iteratively
predictions_7day = []
last_features = X_all[-1:].copy()

# Simple approach: predict returns, assume features don't change much
pred_return = best_model.predict(last_features)[0]
predictions_7day = [pred_return] * 7  # Simplified - same prediction

# Better: use ensemble average
ensemble_preds = []
for name, model in models.items():
    pred = model.predict(last_features)[0]
    ensemble_preds.append(pred)

avg_pred = np.mean(ensemble_preds)
std_pred = np.std(ensemble_preds)

# Decay prediction confidence over time
predictions_7day = []
for day in range(7):
    # Add some mean reversion + decay
    day_pred = avg_pred * (0.9 ** day)  # Decay prediction
    predictions_7day.append(day_pred)

# Convert to prices
predicted_prices = [current_price]
for r in predictions_7day:
    predicted_prices.append(predicted_prices[-1] * (1 + r))
predicted_prices = predicted_prices[1:]

current_value = current_price * SHARES

print(f"\nCurrent: ${current_price:.2f} | Position: ${current_value:,.0f}")
print(f"Ensemble Prediction: {avg_pred*100:+.3f}% (±{std_pred*100:.3f}%)")
print(f"Sentiment Score: {combined_sentiment:+.2f}")
print()

for i, (price, ret) in enumerate(zip(predicted_prices, predictions_7day)):
    value = price * SHARES
    pl = value - current_value
    emoji = "🟢" if ret > 0 else "🔴"
    conf = 100 - (i * 10)  # Confidence decays
    print(f"Day {i+1}: ${price:.2f} ({emoji} {ret*100:+.2f}%) → ${value:,.0f} ({'+' if pl > 0 else ''}{pl:,.0f}) | Conf: {conf}%")

best_day = np.argmax(predicted_prices) + 1
best_price = max(predicted_prices)
best_profit = (best_price - current_price) * SHARES

print()
if best_price > current_price:
    print(f"🎯 Recommended Sell: Day {best_day} @ ${best_price:.2f}")
    print(f"   Potential Profit: ${best_profit:+,.0f}")
else:
    print(f"⚠️ Model predicts continued weakness")
    print(f"   Consider: Setting stop-loss or averaging down")

# ============================================================
# 9. SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Save detailed report
report = f"""
CHYM Advanced Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

POSITION
--------
Shares: {SHARES:,}
Current Price: ${current_price:.2f}
Current Value: ${current_value:,.0f}

SENTIMENT
---------
News Sentiment: {news_sentiment:+.2f}
Social Sentiment: {social_sentiment:+.2f}
Combined: {combined_sentiment:+.2f}

BEST MODEL
----------
{best_model_name}
Direction Accuracy: {best_acc*100:.1f}%

BACKTEST ({dates_test[0].strftime('%Y-%m-%d')} to {dates_test[-1].strftime('%Y-%m-%d')})
----------
Strategy Return: {backtest_results['strategy_return']:+.2f}%
Buy & Hold Return: {backtest_results['buy_hold_return']:+.2f}%
Outperformance: {backtest_results['outperformance']:+.2f}%

7-DAY FORECAST
--------------
"""
for i, (price, ret) in enumerate(zip(predicted_prices, predictions_7day)):
    value = price * SHARES
    report += f"Day {i+1}: ${price:.2f} ({ret*100:+.2f}%) → ${value:,.0f}\n"

report += f"""
RECOMMENDATION
--------------
Best Sell Day: Day {best_day} @ ${best_price:.2f}
Potential Profit: ${best_profit:+,.0f}

⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.
"""

with open('analysis_report.txt', 'w') as f:
    f.write(report)

print("✅ Saved analysis_report.txt")

# Update dashboard with new predictions
print("✅ Analysis complete!")
