"""
Generate static HTML dashboard with live CHYM predictions.
12-month forecast with monthly hold/sell recommendations.
Uses market context (SPY, VIX, sector, rates) + proper early stopping.
Runs daily via GitHub Actions to update predictions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json

# ============================================================
# MODEL
# ============================================================
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4), nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_size // 4, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


# ============================================================
# DATA: fetch stock + market context
# ============================================================
def fetch_all_data(ticker="CHYM"):
    """Fetch target stock plus market/macro variables."""
    print(f"  Fetching {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")

    if df.empty:
        print(f"  {ticker} unavailable, using PYPL as fallback...")
        df = yf.Ticker("PYPL").history(period="2y")

    # Market context tickers
    context_tickers = {
        'SPY': 'SPY',       # S&P 500 — broad market direction
        'VIX': '^VIX',      # Volatility index — fear gauge
        'XLF': 'XLF',       # Financials sector ETF — Chime's sector
        'QQQ': 'QQQ',       # Nasdaq — tech/growth sentiment
        'TLT': 'TLT',       # 20+ yr Treasury — interest rate proxy
        'HYG': 'HYG',       # High yield bonds — credit risk appetite
    }

    for name, ctx_ticker in context_tickers.items():
        print(f"  Fetching {name} ({ctx_ticker})...")
        try:
            ctx = yf.Ticker(ctx_ticker).history(period="2y")
            if not ctx.empty:
                # Reindex to match target stock dates, forward-fill gaps
                ctx = ctx.reindex(df.index, method='ffill')
                df[f'{name}_Close'] = ctx['Close']
                df[f'{name}_Return'] = ctx['Close'].pct_change()
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    return df


# ============================================================
# FEATURES: technical + market + macro
# ============================================================
def create_features(df):
    data = df.copy()

    # --- Price action ---
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # --- Moving averages (multiple timeframes) ---
    for window in [5, 10, 20, 50]:
        data[f'MA_{window}'] = data['Close'].rolling(window).mean()
        data[f'MA_{window}_Ratio'] = data['Close'] / data[f'MA_{window}']

    # --- Exponential moving averages ---
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()

    # --- Volatility (multiple timeframes) ---
    data['Volatility_10'] = data['Returns'].rolling(10).std()
    data['Volatility_20'] = data['Returns'].rolling(20).std()
    data['Volatility_50'] = data['Returns'].rolling(50).std()

    # --- RSI ---
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # --- MACD + Signal line ---
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # --- Bollinger Bands ---
    bb_ma = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = bb_ma + 2 * bb_std
    data['BB_Lower'] = bb_ma - 2 * bb_std
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-10)

    # --- Volume ---
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    data['Volume_Trend'] = data['Volume'].rolling(5).mean() / data['Volume'].rolling(20).mean()

    # --- Momentum ---
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1

    # --- Trend strength (ADX proxy via directional movement) ---
    high = data['High'] if 'High' in data.columns else data['Close']
    low = data['Low'] if 'Low' in data.columns else data['Close']
    data['Range'] = (high - low) / data['Close']
    data['Range_MA'] = data['Range'].rolling(14).mean()

    # --- Market context features ---
    # SPY relative strength
    if 'SPY_Close' in data.columns:
        data['SPY_Relative'] = data['Returns'] - data['SPY_Return']
        data['SPY_Corr_20'] = data['Returns'].rolling(20).corr(data['SPY_Return'])
        data['SPY_MA_Ratio'] = data['SPY_Close'] / data['SPY_Close'].rolling(50).mean()

    # VIX level and trend
    if 'VIX_Close' in data.columns:
        data['VIX_Level'] = data['VIX_Close']
        data['VIX_MA_Ratio'] = data['VIX_Close'] / data['VIX_Close'].rolling(20).mean()
        data['VIX_Change'] = data['VIX_Close'].pct_change()

    # Sector (financials) relative strength
    if 'XLF_Close' in data.columns:
        data['Sector_Relative'] = data['Returns'] - data['XLF_Return']
        data['Sector_MA_Ratio'] = data['XLF_Close'] / data['XLF_Close'].rolling(50).mean()

    # Tech sentiment
    if 'QQQ_Close' in data.columns:
        data['QQQ_MA_Ratio'] = data['QQQ_Close'] / data['QQQ_Close'].rolling(50).mean()

    # Interest rate proxy (inverse TLT = rising rates)
    if 'TLT_Close' in data.columns:
        data['Rate_Trend'] = -data['TLT_Return']  # Inverse: TLT down = rates up
        data['TLT_MA_Ratio'] = data['TLT_Close'] / data['TLT_Close'].rolling(50).mean()

    # Credit risk appetite
    if 'HYG_Close' in data.columns:
        data['Credit_Trend'] = data['HYG_Return']
        data['HYG_MA_Ratio'] = data['HYG_Close'] / data['HYG_Close'].rolling(50).mean()

    # Drop raw context price columns (keep derived features only)
    drop_cols = [c for c in data.columns if c.endswith('_Close') and c != 'Close']
    drop_cols += [c for c in data.columns if c.endswith('_Return') and c != 'Returns']
    drop_cols += ['EMA_12', 'EMA_26', 'High', 'Low', 'Open', 'Dividends', 'Stock Splits']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # Drop intermediate MA columns (keep ratios)
    for window in [5, 10, 20, 50]:
        if f'MA_{window}' in data.columns:
            data = data.drop(columns=[f'MA_{window}'])

    data = data.dropna()
    return data


# ============================================================
# TRAINING with proper val split + early stopping
# ============================================================
def train_and_predict(df, lookback=60, forecast_days=252, shares=26000):
    # Determine feature columns (everything except Close, Volume, Target)
    exclude = {'Close', 'Volume', 'Target'}
    feature_cols = ['Close', 'Volume'] + [c for c in df.columns if c not in exclude]
    # Deduplicate while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if c not in seen and not seen.add(c)]

    data = df.copy()
    data['Target'] = data['Returns'].shift(-1)
    data = data.dropna()

    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    scaler = MinMaxScaler()
    features = scaler.fit_transform(data[feature_cols].values)
    targets = data['Target'].values

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(targets[i])

    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)

    # --- Train / Validation split (80/20, time-ordered) ---
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    model = StockPredictor(input_size=len(feature_cols), hidden_size=256, num_layers=3, dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    # --- Early stopping on validation loss ---
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_state = None
    min_delta = 1e-6  # Minimum improvement to count

    max_epochs = 500
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

            # Direction accuracy on validation
            val_dir = ((val_pred > 0) == (y_val > 0)).float().mean().item()

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}: train={train_loss:.6f} val={val_loss:.6f} dir_acc={val_dir:.1%} lr={lr:.1e} patience={patience - patience_counter}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (val_loss={best_val_loss:.6f})")
            break
    else:
        print(f"  Reached max epochs ({max_epochs})")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # --- Predict ---
    model.eval()
    last_sequence = features[-lookback:]
    current_price = df['Close'].iloc[-1]

    predictions = []
    with torch.no_grad():
        seq = last_sequence.copy()
        for _ in range(forecast_days):
            x = torch.FloatTensor(seq).unsqueeze(0)
            ret = model(x).item()
            # Clamp daily moves to +/-5% for stability
            ret = max(min(ret, 0.05), -0.05)
            predictions.append(ret)
            seq = np.roll(seq, -1, axis=0)

    # Convert to prices
    prices = [current_price]
    for r in predictions:
        prices.append(prices[-1] * (1 + r))

    return {
        'current_price': current_price,
        'predicted_prices': prices[1:],
        'predicted_returns': predictions,
        'shares': shares,
        'current_value': current_price * shares,
        'history': df['Close'].tail(120).to_dict(),
        'dates': [d.strftime('%Y-%m-%d') for d in df.index[-120:]],
        'val_direction_acc': val_dir,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'num_features': len(feature_cols),
    }


# ============================================================
# SIGNAL LOGIC: multi-factor
# ============================================================
def compute_monthly_summary(data):
    """Aggregate daily predictions into monthly summaries with multi-factor signals."""
    current = data['current_price']
    prices = data['predicted_prices']
    returns = data['predicted_returns']
    shares = data['shares']

    months = []
    for m in range(12):
        start = m * 21
        end = min((m + 1) * 21, len(prices))
        if start >= len(prices):
            break
        month_prices = prices[start:end]
        month_returns = returns[start:end]
        end_price = month_prices[-1]
        high_price = np.max(month_prices)
        low_price = np.min(month_prices)
        change_pct = (end_price - current) / current * 100
        pl = (end_price - current) * shares

        # Month-over-month momentum
        if m > 0:
            prev_end = prices[start - 1]
            mom = (end_price - prev_end) / prev_end * 100
        else:
            mom = change_pct

        # Volatility within month
        month_vol = np.std(month_returns) * np.sqrt(21) * 100  # Annualized-ish

        # Trend consistency: what fraction of days were positive
        pos_days = sum(1 for r in month_returns if r > 0) / len(month_returns)

        # Drawdown from current price
        drawdown = (low_price - current) / current * 100

        # Signal scoring (multi-factor)
        score = 0
        reasons = []

        # Factor 1: Cumulative return vs current
        if change_pct > 10:
            score += 3
            reasons.append(f"strong gain ({change_pct:+.1f}%)")
        elif change_pct > 5:
            score += 2
            reasons.append(f"solid gain ({change_pct:+.1f}%)")
        elif change_pct > 2:
            score += 1
            reasons.append(f"modest gain ({change_pct:+.1f}%)")
        elif change_pct > -2:
            reasons.append("flat")
        elif change_pct > -5:
            score -= 1
            reasons.append(f"declining ({change_pct:+.1f}%)")
        else:
            score -= 2
            reasons.append(f"significant drop ({change_pct:+.1f}%)")

        # Factor 2: Momentum direction
        if mom < -3:
            score -= 1
            reasons.append("losing momentum")
        elif mom > 3:
            score += 1
            reasons.append("gaining momentum")

        # Factor 3: Trend consistency
        if pos_days < 0.35:
            score -= 1
            reasons.append(f"weak trend ({pos_days:.0%} up days)")
        elif pos_days > 0.6:
            score += 1
            reasons.append(f"strong trend ({pos_days:.0%} up days)")

        # Factor 4: Volatility risk
        if month_vol > 40:
            score -= 1
            reasons.append("high volatility")

        # Factor 5: Drawdown risk
        if drawdown < -10:
            score -= 1
            reasons.append(f"drawdown risk ({drawdown:.1f}%)")

        # Convert score to signal
        if score >= 2 and change_pct > 5:
            signal = "SELL"
            signal_reason = "Take profits — " + ", ".join(reasons[:2])
        elif score >= 1:
            signal = "HOLD"
            signal_reason = "Upside potential — " + ", ".join(reasons[:2])
        elif score == 0:
            signal = "HOLD"
            signal_reason = "Neutral — " + ", ".join(reasons[:2])
        elif score >= -1 and change_pct > -3:
            signal = "HOLD"
            signal_reason = "Caution — " + ", ".join(reasons[:2])
        else:
            signal = "SELL"
            signal_reason = "Cut losses — " + ", ".join(reasons[:2])

        months.append({
            'month': m + 1,
            'end_price': end_price,
            'high': high_price,
            'low': low_price,
            'change_pct': change_pct,
            'pl': pl,
            'signal': signal,
            'reason': signal_reason,
            'score': score,
            'momentum': mom,
            'volatility': month_vol,
        })

    return months


# ============================================================
# HTML GENERATION
# ============================================================
def generate_html(data):
    current = data['current_price']
    prices = data['predicted_prices']
    shares = data['shares']

    months = compute_monthly_summary(data)

    best_month = max(months, key=lambda m: m['end_price'])
    worst_month = min(months, key=lambda m: m['end_price'])
    best_price = best_month['end_price']
    best_profit = (best_price - current) * shares
    final_price = months[-1]['end_price']
    final_change = (final_price - current) / current * 100

    # Overall recommendation
    sell_months = [m for m in months if m['signal'] == 'SELL' and m['change_pct'] > 0]
    if sell_months:
        first_sell = sell_months[0]
        overall_rec = f"Best time to sell: Month {best_month['month']} @ ${best_price:.2f} (profit: ${best_profit:+,.0f})"
        overall_class = "positive"
    elif final_change > 0:
        overall_rec = f"Hold — gradual upside expected. Peak at Month {best_month['month']} @ ${best_price:.2f}"
        overall_class = "positive"
    else:
        overall_rec = f"Consider selling soon — model predicts {final_change:.1f}% decline over 12 months"
        overall_class = "negative"

    # Model info
    model_info = f"Trained {data['epochs_trained']} epochs | {data['num_features']} features | Val direction accuracy: {data['val_direction_acc']:.1%}"

    # Monthly forecast rows
    month_rows = ""
    month_labels = []
    now = datetime.now()
    for m in months:
        target_date = now + timedelta(days=m['month'] * 30)
        month_name = target_date.strftime('%b %Y')
        month_labels.append(month_name)
        change_class = "positive" if m['change_pct'] > 0 else "negative"
        signal_badge_bg = "rgba(0,200,83,0.2)" if m['signal'] == 'HOLD' else ("rgba(255,152,0,0.2)" if m['change_pct'] > 0 else "rgba(255,82,82,0.2)")
        signal_color = "#00c853" if m['signal'] == 'HOLD' else ("#ff9800" if m['change_pct'] > 0 else "#ff5252")
        month_rows += f"""
        <tr>
            <td>{month_name}</td>
            <td>${m['end_price']:.2f}</td>
            <td>${m['high']:.2f}</td>
            <td>${m['low']:.2f}</td>
            <td class="{change_class}">{m['change_pct']:+.1f}%</td>
            <td class="{change_class}">${m['pl']:+,.0f}</td>
            <td><span style="background:{signal_badge_bg}; color:{signal_color}; padding:4px 12px; border-radius:20px; font-weight:bold;">{m['signal']}</span></td>
            <td style="color:#888; font-size:0.85em;">{m['reason']}</td>
        </tr>"""

    # Chart data
    monthly_end_prices = [m['end_price'] for m in months]
    monthly_highs = [m['high'] for m in months]
    monthly_lows = [m['low'] for m in months]

    hist_dates = data['dates']
    hist_prices_raw = list(data['history'].values())
    hist_monthly_dates = hist_dates[::21]
    hist_monthly_prices = hist_prices_raw[::21]
    if hist_dates[-1] not in hist_monthly_dates:
        hist_monthly_dates.append(hist_dates[-1])
        hist_monthly_prices.append(hist_prices_raw[-1])

    all_labels = hist_monthly_dates + month_labels
    hist_chart_data = hist_monthly_prices + [None] * len(month_labels)
    pred_chart_data = [None] * (len(hist_monthly_prices) - 1) + [hist_monthly_prices[-1]] + monthly_end_prices
    high_chart_data = [None] * (len(hist_monthly_prices) - 1) + [hist_monthly_prices[-1]] + monthly_highs
    low_chart_data = [None] * (len(hist_monthly_prices) - 1) + [hist_monthly_prices[-1]] + monthly_lows

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CHYM 12-Month Stock Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        h2 {{ margin-bottom: 15px; }}
        .subtitle {{ color: #888; margin-bottom: 5px; }}
        .model-info {{ color: #666; font-size: 0.8em; margin-bottom: 25px; font-style: italic; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        .stat-label {{ color: #888; font-size: 0.9em; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; margin-top: 5px; }}
        .chart-container {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ background: rgba(255,255,255,0.1); }}
        .positive {{ color: #00c853; }}
        .negative {{ color: #ff5252; }}
        .recommendation {{
            background: rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 12px;
            font-size: 1.3em;
            text-align: center;
            margin-bottom: 30px;
        }}
        .signal-legend {{
            display: flex;
            gap: 30px;
            justify-content: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .signal-legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
            color: #aaa;
        }}
        .signal-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; font-size: 0.9em; }}
        .update-time {{ color: #888; font-size: 0.85em; margin-top: 5px; }}
        .table-scroll {{ overflow-x: auto; }}
        .factors {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 30px; }}
        .factors h3 {{ margin-bottom: 10px; color: #aaa; }}
        .factor-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }}
        .factor-item {{ display: flex; justify-content: space-between; padding: 8px 12px; background: rgba(255,255,255,0.05); border-radius: 8px; }}
        .factor-name {{ color: #888; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CHYM 12-Month Stock Forecast</h1>
        <p class="subtitle">LSTM + Attention Model | Your Position: {shares:,} shares</p>
        <p class="model-info">{model_info}</p>
        <p class="update-time">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Current Price</div>
                <div class="stat-value">${current:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Position Value</div>
                <div class="stat-value">${current * shares:,.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">12-Month Target</div>
                <div class="stat-value {'positive' if final_change > 0 else 'negative'}">${final_price:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak Predicted</div>
                <div class="stat-value">${best_price:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">12-Month Change</div>
                <div class="stat-value {'positive' if final_change > 0 else 'negative'}">{final_change:+.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Potential P/L</div>
                <div class="stat-value {'positive' if best_profit > 0 else 'negative'}">${best_profit:+,.0f}</div>
            </div>
        </div>

        <div class="recommendation {overall_class}">
            {overall_rec}
        </div>

        <div class="chart-container">
            <canvas id="priceChart"></canvas>
        </div>

        <div class="signal-legend">
            <div class="signal-legend-item">
                <div class="signal-dot" style="background:#00c853;"></div>
                HOLD — Keep your position
            </div>
            <div class="signal-legend-item">
                <div class="signal-dot" style="background:#ff9800;"></div>
                SELL (profit) — Take profits
            </div>
            <div class="signal-legend-item">
                <div class="signal-dot" style="background:#ff5252;"></div>
                SELL (loss) — Cut losses
            </div>
        </div>

        <h2>Monthly Forecast</h2>
        <div class="table-scroll">
        <table>
            <thead>
                <tr>
                    <th>Month</th>
                    <th>Price</th>
                    <th>High</th>
                    <th>Low</th>
                    <th>vs Today</th>
                    <th>P/L</th>
                    <th>Signal</th>
                    <th>Reasoning</th>
                </tr>
            </thead>
            <tbody>
                {month_rows}
            </tbody>
        </table>
        </div>

        <div class="footer">
            <p>Built with PyTorch LSTM + Attention | Market data: SPY, VIX, XLF, QQQ, TLT, HYG | Data from Yahoo Finance</p>
            <p style="margin-top: 10px;">This is for learning ML, not financial advice. Stock markets are unpredictable. Past performance does not guarantee future results.</p>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('priceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(all_labels)},
                datasets: [{{
                    label: 'Historical Price',
                    data: {json.dumps(hist_chart_data)},
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4
                }}, {{
                    label: 'Predicted Price',
                    data: {json.dumps(pred_chart_data)},
                    borderColor: '#ff7043',
                    borderDash: [5, 5],
                    pointBackgroundColor: '#ff7043',
                    pointRadius: 5,
                    fill: false,
                    tension: 0.3
                }}, {{
                    label: 'Predicted High',
                    data: {json.dumps(high_chart_data)},
                    borderColor: 'rgba(0, 200, 83, 0.3)',
                    backgroundColor: 'rgba(0, 200, 83, 0.05)',
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3
                }}, {{
                    label: 'Predicted Low',
                    data: {json.dumps(low_chart_data)},
                    borderColor: 'rgba(255, 82, 82, 0.3)',
                    backgroundColor: 'rgba(255, 82, 82, 0.05)',
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: '-1',
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }},
                    title: {{
                        display: true,
                        text: 'CHYM Price: Historical + 12-Month Forecast',
                        color: '#eee',
                        font: {{ size: 16 }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#888', maxRotation: 45 }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        ticks: {{ color: '#888', callback: v => '$' + v.toFixed(2) }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    return html


def main():
    print("Fetching CHYM + market data (2 years)...")
    df = fetch_all_data("CHYM")

    df = create_features(df)
    print(f"Loaded {len(df)} days, {len(df.columns)} columns")

    print("Training model (early stopping on val loss)...")
    data = train_and_predict(df, lookback=60, forecast_days=252, shares=26000)

    print("Generating dashboard...")
    html = generate_html(data)

    with open("index.html", "w") as f:
        f.write(html)

    print("Dashboard saved to index.html")


if __name__ == "__main__":
    main()
