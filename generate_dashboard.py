"""
Generate static HTML dashboard with live CHYM predictions.
Runs daily via GitHub Actions to update predictions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json

# ============================================================
# MODEL
# ============================================================
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1))
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


def fetch_data(ticker="CHYM"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    return df


def create_features(df):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['Volatility'] = data['Returns'].rolling(20).std()
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    return data.dropna()


def train_and_predict(df, lookback=20, forecast_days=7, shares=26000):
    feature_cols = ['Close', 'Volume', 'Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'Volume_Ratio']
    
    data = df.copy()
    data['Target'] = data['Returns'].shift(-1)
    data = data.dropna()
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(data[feature_cols].values)
    targets = data['Target'].values
    
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(targets[i])
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    model = StockPredictor(input_size=len(feature_cols), hidden_size=128, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    # Predict
    model.eval()
    last_sequence = features[-lookback:]
    current_price = df['Close'].iloc[-1]
    
    predictions = []
    with torch.no_grad():
        seq = last_sequence.copy()
        for _ in range(forecast_days):
            x = torch.FloatTensor(seq).unsqueeze(0)
            ret = model(x).item()
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
        'history': df['Close'].tail(60).to_dict(),
        'dates': [d.strftime('%Y-%m-%d') for d in df.index[-60:]],
    }


def generate_html(data):
    current = data['current_price']
    prices = data['predicted_prices']
    returns = data['predicted_returns']
    shares = data['shares']
    
    best_idx = np.argmax(prices)
    best_price = prices[best_idx]
    best_profit = (best_price - current) * shares
    
    # Generate prediction rows
    pred_rows = ""
    for i, (p, r) in enumerate(zip(prices, returns)):
        change_class = "positive" if r > 0 else "negative"
        pl = (p - current) * shares
        pred_rows += f"""
        <tr>
            <td>Day {i+1}</td>
            <td>${p:.2f}</td>
            <td class="{change_class}">{r*100:+.2f}%</td>
            <td>${p * shares:,.0f}</td>
            <td class="{change_class}">${pl:+,.0f}</td>
        </tr>"""
    
    # Historical data for chart
    hist_dates = data['dates']
    hist_prices = list(data['history'].values())
    
    # Future dates
    last_date = datetime.strptime(hist_dates[-1], '%Y-%m-%d')
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(prices))]
    
    recommendation = f"🎯 Best Sell Day: Day {best_idx + 1} @ ${best_price:.2f} → Profit: ${best_profit:+,.0f}" if best_price > current else "⚠️ Model predicts decline - consider holding or selling soon"
    rec_class = "positive" if best_price > current else "negative"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CHYM Stock Predictor</title>
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
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ color: #888; margin-bottom: 30px; }}
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
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
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
        .footer {{ text-align: center; color: #666; margin-top: 40px; font-size: 0.9em; }}
        .update-time {{ color: #888; font-size: 0.85em; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 CHYM Stock Predictor</h1>
        <p class="subtitle">LSTM + Attention Model | Your Position: {shares:,} shares</p>
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
                <div class="stat-label">Best Predicted</div>
                <div class="stat-value">${best_price:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Potential P/L</div>
                <div class="stat-value {rec_class}">${best_profit:+,.0f}</div>
            </div>
        </div>
        
        <div class="recommendation {rec_class}">
            {recommendation}
        </div>
        
        <div class="chart-container">
            <canvas id="priceChart"></canvas>
        </div>
        
        <h2 style="margin-bottom: 15px;">📅 7-Day Forecast</h2>
        <table>
            <thead>
                <tr>
                    <th>Day</th>
                    <th>Predicted Price</th>
                    <th>Change</th>
                    <th>Position Value</th>
                    <th>P/L</th>
                </tr>
            </thead>
            <tbody>
                {pred_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Built with PyTorch | Data from Yahoo Finance</p>
            <p style="margin-top: 10px;">⚠️ This is for learning ML, not financial advice. Stock markets are unpredictable.</p>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('priceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(hist_dates[-30:] + future_dates)},
                datasets: [{{
                    label: 'Historical Price',
                    data: {json.dumps(hist_prices[-30:] + [None] * len(prices))},
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    fill: true,
                    tension: 0.1
                }}, {{
                    label: 'Predicted Price',
                    data: {json.dumps([None] * 29 + [hist_prices[-1]] + prices)},
                    borderColor: '#ff7043',
                    borderDash: [5, 5],
                    pointBackgroundColor: '#ff7043',
                    pointRadius: 6,
                    fill: false,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }},
                scales: {{
                    x: {{ 
                        ticks: {{ color: '#888' }},
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
    print("📈 Fetching CHYM data...")
    df = fetch_data("CHYM")
    
    if df.empty:
        print("⚠️ Using PYPL as fallback...")
        df = fetch_data("PYPL")
    
    df = create_features(df)
    print(f"✅ Loaded {len(df)} days")
    
    print("🧠 Training model...")
    data = train_and_predict(df, lookback=20, forecast_days=7, shares=26000)
    
    print("📝 Generating dashboard...")
    html = generate_html(data)
    
    with open("index.html", "w") as f:
        f.write(html)
    
    print("✅ Dashboard saved to index.html")


if __name__ == "__main__":
    main()
