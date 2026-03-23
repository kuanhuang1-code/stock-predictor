"""
Generate static HTML dashboard with live CHYM predictions.
12-month forecast with monthly hold/sell recommendations.
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
    df = stock.history(period="2y")
    return df


def create_features(df):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
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


def train_and_predict(df, lookback=60, forecast_days=252, shares=26000):
    feature_cols = ['Close', 'Volume', 'Returns', 'MA_5', 'MA_20', 'MA_50',
                    'Volatility', 'RSI', 'MACD', 'Volume_Ratio']

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

    model = StockPredictor(input_size=len(feature_cols), hidden_size=256, num_layers=2, dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

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
            # Clamp extreme predictions for stability over long horizon
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
    }


def compute_monthly_summary(data):
    """Aggregate daily predictions into monthly summaries with hold/sell signals."""
    current = data['current_price']
    prices = data['predicted_prices']
    shares = data['shares']

    # ~21 trading days per month
    months = []
    for m in range(12):
        start = m * 21
        end = min((m + 1) * 21, len(prices))
        if start >= len(prices):
            break
        month_prices = prices[start:end]
        avg_price = np.mean(month_prices)
        end_price = month_prices[-1]
        high_price = np.max(month_prices)
        low_price = np.min(month_prices)
        change_pct = (end_price - current) / current * 100
        pl = (end_price - current) * shares

        # Signal logic
        if change_pct > 5:
            signal = "SELL"
            signal_reason = "Strong gain — consider taking profits"
        elif change_pct > 2:
            signal = "HOLD"
            signal_reason = "Moderate upside — hold for more"
        elif change_pct > -2:
            signal = "HOLD"
            signal_reason = "Flat — no clear action"
        elif change_pct > -5:
            signal = "SELL"
            signal_reason = "Declining — consider selling"
        else:
            signal = "SELL"
            signal_reason = "Significant drop predicted"

        months.append({
            'month': m + 1,
            'avg_price': avg_price,
            'end_price': end_price,
            'high': high_price,
            'low': low_price,
            'change_pct': change_pct,
            'pl': pl,
            'signal': signal,
            'reason': signal_reason,
        })

    return months


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
        overall_rec = f"Best time to sell: Month {best_month['month']} @ ${best_price:.2f} (profit: ${best_profit:+,.0f})"
        overall_class = "positive"
    elif final_change > 0:
        overall_rec = f"Hold — gradual upside expected. Peak at Month {best_month['month']} @ ${best_price:.2f}"
        overall_class = "positive"
    else:
        overall_rec = f"Consider selling soon — model predicts {final_change:.1f}% decline over 12 months"
        overall_class = "negative"

    # Monthly forecast rows
    month_rows = ""
    month_labels = []
    now = datetime.now()
    for m in months:
        target_date = now + timedelta(days=m['month'] * 30)
        month_name = target_date.strftime('%b %Y')
        month_labels.append(month_name)
        change_class = "positive" if m['change_pct'] > 0 else "negative"
        signal_class = "positive" if m['signal'] == 'HOLD' else ("positive" if m['change_pct'] > 0 else "negative")
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

    # Chart data: monthly end prices
    monthly_end_prices = [m['end_price'] for m in months]
    monthly_highs = [m['high'] for m in months]
    monthly_lows = [m['low'] for m in months]

    # Historical monthly-ish data (sample every ~21 days)
    hist_dates = data['dates']
    hist_prices_raw = list(data['history'].values())
    # Sample historical at monthly intervals
    hist_monthly_dates = hist_dates[::21]
    hist_monthly_prices = hist_prices_raw[::21]
    # Make sure we include the last point
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
    </style>
</head>
<body>
    <div class="container">
        <h1>CHYM 12-Month Stock Forecast</h1>
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
            <p>Built with PyTorch LSTM + Attention | Data from Yahoo Finance</p>
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
    print("Fetching CHYM data (2 years)...")
    df = fetch_data("CHYM")

    if df.empty:
        print("Using PYPL as fallback...")
        df = fetch_data("PYPL")

    df = create_features(df)
    print(f"Loaded {len(df)} days of data")

    print("Training model (12-month forecast)...")
    data = train_and_predict(df, lookback=60, forecast_days=252, shares=26000)

    print("Generating dashboard...")
    html = generate_html(data)

    with open("index.html", "w") as f:
        f.write(html)

    print("Dashboard saved to index.html")


if __name__ == "__main__":
    main()
