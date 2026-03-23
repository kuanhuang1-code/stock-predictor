"""
Interactive Stock Predictor Dashboard
=====================================
Streamlit app to explore CHYM predictions with your 26k shares.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(
    page_title="CHYM Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# MODEL DEFINITION
# ============================================================
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


# ============================================================
# DATA FUNCTIONS
# ============================================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str):
    """Fetch stock data with caching."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if not df.empty:
            return df, None
    except Exception as e:
        return None, str(e)
    return None, "No data found"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the model."""
    data = df.copy()
    
    data['Returns'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    
    # Bollinger
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * bb_std
    data['BB_Lower'] = data['BB_Mid'] - 2 * bb_std
    
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
    
    return data.dropna()


def train_model(df, lookback, hidden_size, num_layers, epochs, progress_bar):
    """Train the model and return predictions."""
    
    feature_cols = ['Close', 'Volume', 'Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'Volume_Ratio']
    
    data = df.copy()
    data['Target'] = data['Returns'].shift(-1)
    data = data.dropna()
    
    # Scale
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
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Model
    model = StockPredictor(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        progress_bar.progress((epoch + 1) / epochs, f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")
    
    return model, scaler, feature_cols, train_losses, val_losses, data


def predict_future(model, df, scaler, feature_cols, lookback, days):
    """Generate future predictions."""
    model.eval()
    
    features = scaler.transform(df[feature_cols].values)
    last_sequence = features[-lookback:]
    last_price = df['Close'].iloc[-1]
    
    predictions = []
    predicted_returns = []
    
    with torch.no_grad():
        current_seq = last_sequence.copy()
        for _ in range(days):
            x = torch.FloatTensor(current_seq).unsqueeze(0)
            pred_return = model(x).item()
            predicted_returns.append(pred_return)
            current_seq = np.roll(current_seq, -1, axis=0)
    
    # Convert to prices
    prices = [last_price]
    for ret in predicted_returns:
        prices.append(prices[-1] * (1 + ret))
    
    return prices[1:], predicted_returns


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("📈 CHYM Stock Predictor")
st.markdown("**Your Position: 26,000 shares** | LSTM + Attention Model")

# Sidebar controls
st.sidebar.header("⚙️ Model Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="CHYM")
shares = st.sidebar.number_input("Shares Owned", value=26000, step=1000)
lookback = st.sidebar.slider("Lookback Days", 5, 60, 20)
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 5)
hidden_size = st.sidebar.select_slider("Model Size", [32, 64, 128, 256], value=128)
num_layers = st.sidebar.slider("LSTM Layers", 1, 4, 2)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)

# Fetch data
with st.spinner(f"Fetching {ticker} data..."):
    df_raw, error = fetch_stock_data(ticker)

if error or df_raw is None or df_raw.empty:
    st.error(f"❌ Could not fetch {ticker}: {error}")
    st.info("Try AAPL, MSFT, GOOGL, or SPY as alternatives")
    st.stop()

df = create_features(df_raw)
current_price = df['Close'].iloc[-1]

# Display current stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Position Value", f"${current_price * shares:,.0f}")
col3.metric("Today's Change", f"{df['Returns'].iloc[-1]*100:.2f}%")
col4.metric("Data Points", f"{len(df)} days")

st.divider()

# Train button
if st.button("🚀 Train Model & Predict", type="primary"):
    
    progress_bar = st.progress(0, "Starting training...")
    
    model, scaler, feature_cols, train_losses, val_losses, processed_df = train_model(
        df, lookback, hidden_size, num_layers, epochs, progress_bar
    )
    
    progress_bar.empty()
    st.success("✅ Training complete!")
    
    # Training chart
    st.subheader("📉 Training Progress")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=train_losses, name="Train Loss", line=dict(color="blue")))
    fig_loss.add_trace(go.Scatter(y=val_losses, name="Val Loss", line=dict(color="red")))
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss (MSE)", height=300)
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Predictions
    st.subheader("🔮 Price Predictions")
    predicted_prices, predicted_returns = predict_future(
        model, processed_df, scaler, feature_cols, lookback, forecast_days
    )
    
    # Prediction table
    last_date = df.index[-1]
    pred_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(forecast_days)],
        "Predicted Price": [f"${p:.2f}" for p in predicted_prices],
        "Change": [f"{r*100:+.2f}%" for r in predicted_returns],
        "Position Value": [f"${p * shares:,.0f}" for p in predicted_prices],
        "P/L": [f"${(p - current_price) * shares:+,.0f}" for p in predicted_prices],
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Best day to sell
    best_idx = np.argmax(predicted_prices)
    best_price = predicted_prices[best_idx]
    best_profit = (best_price - current_price) * shares
    
    if best_price > current_price:
        st.success(f"🎯 **Best Sell Day: Day {best_idx + 1}** @ ${best_price:.2f} → Profit: ${best_profit:+,.0f}")
    else:
        st.warning("⚠️ Model predicts decline - consider selling soon or holding longer")
    
    # Price chart with predictions
    st.subheader("📊 Price History & Forecast")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Price", "RSI", "Volume"))
    
    # Historical prices
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df_raw.loc[df.index, 'Open'],
        high=df_raw.loc[df.index, 'High'],
        low=df_raw.loc[df.index, 'Low'],
        close=df['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dash'), name="BB Upper", opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dash'), name="BB Lower", fill='tonexty', opacity=0.2), row=1, col=1)
    
    # Predicted prices
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
        mode='lines+markers',
        name="Predicted",
        line=dict(color='red', dash='dash', width=3),
        marker=dict(size=10)
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['green' if df['Returns'].iloc[i] > 0 else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume", opacity=0.7), row=3, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    with st.expander("🧠 Model Architecture"):
        st.code(f"""
StockPredictor(
  LSTM: input={len(feature_cols)} → hidden={hidden_size} × {num_layers} layers
  Attention: {hidden_size} → {hidden_size//2} → 1
  Output: {hidden_size} → {hidden_size//2} → 1
  
  Total Parameters: {sum(p.numel() for p in model.parameters()):,}
)

Features Used:
{chr(10).join(f'  - {f}' for f in feature_cols)}
        """)
    
    with st.expander("📚 How It Works"):
        st.markdown("""
        ### The Training Loop (PyTorch Lightning Style)
        
        ```python
        for epoch in range(epochs):
            # Forward pass
            predictions = model(sequences)
            
            # Compute loss
            loss = MSELoss(predictions, actual_returns)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        ```
        
        ### Model Architecture
        
        1. **LSTM Layers**: Learn temporal patterns in stock data
        2. **Attention**: Focus on the most important days in the sequence  
        3. **Dense Layers**: Convert patterns to predicted returns
        
        ### Features
        
        - Price data (close, returns)
        - Moving averages (5, 20 day)
        - Technical indicators (RSI, MACD)
        - Volatility & Volume patterns
        
        ### ⚠️ Disclaimer
        
        This is for learning ML, not financial advice. Stock markets are unpredictable.
        """)

else:
    # Show current chart without predictions
    st.subheader("📊 Current Price History")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df_raw.loc[df.index, 'Open'],
        high=df_raw.loc[df.index, 'High'],
        low=df_raw.loc[df.index, 'Low'],
        close=df['Close'],
        name="Price"
    ))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("👆 Click **Train Model & Predict** to generate forecasts")


# Footer
st.divider()
st.caption("Built with PyTorch + Streamlit | Data from Yahoo Finance")
