import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit page config
st.set_page_config(layout="wide")
st.title("📊 AI Stock Chart Analyzer")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Load and calculate data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

    # EMA
    df['ema50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['golden_cross'] = ((df['ema50'] > df['ema200']) & (df['ema50'].shift(1) <= df['ema200'].shift(1))).astype(int)

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

    return df.dropna()

df = load_data(ticker, start_date, end_date)

# --- Price Chart with Indicators ---
st.subheader(f"{ticker} Price & Technical Signals")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Close'], label='Close Price')
ax.plot(df['ema50'], label='EMA 50')
ax.plot(df['ema200'], label='EMA 200')

# Golden Cross
ax.scatter(df.index[df['golden_cross'] == 1], 
           df['Close'][df['golden_cross'] == 1], 
           color='green', marker='^', s=100, label='Golden Cross')

# RSI signals
ax.scatter(df.index[df['rsi_oversold'] == 1], 
           df['Close'][df['rsi_oversold'] == 1], 
           color='blue', marker='o', s=70, label='RSI Oversold')

ax.scatter(df.index[df['rsi_overbought'] == 1], 
           df['Close'][df['rsi_overbought'] == 1], 
           color='red', marker='x', s=70, label='RSI Overbought')

# MACD signals
ax.scatter(df.index[df['macd_bullish'] == 1], 
           df['Close'][df['macd_bullish'] == 1], 
           color='lime', marker='v', s=100, label='MACD Bullish')

ax.scatter(df.index[df['macd_bearish'] == 1], 
           df['Close'][df['macd_bearish'] == 1], 
           color='orange', marker='v', s=100, label='MACD Bearish')

ax.legend()
st.pyplot(fig)

# --- RSI Chart ---
st.subheader("📉 RSI Indicator")
fig_rsi, ax_rsi = plt.subplots()
ax_rsi.plot(df['rsi'], label='RSI', color='purple')
ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought')
ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold')
ax_rsi.legend()
st.pyplot(fig_rsi)

# --- MACD Chart ---
st.subheader("📉 MACD Indicator")
fig_macd, ax_macd = plt.subplots()
ax_macd.plot(df['macd'], label='MACD', color='blue')
ax_macd.plot(df['macd_signal'], label='Signal Line', color='orange')
ax_macd.bar(df.index, df['macd_hist'], label='Histogram', color='gray')
ax_macd.legend()
st.pyplot(fig_macd)

# --- AI Model Prediction ---
st.subheader("🤖 AI Model: Next-Day Movement Prediction")

df_ml = df.copy()
X = df_ml[['rsi', 'ema50', 'ema200']]
y = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)

X = X[:-1]
y = y[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.markdown(f"**Model Accuracy:** `{accuracy:.2f}`")

latest_features = X.tail(1)
prediction = model.predict(latest_features)[0]
st.markdown(f"**📊 Tomorrow's Prediction:** {'📈 Up' if prediction == 1 else '📉 Down'}")
