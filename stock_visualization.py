import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import time

# App Title
st.title("📈 Real-Time Stock Market Data Visualization")

# Load Stock Symbols from a Precompiled List (S&P 500, NASDAQ, etc.)
@st.cache_data
def load_stock_symbols():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

all_stocks = load_stock_symbols()

# User Input for Stock Symbols
symbols = st.multiselect("Select stock symbols:", all_stocks, ["AAPL", "MSFT"])

# Set Different Alerts for Each Stock
alerts = {}
for stock in symbols:
    alerts[stock] = st.number_input(f"Set price alert threshold for {stock}:", min_value=0.0, value=100.0, key=stock)

# Fetch Stock Data using Yahoo Finance API
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1d", interval="1m")
    return hist

# Historical Data using Yahoo Finance API
def get_historical_data(symbol, days):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=f"{days}d")
    return hist

# Toggle between Live and Historical Graphs
view_mode = st.radio("Select View Mode:", ["Live Stock Prices", "Historical Data Comparison"])

# Create columns for live data and historical data side by side
if view_mode == "Live Stock Prices":
    col1, col2 = st.columns(2)

    with col1:  # Live Stock Data
        st.subheader("Live Stock Prices")
        stock_data = {}
        for stock in symbols:
            try:
                data = get_stock_data(stock)
                stock_data[stock] = data
                fig = px.line(data, x=data.index, y='Close', title=f'{stock} Price Trend')
                st.plotly_chart(fig)

                # Alert Logic
                current_price = data['Close'].iloc[-1]
                if current_price >= alerts[stock]:
                    st.warning(f"🚨 {stock} has crossed the threshold! Current Price: ${current_price:.2f}")
            except Exception as e:
                st.error(f"Error fetching data for {stock}: {e}")
            time.sleep(2)  # Delay between requests to prevent hitting rate limits

elif view_mode == "Historical Data Comparison":
    col1, col2 = st.columns(2)

    with col1:  # Historical Data
        st.subheader("Historical Data Comparison")
        history_days = st.slider("Select history days for comparison:", 1, 365, 30)
        for stock in symbols:
            try:
                hist_data = get_historical_data(stock, history_days)
                fig = px.line(hist_data, x=hist_data.index, y='Close', title=f'{stock} Historical Trend')
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error fetching historical data for {stock}: {e}")
            time.sleep(2)  # Delay between requests to prevent hitting rate limits

# Export to CSV
if st.button("Download Stock Data as CSV"):
    all_data = pd.concat([df.assign(Symbol=stock) for stock, df in stock_data.items()], ignore_index=True)
    csv = all_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "stock_data.csv", "text/csv")

# Auto Refresh Every Minute
st.rerun()
