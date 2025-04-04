import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import time
from datetime import datetime
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customization
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-text {
        color: #616161;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Cache management functions
def clear_cache():
    """Clear all st.cache_data caches"""
    st.cache_data.clear()

class StockDataManager:
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_stock_symbols():
        """Load stock symbols from S&P 500 constituents"""
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            df = pd.read_csv(url)
            # Add company names for better UI
            symbols_with_names = {row['Symbol']: f"{row['Symbol']} - {row['Name']}" 
                                 for _, row in df.iterrows()}
            return symbols_with_names
        except Exception as e:
            logger.error(f"Error loading stock symbols: {e}")
            return {"AAPL": "AAPL - Apple Inc.", 
                    "MSFT": "MSFT - Microsoft Corp.", 
                    "GOOGL": "GOOGL - Alphabet Inc."}

    @staticmethod
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_live_stock_data(symbol):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d", interval="1m")
            if hist.empty:
                raise ValueError(f"No data returned for {symbol}")
            return hist
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            raise

    @staticmethod
    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def get_historical_data(symbol, days):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=f"{days}d")
            if hist.empty:
                raise ValueError(f"No historical data returned for {symbol}")
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    @staticmethod
    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def get_stock_info(symbol):
        """Fetch stock information and fundamentals"""
        try:
            stock = yf.Ticker(symbol)
            info = {k: v for k, v in stock.info.items() 
                   if k in ['sector', 'industry', 'marketCap', 'trailingPE', 
                           'dividendYield', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow']}
            return info
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return {}

class StockDataVisualizer:
    @staticmethod
    def create_price_chart(data, symbol, include_volume=True):
        """Create an interactive price chart with optional volume data"""
        if include_volume:
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name="Price", line=dict(color="#1E88E5")),
                secondary_y=False,
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(x=data.index, y=data['Volume'], name="Volume", marker=dict(color="#90CAF9")),
                secondary_y=True,
            )
            
            # Set titles
            fig.update_layout(
                title=f"{symbol} Price and Volume",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            fig.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Volume", secondary_y=True)
            
        else:
            fig = px.line(data, x=data.index, y='Close', title=f'{symbol} Price')
            fig.update_layout(
                xaxis_title="Date", 
                yaxis_title="Price ($)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
        return fig

    @staticmethod
    def create_multi_stock_comparison(stock_data_dict):
        """Create a comparison chart for multiple stocks"""
        fig = go.Figure()
        
        for symbol, data in stock_data_dict.items():
            # Normalize to percentage change from first day
            normalized = data['Close'] / data['Close'].iloc[0] * 100 - 100
            fig.add_trace(go.Scatter(
                x=data.index,
                y=normalized,
                mode='lines',
                name=symbol
            ))
            
        fig.update_layout(
            title="Percentage Change Comparison",
            xaxis_title="Date",
            yaxis_title="% Change",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        return fig

    @staticmethod
    def display_stock_metrics(data, symbol, alert_threshold=None):
        """Display key metrics for a stock with styling using Streamlit native components"""
        try:
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
            percent_change = (price_change / data['Close'].iloc[0]) * 100
        except Exception as e:
            st.error(f"Error calculating metrics for {symbol}: {e}")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(f"{symbol} Price")
            st.metric(label="Current", value=f"${current_price:.2f}")
            
        with col2:
            st.subheader("Change")
            st.metric(
                label="$ Change", 
                value=f"${abs(price_change):.2f}",
                delta=f"{price_change:.2f}"
            )
            
        with col3:
            st.subheader("Percentage")
            st.metric(
                label="% Change", 
                value=f"{abs(percent_change):.2f}%",
                delta=f"{percent_change:.2f}%"
            )

def fetch_data_with_retry(fetch_function, *args, max_retries=3):
    """Retry function with exponential backoff"""
    for i in range(max_retries):
        try:
            return fetch_function(*args)
        except Exception as e:
            if i == max_retries - 1:
                raise e
            wait_time = random.uniform(1, 2) * (2 ** i)  # Exponential backoff
            time.sleep(wait_time)

def fetch_data_parallel(symbols, fetch_function, *args):
    """Fetch data for multiple symbols in parallel"""
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a dictionary mapping futures to their corresponding symbols
        future_to_symbol = {
            executor.submit(fetch_data_with_retry, fetch_function, symbol, *args): symbol 
            for symbol in symbols
        }
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
                
    return results

def render_sidebar():
    """Render the sidebar with app controls"""
    with st.sidebar:
        st.markdown("<h1 class='main-header'>📈 Stock Dashboard</h1>", unsafe_allow_html=True)
        
        # App sections
        st.markdown("<h2 class='subheader'>Settings</h2>", unsafe_allow_html=True)
        
        # Stock Selection
        stock_options = StockDataManager.load_stock_symbols()
        default_stocks = ["AAPL", "MSFT"] if "selected_stocks" not in st.session_state else st.session_state.selected_stocks
        
        selected_stocks = st.multiselect(
            "Select stocks to monitor:",
            options=list(stock_options.keys()),
            default=default_stocks,
            format_func=lambda x: stock_options[x]
        )
        
        if "selected_stocks" not in st.session_state or st.session_state.selected_stocks != selected_stocks:
            st.session_state.selected_stocks = selected_stocks
        
        # Alert Settings
        st.markdown("<h2 class='subheader'>Price Alerts</h2>", unsafe_allow_html=True)
        
        alerts = {}
        for stock in selected_stocks:
            # Get last known price as default value
            default_value = 100.0
            try:
                data = StockDataManager.get_live_stock_data(stock)
                default_value = data['Close'].iloc[-1] * 1.05  # 5% above current price
            except:
                pass
                
            alerts[stock] = st.number_input(
                f"Alert threshold for {stock}:",
                min_value=0.0,
                value=default_value,
                step=1.0,
                format="%.2f",
                key=f"alert_{stock}"
            )
        
        # View Mode
        st.markdown("<h2 class='subheader'>View Options</h2>", unsafe_allow_html=True)
        view_mode = st.radio(
            "Select view mode:",
            ["Live Prices", "Historical Comparison", "Technical Analysis"],
            key="view_mode"
        )
        
        if view_mode == "Historical Comparison":
            history_days = st.slider("Historical period (days):", 7, 365, 30)
            st.session_state.history_days = history_days
        
        # Refresh Button
        if st.button("🔄 Refresh Data"):
            clear_cache()
            st.rerun()
        
        # App info
        st.markdown("---")
        st.markdown(
            "<p class='info-text'>Data provided by Yahoo Finance.</p>", 
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p class='info-text'>Last updated: {datetime.now().strftime('%H:%M:%S')}</p>",
            unsafe_allow_html=True
        )

def render_live_prices(selected_stocks, alerts):
    """Render the live prices view"""
    st.markdown("<h1 class='main-header'>Live Stock Prices</h1>", unsafe_allow_html=True)
    
    if not selected_stocks:
        st.info("Please select at least one stock in the sidebar.")
        return
        
    # Show a spinner while loading data
    with st.spinner("Fetching latest stock data..."):
        # Fetch data in parallel
        stock_data = fetch_data_parallel(selected_stocks, StockDataManager.get_live_stock_data)
    
    # Display metrics and charts
    for symbol, data in stock_data.items():
        st.markdown(f"<h2 class='subheader'>{symbol}</h2>", unsafe_allow_html=True)
        
        # Display metrics
        StockDataVisualizer.display_stock_metrics(data, symbol, alerts.get(symbol))
        
        # Display chart
        fig = StockDataVisualizer.create_price_chart(data, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add horizontal line
        st.markdown("---")

def render_historical_comparison(selected_stocks):
    """Render the historical comparison view"""
    st.markdown("<h1 class='main-header'>Historical Performance Comparison</h1>", unsafe_allow_html=True)
    
    if not selected_stocks:
        st.info("Please select at least one stock in the sidebar.")
        return
    
    history_days = st.session_state.get("history_days", 30)
    
    # Show a spinner while loading data
    with st.spinner(f"Fetching {history_days} days of historical data..."):
        # Fetch data in parallel
        stock_data = fetch_data_parallel(selected_stocks, StockDataManager.get_historical_data, history_days)
    
    if stock_data:
        # Create comparison chart
        comparison_fig = StockDataVisualizer.create_multi_stock_comparison(stock_data)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Individual stock charts
        st.markdown("<h2 class='subheader'>Individual Stock Performance</h2>", unsafe_allow_html=True)
        
        # Create two columns layout
        cols = st.columns(2)
        
        # Distribute stocks between columns
        for i, (symbol, data) in enumerate(stock_data.items()):
            with cols[i % 2]:
                st.subheader(symbol)
                fig = StockDataVisualizer.create_price_chart(data, symbol, include_volume=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display key statistics
                with st.expander("Stock Information"):
                    info = StockDataManager.get_stock_info(symbol)
                    if info:
                        info_df = pd.DataFrame([info]).T
                        info_df.columns = ["Value"]
                        st.dataframe(info_df)
                    else:
                        st.info("No additional information available")

def render_technical_analysis(selected_stocks):
    """Render technical analysis view"""
    st.markdown("<h1 class='main-header'>Technical Analysis</h1>", unsafe_allow_html=True)
    
    if not selected_stocks:
        st.info("Please select at least one stock in the sidebar.")
        return
    
    selected_stock = st.selectbox("Select stock for analysis:", selected_stocks)
    
    # Technical indicators
    indicators = st.multiselect(
        "Select technical indicators:",
        ["Moving Averages", "Bollinger Bands", "RSI", "MACD"],
        default=["Moving Averages"]
    )
    
    # Period selection
    period = st.radio("Select time period:", ["1 Month", "3 Months", "6 Months", "1 Year"], index=1)
    period_days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}[period]
    
    # Show a spinner while loading data
    with st.spinner(f"Calculating technical indicators for {selected_stock}..."):
        try:
            data = StockDataManager.get_historical_data(selected_stock, period_days)
            
            # Create figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, 
                               row_heights=[0.7, 0.3])
            
            # Add price
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Add indicators
            if "Moving Averages" in indicators:
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MA20'], name="MA20", line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(color='red')),
                    row=1, col=1
                )
            
            if "Bollinger Bands" in indicators:
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['Upper'] = data['MA20'] + (data['Close'].rolling(window=20).std() * 2)
                data['Lower'] = data['MA20'] - (data['Close'].rolling(window=20).std() * 2)
                
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['Upper'], name="Upper Band",
                              line=dict(color='rgba(0,0,255,0.3)')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['Lower'], name="Lower Band",
                              line=dict(color='rgba(0,0,255,0.3)'),
                              fill='tonexty', fillcolor='rgba(0,0,255,0.1)'),
                    row=1, col=1
                )
            
            if "RSI" in indicators:
                # Calculate RSI
                delta = data['Close'].diff()
                gain = delta.mask(delta < 0, 0)
                loss = -delta.mask(delta > 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['RSI'], name="RSI"),
                    row=2, col=1
                )
                
                # Add RSI reference lines
                fig.add_shape(
                    type="line", line_color="red", line_width=1, opacity=0.5,
                    y0=70, y1=70, x0=data.index[0], x1=data.index[-1],
                    xref="x2", yref="y2"
                )
                fig.add_shape(
                    type="line", line_color="green", line_width=1, opacity=0.5,
                    y0=30, y1=30, x0=data.index[0], x1=data.index[-1],
                    xref="x2", yref="y2"
                )
            
            if "MACD" in indicators:
                # Calculate MACD
                data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Histogram'] = data['MACD'] - data['Signal']
                
                # If RSI is already plotted, we need to adjust
                if "RSI" in indicators:
                    # Remove the existing trace in row 2
                    fig.data = fig.data[:-1]
                    
                    # Create a new row
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                       vertical_spacing=0.1, 
                                       row_heights=[0.6, 0.2, 0.2])
                    
                    # Re-add the price and indicators
                    for trace in fig.data:
                        fig.add_trace(trace, row=1, col=1)
                    
                    # Add RSI to row 2
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['RSI'], name="RSI"),
                        row=2, col=1
                    )
                    
                    # Add MACD to row 3
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MACD'], name="MACD"),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['Signal'], name="Signal"),
                        row=3, col=1
                    )
                    
                    # Add histogram as bars
                    colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
                    fig.add_trace(
                        go.Bar(x=data.index, y=data['Histogram'], name="Histogram", marker_color=colors),
                        row=3, col=1
                    )
                else:
                    # Add MACD to row 2
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MACD'], name="MACD"),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['Signal'], name="Signal"),
                        row=2, col=1
                    )
                    
                    # Add histogram as bars
                    colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
                    fig.add_trace(
                        go.Bar(x=data.index, y=data['Histogram'], name="Histogram", marker_color=colors),
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f"{selected_stock} Technical Analysis",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=800,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error performing technical analysis: {e}")

def render_export_options(stock_data):
    """Render data export options"""
    st.markdown("---")
    st.markdown("<h2 class='subheader'>Export Data</h2>", unsafe_allow_html=True)
    
    if not stock_data:
        st.info("No data available to export.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV"):
            # Prepare data for export
            all_data = pd.concat([df.assign(Symbol=symbol) for symbol, df in stock_data.items()], ignore_index=False)
            
            # Generate CSV
            csv = all_data.to_csv().encode('utf-8')
            
            # Provide download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="stock_data.csv",
                mime="text/csv",
            )
    
    with col2:
        if st.button("📄 Export to Excel"):
            # Create a Pandas Excel writer
            output = pd.ExcelWriter("stock_data.xlsx", engine='xlsxwriter')
            
            # Write each dataframe to a different worksheet
            for symbol, df in stock_data.items():
                df.to_excel(output, sheet_name=symbol)
            
            output.close()
            
            with open("stock_data.xlsx", "rb") as f:
                excel_data = f.read()
                
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="stock_data.xlsx",
                mime="application/vnd.ms-excel",
            )

def main():
    """Main application function"""
    # Initialize session state
    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = ["AAPL", "MSFT"]
    
    # Render sidebar
    render_sidebar()
    
    # Get values from session state
    selected_stocks = st.session_state.selected_stocks
    
    # Get view mode
    view_mode = st.session_state.get("view_mode", "Live Prices")
    
    # Get price alerts
    alerts = {stock: st.session_state.get(f"alert_{stock}", 100.0) for stock in selected_stocks}
    
    # Main content based on view mode
    if view_mode == "Live Prices":
        render_live_prices(selected_stocks, alerts)
    elif view_mode == "Historical Comparison":
        render_historical_comparison(selected_stocks)
    elif view_mode == "Technical Analysis":
        render_technical_analysis(selected_stocks)
    
    # Get stock data for export (if needed)
    stock_data = {}
    if selected_stocks:
        if view_mode == "Live Prices":
            try:
                stock_data = {symbol: StockDataManager.get_live_stock_data(symbol) 
                             for symbol in selected_stocks}
            except:
                pass
        else:
            history_days = st.session_state.get("history_days", 30)
            try:
                stock_data = {symbol: StockDataManager.get_historical_data(symbol, history_days) 
                             for symbol in selected_stocks}
            except:
                pass
    
    # Render export options if we have data
    render_export_options(stock_data)

if __name__ == "__main__":
    main()