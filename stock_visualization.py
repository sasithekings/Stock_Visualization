import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import time
from datetime import datetime, timedelta
import random
import logging
import io
import base64

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

# CSS customization with fullscreen support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
        text-align: center;
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
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .fullscreen-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
        background: rgba(255,255,255,0.8);
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache management functions
def clear_cache():
    """Clear all st.cache_data caches"""
    try:
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing cache: {e}")

class StockDataManager:
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_stock_symbols():
        """Load stock symbols from S&P 500 constituents"""
        try:
            # Primary source
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            df = pd.read_csv(url)
            symbols_with_names = {row['Symbol']: f"{row['Symbol']} - {row['Name']}" 
                                 for _, row in df.iterrows()}
            logger.info(f"Successfully loaded {len(symbols_with_names)} stock symbols")
            return symbols_with_names
        except Exception as e:
            logger.error(f"Error loading stock symbols from primary source: {e}")
            # Fallback to a hardcoded list of popular stocks
            fallback_stocks = {
                "AAPL": "AAPL - Apple Inc.",
                "MSFT": "MSFT - Microsoft Corp.",
                "GOOGL": "GOOGL - Alphabet Inc.",
                "AMZN": "AMZN - Amazon.com Inc.",
                "TSLA": "TSLA - Tesla Inc.",
                "META": "META - Meta Platforms Inc.",
                "NVDA": "NVDA - NVIDIA Corp.",
                "JPM": "JPM - JPMorgan Chase & Co.",
                "JNJ": "JNJ - Johnson & Johnson",
                "V": "V - Visa Inc."
            }
            logger.info("Using fallback stock list")
            return fallback_stocks

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes for live data
    def get_live_stock_data(symbol):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            # Try to get intraday data first
            hist = stock.history(period="1d", interval="5m")
            if hist.empty:
                # Fallback to daily data
                hist = stock.history(period="5d", interval="1d")
            if hist.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in hist.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            logger.info(f"Successfully fetched live data for {symbol}: {len(hist)} records")
            return hist
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            raise

    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
    def get_historical_data(symbol, period="1mo"):
        """Fetch historical stock data with improved period handling"""
        try:
            stock = yf.Ticker(symbol)
            
            # Convert days to yfinance period format
            if isinstance(period, int):
                if period <= 5:
                    period_str = "5d"
                elif period <= 30:
                    period_str = "1mo"
                elif period <= 90:
                    period_str = "3mo"
                elif period <= 180:
                    period_str = "6mo"
                elif period <= 365:
                    period_str = "1y"
                else:
                    period_str = "2y"
            else:
                period_str = period
            
            hist = stock.history(period=period_str)
            if hist.empty:
                raise ValueError(f"No historical data returned for {symbol}")
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in hist.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            logger.info(f"Successfully fetched historical data for {symbol}: {len(hist)} records")
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_stock_info(symbol):
        """Fetch stock information and fundamentals"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key information with error handling
            safe_info = {}
            info_keys = ['sector', 'industry', 'marketCap', 'trailingPE', 
                        'dividendYield', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
                        'longName', 'currency', 'country', 'website']
            
            for key in info_keys:
                safe_info[key] = info.get(key, 'N/A')
            
            # Format market cap
            if safe_info['marketCap'] != 'N/A' and safe_info['marketCap']:
                try:
                    market_cap = float(safe_info['marketCap'])
                    if market_cap >= 1e12:
                        safe_info['marketCap'] = f"${market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        safe_info['marketCap'] = f"${market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        safe_info['marketCap'] = f"${market_cap/1e6:.2f}M"
                except:
                    pass
            
            logger.info(f"Successfully fetched stock info for {symbol}")
            return safe_info
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return {}

class StockDataVisualizer:
    @staticmethod
    def create_price_chart(data, symbol, chart_type="line", include_volume=True):
        """Create an interactive price chart with multiple chart types"""
        try:
            if include_volume and len(data) > 0:
                # Create figure with secondary y-axis
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{symbol} Price", "Volume")
                )
                
                # Add price chart based on type
                if chart_type == "candlestick" and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price",
                            increasing_line_color='#00ff00',
                            decreasing_line_color='#ff0000'
                        ),
                        row=1, col=1
                    )
                else:
                    # Line chart
                    fig.add_trace(
                        go.Scatter(
                            x=data.index, 
                            y=data['Close'], 
                            name="Price", 
                            line=dict(color="#1E88E5", width=2)
                        ),
                        row=1, col=1
                    )
                
                # Add volume bars
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(data['Close'], data['Open'])]
                fig.add_trace(
                    go.Bar(
                        x=data.index, 
                        y=data['Volume'], 
                        name="Volume", 
                        marker=dict(color=colors, opacity=0.7)
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol} - {chart_type.title()} Chart with Volume",
                    xaxis_title="Date",
                    height=600,
                    margin=dict(l=20, r=20, t=50, b=20),
                    showlegend=True,
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=1.02, 
                        xanchor="right", 
                        x=1
                    ),
                    hovermode='x unified'
                )
                
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
            else:
                # Simple price chart without volume
                fig = go.Figure()
                
                if chart_type == "candlestick" and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price"
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index, 
                            y=data['Close'], 
                            name="Price", 
                            line=dict(color="#1E88E5", width=2)
                        )
                    )
                
                fig.update_layout(
                    title=f"{symbol} - {chart_type.title()} Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    hovermode='x unified'
                )
            
            # Add fullscreen button functionality
            fig.update_layout(
                modebar_add=['pan', 'zoom', 'select', 'lasso', 'autoScale', 'resetScale']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart for {symbol}: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig

    @staticmethod
    def create_multi_stock_comparison(stock_data_dict):
        """Create a comparison chart for multiple stocks"""
        try:
            fig = go.Figure()
            
            colors = ['#1E88E5', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
            color_idx = 0
            
            for symbol, data in stock_data_dict.items():
                if len(data) > 0:
                    # Normalize to percentage change from first day
                    first_price = data['Close'].iloc[0]
                    normalized = ((data['Close'] / first_price) - 1) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[color_idx % len(colors)], width=2),
                        hovertemplate=f'<b>{symbol}</b><br>Date: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>'
                    ))
                    color_idx += 1
            
            fig.update_layout(
                title="Stock Performance Comparison (% Change)",
                xaxis_title="Date",
                yaxis_title="Percentage Change (%)",
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1
                ),
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode='x unified'
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating comparison chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig

    @staticmethod
    def display_stock_metrics(data, symbol, info=None, alert_threshold=None):
        """Display key metrics for a stock with enhanced styling"""
        try:
            if len(data) == 0:
                st.error(f"No data available for {symbol}")
                return
                
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[0]
            price_change = current_price - prev_price
            percent_change = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # Calculate additional metrics
            high_52w = data['High'].max() if len(data) > 0 else current_price
            low_52w = data['Low'].min() if len(data) > 0 else current_price
            volume = data['Volume'].iloc[-1] if len(data) > 0 else 0
            avg_volume = data['Volume'].mean() if len(data) > 0 else 0
            
            # Main metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=f"💰 {symbol} Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change:+.2f} ({percent_change:+.2f}%)"
                )
                
            with col2:
                st.metric(
                    label="📊 Volume",
                    value=f"{volume:,.0f}",
                    delta=f"{((volume/avg_volume - 1) * 100):+.1f}% vs avg" if avg_volume > 0 else None
                )
                
            with col3:
                st.metric(
                    label="📈 52W High",
                    value=f"${high_52w:.2f}",
                    delta=f"{((current_price/high_52w - 1) * 100):.1f}% from high"
                )
                
            with col4:
                st.metric(
                    label="📉 52W Low",
                    value=f"${low_52w:.2f}",
                    delta=f"{((current_price/low_52w - 1) * 100):+.1f}% from low"
                )
            
            # Alert check
            if alert_threshold and alert_threshold > 0:
                if current_price >= alert_threshold:
                    st.markdown(
                        f'<div class="alert-box alert-success">🎯 <strong>Price Alert!</strong> {symbol} has reached your target price of ${alert_threshold:.2f}</div>',
                        unsafe_allow_html=True
                    )
                elif current_price >= alert_threshold * 0.95:
                    st.markdown(
                        f'<div class="alert-box alert-warning">⚠️ <strong>Approaching Target!</strong> {symbol} is close to your target price of ${alert_threshold:.2f}</div>',
                        unsafe_allow_html=True
                    )
            
            # Additional info if available
            if info and any(v != 'N/A' for v in info.values()):
                with st.expander("📋 Company Information"):
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        if info.get('longName', 'N/A') != 'N/A':
                            st.write(f"**Company:** {info['longName']}")
                        if info.get('sector', 'N/A') != 'N/A':
                            st.write(f"**Sector:** {info['sector']}")
                        if info.get('industry', 'N/A') != 'N/A':
                            st.write(f"**Industry:** {info['industry']}")
                        if info.get('country', 'N/A') != 'N/A':
                            st.write(f"**Country:** {info['country']}")
                    
                    with info_col2:
                        if info.get('marketCap', 'N/A') != 'N/A':
                            st.write(f"**Market Cap:** {info['marketCap']}")
                        if info.get('trailingPE', 'N/A') != 'N/A':
                            st.write(f"**P/E Ratio:** {info['trailingPE']}")
                        if info.get('dividendYield', 'N/A') != 'N/A' and info['dividendYield']:
                            st.write(f"**Dividend Yield:** {info['dividendYield']:.2%}")
                        if info.get('website', 'N/A') != 'N/A':
                            st.write(f"**Website:** [Link]({info['website']})")
            
        except Exception as e:
            st.error(f"Error displaying metrics for {symbol}: {e}")
            logger.error(f"Error in display_stock_metrics for {symbol}: {e}")

def fetch_data_with_retry(fetch_function, *args, max_retries=3):
    """Retry function with exponential backoff"""
    for i in range(max_retries):
        try:
            result = fetch_function(*args)
            return result
        except Exception as e:
            if i == max_retries - 1:
                logger.error(f"Failed after {max_retries} retries: {e}")
                raise e
            wait_time = random.uniform(1, 3) * (2 ** i)  # Exponential backoff
            logger.warning(f"Retry {i+1}/{max_retries} after {wait_time:.1f}s: {e}")
            time.sleep(wait_time)

def fetch_data_parallel(symbols, fetch_function, *args, max_workers=5):
    """Fetch data for multiple symbols in parallel with improved error handling"""
    results = {}
    errors = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures
        future_to_symbol = {
            executor.submit(fetch_data_with_retry, fetch_function, symbol, *args): symbol 
            for symbol in symbols
        }
        
        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_symbol, timeout=30):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
                logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                error_msg = f"Failed to fetch data for {symbol}: {str(e)}"
                errors[symbol] = error_msg
                logger.error(error_msg)
    
    # Display errors if any
    if errors:
        st.warning(f"Failed to fetch data for: {', '.join(errors.keys())}")
        for symbol, error in errors.items():
            st.error(f"{symbol}: {error}")
    
    return results

def render_sidebar():
    """Render the sidebar with app controls"""
    with st.sidebar:
        st.markdown("<h1 class='main-header'>📈 Stock Dashboard</h1>", unsafe_allow_html=True)
        
        # Load stock symbols
        try:
            stock_options = StockDataManager.load_stock_symbols()
        except Exception as e:
            st.error(f"Error loading stock symbols: {e}")
            stock_options = {"AAPL": "AAPL - Apple Inc."}
        
        # Stock Selection
        st.markdown("### 🏢 Stock Selection")
        default_stocks = st.session_state.get("selected_stocks", ["AAPL", "MSFT"])
        
        selected_stocks = st.multiselect(
            "Select stocks to monitor:",
            options=list(stock_options.keys()),
            default=[s for s in default_stocks if s in stock_options],
            format_func=lambda x: stock_options.get(x, x),
            help="Choose up to 10 stocks for analysis"
        )
        
        # Limit selection
        if len(selected_stocks) > 10:
            st.warning("Maximum 10 stocks allowed. Please remove some selections.")
            selected_stocks = selected_stocks[:10]
        
        st.session_state.selected_stocks = selected_stocks
        
        # Chart Settings
        st.markdown("### 📊 Chart Settings")
        chart_type = st.radio(
            "Chart Type:",
            ["line", "candlestick"],
            format_func=lambda x: "📈 Line Chart" if x == "line" else "🕯️ Candlestick Chart",
            help="Choose between line charts and candlestick charts"
        )
        st.session_state.chart_type = chart_type
        
        include_volume = st.checkbox("Include Volume", value=True, help="Show volume data below price chart")
        st.session_state.include_volume = include_volume
        
        # View Mode
        st.markdown("### 👁️ View Options")
        view_mode = st.radio(
            "Select view mode:",
            ["Live Prices", "Historical Comparison", "Technical Analysis"],
            help="Choose your preferred analysis view"
        )
        st.session_state.view_mode = view_mode
        
        # Historical period for comparison view
        if view_mode == "Historical Comparison":
            history_period = st.selectbox(
                "Historical Period:",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                format_func=lambda x: {
                    "1mo": "1 Month",
                    "3mo": "3 Months", 
                    "6mo": "6 Months",
                    "1y": "1 Year",
                    "2y": "2 Years"
                }[x],
                help="Select the time period for historical comparison"
            )
            st.session_state.history_period = history_period
        
        # Alert Settings
        st.markdown("### 🚨 Price Alerts")
        alerts = {}
        
        if selected_stocks:
            for stock in selected_stocks[:3]:  # Limit alerts to first 3 stocks
                try:
                    # Try to get current price for default
                    current_data = StockDataManager.get_live_stock_data(stock)
                    current_price = current_data['Close'].iloc[-1]
                    default_alert = current_price * 1.05  # 5% above current
                except:
                    default_alert = 100.0
                
                alerts[stock] = st.number_input(
                    f"Alert for {stock}:",
                    min_value=0.01,
                    value=default_alert,
                    step=0.01,
                    format="%.2f",
                    key=f"alert_{stock}",
                    help=f"Get notified when {stock} reaches this price"
                )
        
        st.session_state.alerts = alerts
        
        # Control buttons
        st.markdown("### ⚙️ Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", help="Refresh all data"):
                clear_cache()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", help="Clear all selections"):
                for key in list(st.session_state.keys()):
                    if key.startswith(('selected_stocks', 'alerts', 'chart_type')):
                        del st.session_state[key]
                st.rerun()
        
        # App info
        st.markdown("---")
        st.markdown("### ℹ️ Information")
        st.info("📊 Data provided by Yahoo Finance")
        st.info(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")
        st.info("💡 Use fullscreen mode in charts for better viewing")

def render_live_prices(selected_stocks, alerts, chart_type, include_volume):
    """Render the live prices view with enhanced error handling"""
    st.markdown("<h1 class='main-header'>📊 Live Stock Prices</h1>", unsafe_allow_html=True)
    
    if not selected_stocks:
        st.info("👈 Please select at least one stock in the sidebar to get started.")
        return
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Fetch data with progress updates
        status_text.text("🔄 Fetching live stock data...")
        progress_bar.progress(25)
        
        stock_data = fetch_data_parallel(selected_stocks, StockDataManager.get_live_stock_data)
        progress_bar.progress(75)
        
        # Fetch stock info
        stock_info = fetch_data_parallel(selected_stocks, StockDataManager.get_stock_info)
        progress_bar.progress(100)
        
        status_text.text("✅ Data loaded successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        if not stock_data:
            st.error("❌ No data could be fetched. Please try again later.")
            return
        
        # Display data for each stock
        for i, (symbol, data) in enumerate(stock_data.items()):
            st.markdown(f"### 📈 {symbol}")
            
            # Display metrics
            info = stock_info.get(symbol, {})
            StockDataVisualizer.display_stock_metrics(
                data, symbol, info, alerts.get(symbol)
            )
            
            # Display chart with fullscreen option
            col1, col2 = st.columns([10, 1])
            
            with col1:
                fig = StockDataVisualizer.create_price_chart(
                    data, symbol, chart_type, include_volume
                )
                st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{symbol}")
            
            with col2:
                if st.button("🔍", key=f"fullscreen_{symbol}", help="View in fullscreen"):
                    # Create a fullscreen version
                    fig_full = StockDataVisualizer.create_price_chart(
                        data, symbol, chart_type, include_volume
                    )
                    fig_full.update_layout(height=800)
                    
                    # Show in a container that takes full width
                    with st.container():
                        st.plotly_chart(fig_full, use_container_width=True)
            
            # Add separator between stocks
            if i < len(stock_data) - 1:
                st.markdown("---")
        
        # Store data for export
        st.session_state.current_stock_data = stock_data
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error loading live prices: {str(e)}")
        logger.error(f"Error in render_live_prices: {e}")

def render_historical_comparison(selected_stocks, history_period, chart_type, include_volume):
    """Render the historical comparison view with enhanced features"""
    st.markdown("<h1 class='main-header'>📊 Historical Performance Comparison</h1>", unsafe_allow_html=True)
    
    if not selected_stocks:
        st.info("👈 Please select at least one stock in the sidebar to get started.")
        return
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Fetch historical data
        status_text.text(f"🔄 Fetching {history_period} historical data...")
        progress_bar.progress(25)
        
        stock_data = fetch_data_parallel(selected_stocks, StockDataManager.get_historical_data, history_period)
        progress_bar.progress(75)
        
        # Fetch stock info
        stock_info = fetch_data_parallel(selected_stocks, StockDataManager.get_stock_info)
        progress_bar.progress(100)
        
        status_text.text("✅ Data loaded successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        if not stock_data:
            st.error("❌ No historical data could be fetched. Please try again later.")
            return
        
        # Create comparison chart
        st.markdown("### 📈 Performance Comparison")
        comparison_fig = StockDataVisualizer.create_multi_stock_comparison(stock_data)
        
        col1, col2 = st.columns([10, 1])
        with col1:
            st.plotly_chart(comparison_fig, use_container_width=True, key="comparison_chart")
        with col2:
            if st.button("🔍", key="fullscreen_comparison", help="View comparison in fullscreen"):
                comparison_fig.update_layout(height=800)
                st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Performance summary table
        st.markdown("### 📊 Performance Summary")
        summary_data = []
        for symbol, data in stock_data.items():
            if len(data) > 0:
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                total_return = ((end_price / start_price) - 1) * 100
                volatility = data['Close'].pct_change().std() * 100
                max_price = data['Close'].max()
                min_price = data['Close'].min()
                
                summary_data.append({
                    'Symbol': symbol,
                    'Start Price': f"${start_price:.2f}",
                    'End Price': f"${end_price:.2f}",
                    'Total Return': f"{total_return:+.2f}%",
                    'Volatility': f"{volatility:.2f}%",
                    'High': f"${max_price:.2f}",
                    'Low': f"${min_price:.2f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # Individual stock charts
        st.markdown("### 📈 Individual Stock Performance")
        
        # Allow users to select which stocks to display
        display_stocks = st.multiselect(
            "Select stocks to display individual charts:",
            options=list(stock_data.keys()),
            default=list(stock_data.keys())[:2],  # Show first 2 by default
            key="individual_display"
        )
        
        # Create columns for layout
        if display_stocks:
            cols = st.columns(min(2, len(display_stocks)))
            
            for i, symbol in enumerate(display_stocks):
                data = stock_data[symbol]
                info = stock_info.get(symbol, {})
                
                with cols[i % 2]:
                    st.markdown(f"#### {symbol}")
                    
                    # Mini metrics
                    if len(data) > 0:
                        current_price = data['Close'].iloc[-1]
                        price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                        percent_change = (price_change / data['Close'].iloc[0]) * 100
                        
                        st.metric(
                            label="Period Performance",
                            value=f"${current_price:.2f}",
                            delta=f"{price_change:+.2f} ({percent_change:+.2f}%)"
                        )
                    
                    # Chart
                    fig = StockDataVisualizer.create_price_chart(data, symbol, chart_type, include_volume)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_chart_{symbol}")
                    
                    # Stock info expandable
                    if info and any(v != 'N/A' for v in info.values()):
                        with st.expander(f"📋 {symbol} Details"):
                            info_items = []
                            for key, value in info.items():
                                if value != 'N/A' and value is not None:
                                    display_key = key.replace('_', ' ').title()
                                    info_items.append(f"**{display_key}:** {value}")
                            
                            if info_items:
                                st.markdown('\n\n'.join(info_items))
        
        # Store data for export
        st.session_state.current_stock_data = stock_data
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error loading historical data: {str(e)}")
        logger.error(f"Error in render_historical_comparison: {e}")

def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    indicators = {}
    
    try:
        # Moving Averages
        indicators['MA20'] = data['Close'].rolling(window=20).mean()
        indicators['MA50'] = data['Close'].rolling(window=50).mean()
        indicators['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        ma20 = indicators['MA20']
        std20 = data['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = ma20 + (std20 * 2)
        indicators['BB_Lower'] = ma20 - (std20 * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = ema12 - ema26
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
        # Stochastic
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        indicators['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        indicators['%D'] = indicators['%K'].rolling(window=3).mean()
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
    
    return indicators

def render_technical_analysis(selected_stocks, chart_type):
    """Render technical analysis view with advanced indicators"""
    st.markdown("<h1 class='main-header'>🔬 Technical Analysis</h1>", unsafe_allow_html=True)
    
    if not selected_stocks:
        st.info("👈 Please select at least one stock in the sidebar to get started.")
        return
    
    # Stock selection for analysis
    selected_stock = st.selectbox(
        "Select stock for technical analysis:",
        selected_stocks,
        key="tech_analysis_stock"
    )
    
    # Technical indicators selection
    col1, col2 = st.columns(2)
    
    with col1:
        indicators = st.multiselect(
            "Select technical indicators:",
            ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Stochastic"],
            default=["Moving Averages", "RSI"],
            key="tech_indicators"
        )
    
    with col2:
        period = st.selectbox(
            "Select time period:",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
            format_func=lambda x: {
                "1mo": "1 Month",
                "3mo": "3 Months", 
                "6mo": "6 Months",
                "1y": "1 Year",
                "2y": "2 Years"
            }[x],
            key="tech_period"
        )
    
    if not indicators:
        st.warning("Please select at least one technical indicator.")
        return
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Fetch data
        status_text.text(f"🔄 Fetching data and calculating indicators for {selected_stock}...")
        progress_bar.progress(25)
        
        data = StockDataManager.get_historical_data(selected_stock, period)
        progress_bar.progress(50)
        
        # Calculate indicators
        tech_indicators = calculate_technical_indicators(data)
        progress_bar.progress(75)
        
        # Get stock info
        stock_info = StockDataManager.get_stock_info(selected_stock)
        progress_bar.progress(100)
        
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        if len(data) == 0:
            st.error("❌ No data available for technical analysis.")
            return
        
        # Determine number of subplots needed
        num_subplots = 1  # Price chart
        if "RSI" in indicators or "Stochastic" in indicators:
            num_subplots += 1
        if "MACD" in indicators:
            num_subplots += 1
        
        # Create subplots
        if num_subplots == 1:
            fig = go.Figure()
            price_row = 1
            rsi_row = None
            macd_row = None
        else:
            subplot_titles = [f"{selected_stock} Price"]
            row_heights = [0.6]
            
            if "RSI" in indicators or "Stochastic" in indicators:
                subplot_titles.append("RSI / Stochastic")
                row_heights.append(0.2)
            if "MACD" in indicators:
                subplot_titles.append("MACD")
                row_heights.append(0.2)
            
            fig = make_subplots(
                rows=num_subplots, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=subplot_titles,
                row_heights=row_heights
            )
            
            price_row = 1
            rsi_row = 2 if ("RSI" in indicators or "Stochastic" in indicators) else None
            macd_row = num_subplots if "MACD" in indicators else None
        
        # Add price chart (candlestick or line)
        if chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=price_row, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name="Price",
                    line=dict(color='#1E88E5', width=2)
                ),
                row=price_row, col=1
            )
        
        # Add technical indicators
        if "Moving Averages" in indicators:
            for ma_period, color in [(20, 'orange'), (50, 'red'), (200, 'purple')]:
                ma_key = f'MA{ma_period}'
                if ma_key in tech_indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=tech_indicators[ma_key],
                            name=f"MA{ma_period}",
                            line=dict(color=color, width=1)
                        ),
                        row=price_row, col=1
                    )
        
        if "Bollinger Bands" in indicators:
            if 'BB_Upper' in tech_indicators and 'BB_Lower' in tech_indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=tech_indicators['BB_Upper'],
                        name="BB Upper",
                        line=dict(color='rgba(173,216,230,0.8)', width=1)
                    ),
                    row=price_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=tech_indicators['BB_Lower'],
                        name="BB Lower",
                        line=dict(color='rgba(173,216,230,0.8)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(173,216,230,0.2)'
                    ),
                    row=price_row, col=1
                )
        
        # RSI
        if "RSI" in indicators and rsi_row and 'RSI' in tech_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=tech_indicators['RSI'],
                    name="RSI",
                    line=dict(color='purple', width=2)
                ),
                row=rsi_row, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=rsi_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=rsi_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=rsi_row, col=1)
        
        # Stochastic
        if "Stochastic" in indicators and rsi_row and '%K' in tech_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=tech_indicators['%K'],
                    name="%K",
                    line=dict(color='blue', width=1)
                ),
                row=rsi_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=tech_indicators['%D'],
                    name="%D",
                    line=dict(color='red', width=1)
                ),
                row=rsi_row, col=1
            )
        
        # MACD
        if "MACD" in indicators and macd_row and 'MACD' in tech_indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=tech_indicators['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=2)
                ),
                row=macd_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=tech_indicators['MACD_Signal'],
                    name="Signal",
                    line=dict(color='red', width=1)
                ),
                row=macd_row, col=1
            )
            
            # MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in tech_indicators['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=tech_indicators['MACD_Histogram'],
                    name="Histogram",
                    marker_color=colors,
                    opacity=0.6
                ),
                row=macd_row, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_stock} Technical Analysis - {period.upper()}",
            height=600 + (num_subplots - 1) * 200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            hovermode='x unified'
        )
        
        # Display chart with fullscreen option
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True, key="tech_analysis_chart")
        
        with col2:
            if st.button("🔍", key="fullscreen_tech", help="View technical analysis in fullscreen"):
                fig.update_layout(height=800 + (num_subplots - 1) * 200)
                st.plotly_chart(fig, use_container_width=True)
        
        # Technical analysis summary
        st.markdown("### 📊 Technical Analysis Summary")
        
        current_price = data['Close'].iloc[-1]
        analysis_summary = []
        
        # Price vs Moving Averages
        if "Moving Averages" in indicators:
            for ma_period in [20, 50, 200]:
                ma_key = f'MA{ma_period}'
                if ma_key in tech_indicators:
                    ma_value = tech_indicators[ma_key].iloc[-1]
                    if not pd.isna(ma_value):
                        trend = "Above" if current_price > ma_value else "Below"
                        analysis_summary.append(f"**MA{ma_period}:** {trend} (${ma_value:.2f})")
        
        # RSI Analysis
        if "RSI" in indicators and 'RSI' in tech_indicators:
            rsi_value = tech_indicators['RSI'].iloc[-1]
            if not pd.isna(rsi_value):
                if rsi_value > 70:
                    rsi_signal = "Overbought"
                elif rsi_value < 30:
                    rsi_signal = "Oversold"
                else:
                    rsi_signal = "Neutral"
                analysis_summary.append(f"**RSI:** {rsi_value:.1f} ({rsi_signal})")
        
        # MACD Analysis
        if "MACD" in indicators and 'MACD' in tech_indicators:
            macd_value = tech_indicators['MACD'].iloc[-1]
            signal_value = tech_indicators['MACD_Signal'].iloc[-1]
            if not pd.isna(macd_value) and not pd.isna(signal_value):
                macd_signal = "Bullish" if macd_value > signal_value else "Bearish"
                analysis_summary.append(f"**MACD:** {macd_signal} ({macd_value:.3f})")
        
        if analysis_summary:
            col1, col2 = st.columns(2)
            mid_point = len(analysis_summary) // 2
            
            with col1:
                st.markdown('\n\n'.join(analysis_summary[:mid_point]))
            
            with col2:
                st.markdown('\n\n'.join(analysis_summary[mid_point:]))
        
        # Store data for export
        st.session_state.current_stock_data = {selected_stock: data}
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error performing technical analysis: {str(e)}")
        logger.error(f"Error in render_technical_analysis: {e}")

def render_export_options():
    """Render enhanced data export options"""
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    current_data = st.session_state.get('current_stock_data', {})
    
    if not current_data:
        st.info("No data available to export. Please load some stock data first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export CSV", help="Download data as CSV file"):
            try:
                # Combine all stock data
                all_data_frames = []
                for symbol, df in current_data.items():
                    df_copy = df.copy()
                    df_copy['Symbol'] = symbol
                    df_copy = df_copy.reset_index()
                    all_data_frames.append(df_copy)
                
                if all_data_frames:
                    combined_df = pd.concat(all_data_frames, ignore_index=True)
                    csv_data = combined_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                    st.success("✅ CSV file ready for download!")
            except Exception as e:
                st.error(f"❌ Error creating CSV: {str(e)}")
    
    with col2:
        if st.button("📄 Export Excel", help="Download data as Excel file"):
            try:
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Write each stock to a separate sheet
                    for symbol, df in current_data.items():
                        df_copy = df.copy().reset_index()
                        sheet_name = symbol[:31]  # Excel sheet name limit
                        df_copy.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Create a summary sheet
                    summary_data = []
                    for symbol, df in current_data.items():
                        if len(df) > 0:
                            summary_data.append({
                                'Symbol': symbol,
                                'Records': len(df),
                                'Start Date': df.index[0].strftime('%Y-%m-%d'),
                                'End Date': df.index[-1].strftime('%Y-%m-%d'),
                                'Start Price': f"${df['Close'].iloc[0]:.2f}",
                                'End Price': f"${df['Close'].iloc[-1]:.2f}",
                                'Change': f"{((df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100):.2f}%"
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
                st.success("✅ Excel file ready for download!")
                
            except Exception as e:
                st.error(f"❌ Error creating Excel file: {str(e)}")
    
    with col3:
        if st.button("📋 Copy to Clipboard", help="Copy data as tab-separated values"):
            try:
                # Create a simple table format
                clipboard_data = []
                for symbol, df in current_data.items():
                    clipboard_data.append(f"\n{symbol} Data:")
                    clipboard_data.append(df.to_string())
                
                clipboard_text = '\n'.join(clipboard_data)
                
                # Use JavaScript to copy to clipboard
                st.code(clipboard_text, language=None)
                st.info("📋 Data displayed above - you can manually copy it to clipboard")
                
            except Exception as e:
                st.error(f"❌ Error preparing clipboard data: {str(e)}")

def main():
    """Enhanced main application function"""
    try:
        # Initialize session state with better defaults
        if "selected_stocks" not in st.session_state:
            st.session_state.selected_stocks = ["AAPL", "MSFT"]
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "Live Prices"
        if "chart_type" not in st.session_state:
            st.session_state.chart_type = "line"
        if "include_volume" not in st.session_state:
            st.session_state.include_volume = True
        if "alerts" not in st.session_state:
            st.session_state.alerts = {}
        if "history_period" not in st.session_state:
            st.session_state.history_period = "1mo"
        
        # Render sidebar
        render_sidebar()
        
        # Get current settings
        selected_stocks = st.session_state.selected_stocks
        view_mode = st.session_state.view_mode
        chart_type = st.session_state.chart_type
        include_volume = st.session_state.include_volume
        alerts = st.session_state.alerts
        history_period = st.session_state.history_period
        
        # Main content area
        if not selected_stocks:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h1>📈 Welcome to Stock Dashboard</h1>
                <p style='font-size: 1.2rem; color: #666;'>
                    Select stocks from the sidebar to get started with your analysis!
                </p>
                <p style='color: #888;'>
                    🏢 Choose stocks • 📊 Pick chart type • 👁️ Select view mode
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Route to appropriate view
        if view_mode == "Live Prices":
            render_live_prices(selected_stocks, alerts, chart_type, include_volume)
        elif view_mode == "Historical Comparison":
            render_historical_comparison(selected_stocks, history_period, chart_type, include_volume)
        elif view_mode == "Technical Analysis":
            render_technical_analysis(selected_stocks, chart_type)
        
        # Export options (always show if we have data)
        render_export_options()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Error in main: {e}")
        
        # Show error details in expander for debugging
        with st.expander("🐛 Error Details (for debugging)"):
            st.code(str(e))
            st.code(f"Session State: {dict(st.session_state)}")

if __name__ == "__main__":
    main()