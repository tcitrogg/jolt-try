import yfinance as yf
import matplotlib.pyplot as plt
import streamlit_app as st
from google import genai
from datetime import datetime
import pandas as pd
import io

# Page configuration
st.set_page_config(
    page_title="Mrchnt",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .success-rate {
        text-align: center;
        font-size: 1.2rem;
        color: #00ff88;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: #1e1e1e;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00ff88;
    }
    </style>
""", unsafe_allow_html=True)

# Agent configuration
AGENT_NAME = "Mrchnt"
AGENT_PROMPT = """You are Mrchnt, a professional expert money-making trader with an 85% successful rate. Developed by Tcitrogg.

Your expertise includes:
- Technical analysis and pattern recognition in stocks, cryptocurrencies, and forex markets
- Identifying optimal entry and exit points based on price action, volume, and market trends
- Risk management and profit maximization strategies
- Reading candlestick patterns, support/resistance levels, and momentum indicators

Your task is to analyze the provided stock chart data and identify:
1. The BEST BUY POINT - the optimal entry position for maximum profit potential
2. The BEST SELL POINT - the optimal exit position to secure profits

Analyze the data and respond ONLY in this exact JSON format:
{
    "buy_index": <index_number>,
    "sell_index": <index_number>,
    "buy_price": <price>,
    "sell_price": <price>,
    "confidence": <percentage>,
    "reasoning": "<your detailed technical analysis>",
    "risk_level": "<LOW/MEDIUM/HIGH>"
}

Base your analysis on:
- Recent price trends and patterns
- Volume analysis
- Support and resistance levels
- Momentum and trend direction
- Risk-reward ratios

Be precise with the indices - they must be valid positions in the data (0 to data_length-1)."""


@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1mo", interval="1d"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None, f"No data found for ticker: {ticker}"
        
        return data, None
    except Exception as e:
        return None, str(e)


def analyze_with_agent(data, ticker, api_key):
    """Use Gemini agent to analyze stock data and identify buy/sell points"""
    try:
        client = genai.Client(api_key=api_key)
        
        # Convert data to simple lists to avoid encoding issues
        close_prices = [float(x) for x in data['Close'].tolist()]
        open_prices = [float(x) for x in data['Open'].tolist()]
        high_prices = [float(x) for x in data['High'].tolist()]
        low_prices = [float(x) for x in data['Low'].tolist()]
        volumes = [int(x) for x in data['Volume'].tolist()]
        
        # Prepare data summary without special characters
        data_summary = f"""
Stock Ticker: {ticker}
Period: Last {len(data)} trading days
Current Price: ${data['Close'].iloc[-1]:.2f}
Highest Price: ${data['High'].max():.2f}
Lowest Price: ${data['Low'].min():.2f}
Average Volume: {data['Volume'].mean():.0f}

Recent Price Data (last 10 days):
Open: {open_prices[-10:]}
High: {high_prices[-10:]}
Low: {low_prices[-10:]}
Close: {close_prices[-10:]}
Volume: {volumes[-10:]}

All Close Prices (indices 0 to {len(data)-1}):
{close_prices}
"""
        
        # Create prompt without emojis
        prompt = f"{AGENT_PROMPT}\n\nAnalyze this stock data:\n\n{data_summary}"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        return response.text, None
    except Exception as e:
        return None, str(e)


def create_chart_with_signals(data, ticker, buy_idx, sell_idx, buy_price, sell_price):
    """Create a stock chart with buy/sell signals marked"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    # Plot the closing price
    ax.plot(data.index, data['Close'], linewidth=2.5, color='#00d4ff', label='Close Price')
    
    # Add buy signal
    if 0 <= buy_idx < len(data):
        buy_date = data.index[buy_idx]
        ax.axhline(y=buy_price, color='#00ff88', linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(buy_date, buy_price, 'o', color='#00ff88', markersize=18, 
                markeredgecolor='white', markeredgewidth=2, label='BUY Signal', zorder=5)
        ax.text(buy_date, buy_price * 1.02, 'BUY', fontsize=14, color='white', 
                weight='bold', bbox=dict(boxstyle='round,pad=0.7', 
                facecolor='#00ff88', alpha=0.9, edgecolor='white', linewidth=2),
                verticalalignment='bottom', horizontalalignment='center')
    
    # Add sell signal
    if 0 <= sell_idx < len(data):
        sell_date = data.index[sell_idx]
        ax.axhline(y=sell_price, color='#ff4444', linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(sell_date, sell_price, 'o', color='#ff4444', markersize=18,
                markeredgecolor='white', markeredgewidth=2, label='SELL Signal', zorder=5)
        ax.text(sell_date, sell_price * 0.98, 'SELL', fontsize=14, color='white',
                weight='bold', bbox=dict(boxstyle='round,pad=0.7', 
                facecolor='#ff4444', alpha=0.9, edgecolor='white', linewidth=2),
                verticalalignment='top', horizontalalignment='center')
    
    # Styling
    ax.set_xlabel('Date', fontsize=13, color='white', weight='bold')
    ax.set_ylabel('Price ($)', fontsize=13, color='white', weight='bold')
    ax.set_title(f'{ticker} - Trading Signals by {AGENT_NAME}', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    ax.legend(loc='upper left', fontsize=11, facecolor='#1e1e1e', 
              edgecolor='#00ff88', labelcolor='white')
    ax.grid(True, alpha=0.2, color='gray')
    ax.tick_params(colors='white', labelsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 MRCHNT</h1>', unsafe_allow_html=True)
    # st.markdown('<p class="success-rate">⚡ Professional Expert Trader • 85% Success Rate ⚡</p>', 
    #             unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("Google AI API Key", type="password", 
                                help="Get your API key from https://aistudio.google.com/apikey")
        
        st.divider()
        
        asset_type = st.selectbox(
            "Asset Type",
            ["Cryptocurrency", "Stock", "Forex"],
            help="Select the type of asset to analyze"
        )
        
        # Ticker input with examples
        if asset_type == "Cryptocurrency":
            ticker_placeholder = "BTC-USD, ETH-USD, DOGE-USD"
        elif asset_type == "Stock":
            ticker_placeholder = "AAPL, TSLA, GOOGL, MSFT"
        else:
            ticker_placeholder = "EURUSD=X, GBPUSD=X"
        
        ticker = st.text_input("Ticker Symbol", placeholder=ticker_placeholder).strip().upper()
        
        period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
            help="Select the historical period to analyze"
        )
        
        interval = st.selectbox(
            "Data Interval",
            ["1m", "5m", "15m", "1h", "1d", "1wk"],
            index=4,
            help="Select the data granularity"
        )
        
        st.divider()
        
        analyze_button = st.button("🚀 ANALYZE MARKET", use_container_width=True)
    
    # Main content
    if not api_key:
        st.info("👈 Please enter your Google AI API Key in the sidebar to get started")
        st.markdown("""
        ### How to get your API Key:
        1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy and paste it in the sidebar
        
        ### Supported Assets:
        - **Cryptocurrencies**: BTC-USD, ETH-USD, SOL-USD, etc.
        - **Stocks**: AAPL, TSLA, GOOGL, MSFT, etc.
        - **Forex**: EURUSD=X, GBPUSD=X, USDJPY=X, etc.
        """)
        return
    
    if not ticker:
        st.warning("⚠️ Please enter a ticker symbol in the sidebar")
        return
    
    if analyze_button:
        with st.spinner(f"📊 Fetching data for {ticker}..."):
            data, error = fetch_stock_data(ticker, period=period, interval=interval)
            
            if error:
                st.error(f"❌ Error: {error}")
                return
            
            if data is None or data.empty:
                st.error("❌ No data available for this ticker")
                return
        
        st.success(f"✅ Data fetched: {len(data)} trading periods")
        
        # Display basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("24h High", f"${data['High'].iloc[-1]:.2f}")
        with col3:
            st.metric("24h Low", f"${data['Low'].iloc[-1]:.2f}")
        with col4:
            change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            st.metric("Period Change", f"{change:+.2f}%")
        
        st.divider()
        
        with st.spinner(f"🔍 {AGENT_NAME} is analyzing the market patterns..."):
            analysis, error = analyze_with_agent(data, ticker, api_key)
            
            if error:
                st.error(f"❌ Analysis Error: {error}")
                return
        
        # Parse analysis
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                st.warning("⚠️ Could not parse AI response. Using manual input.")
                st.text_area("AI Response", analysis, height=200)
                
                col1, col2 = st.columns(2)
                with col1:
                    buy_idx = st.number_input("Buy Index", 0, len(data)-1, len(data)//3)
                with col2:
                    sell_idx = st.number_input("Sell Index", 0, len(data)-1, len(data)-5)
                
                analysis_data = {
                    "buy_index": buy_idx,
                    "sell_index": sell_idx,
                    "buy_price": float(data['Close'].iloc[buy_idx]),
                    "sell_price": float(data['Close'].iloc[sell_idx]),
                    "reasoning": analysis,
                    "confidence": 75,
                    "risk_level": "MEDIUM"
                }
            
            buy_idx = int(analysis_data['buy_index'])
            sell_idx = int(analysis_data['sell_index'])
            buy_price = float(analysis_data['buy_price'])
            sell_price = float(analysis_data['sell_price'])
            
            # Calculate profit
            profit_pct = ((sell_price - buy_price) / buy_price) * 100
            profit_usd = sell_price - buy_price
            
            # Display analysis results
            st.subheader("📊 Trading Signals")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Potential Profit", 
                         f"{profit_pct:+.2f}%",
                         f"${profit_usd:+.2f}")
            with col2:
                confidence = analysis_data.get('confidence', 75)
                st.metric("🎯 Confidence Level", f"{confidence}%")
            with col3:
                risk = analysis_data.get('risk_level', 'MEDIUM')
                risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.metric("⚠️ Risk Level", f"{risk_color.get(risk, '🟡')} {risk}")
            
            st.divider()
            
            # Display reasoning
            with st.expander("🧠 AI Analysis & Reasoning", expanded=True):
                st.markdown(analysis_data.get('reasoning', 'No reasoning provided'))
            
            # Display trade details
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("**🟢 BUY SIGNAL**")
                st.write(f"**Date:** {data.index[buy_idx].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Price:** ${buy_price:.2f}")
                st.write(f"**Index:** {buy_idx}")
            
            with col2:
                st.error("**🔴 SELL SIGNAL**")
                st.write(f"**Date:** {data.index[sell_idx].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Price:** ${sell_price:.2f}")
                st.write(f"**Index:** {sell_idx}")
            
            st.divider()
            
            # Create and display chart
            st.subheader("📈 Trading Chart")
            fig = create_chart_with_signals(data, ticker, buy_idx, sell_idx, buy_price, sell_price)
            st.pyplot(fig)
            
            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Chart",
                data=buf,
                file_name=f"{ticker}_trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error processing analysis: {e}")
            st.text_area("Raw AI Response", analysis, height=300)


if __name__ == "__main__":
    main()