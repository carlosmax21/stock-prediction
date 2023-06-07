import streamlit as st
import yfinance as yf
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Set Streamlit layout
st.set_page_config(layout="wide")

# Streamlit Function
# function to fetch data
def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.download(symbol, start=start_date, end=end_date)
    return stock

def get_stock_info(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    return info

# Set title and sidebar
st.title("Stock Overview")
st.sidebar.markdown("Stock Range")

# Get user input
stock_symbols = ["TLKM.JK", "EXCL.JK", "TBIG.JK", "TOWR.JK"]
selected_symbol = st.sidebar.selectbox('Select a stock symbol:', stock_symbols)
st.session_state["ticker"] = selected_symbol
if "ticker" in st.session_state:
    selected_symbol = st.session_state['ticker']

default_end_date = dt.date.today()
default_start_date = default_end_date - dt.timedelta(days=365)

col1, col2 = st.sidebar.columns(2)
# Get user input for date range
with col1:
    start_date = st.date_input("Start Date:", value=default_start_date)

with col2:
    end_date = st.date_input("End Date:", value=default_end_date)

# Fetch stock data
stock_data = fetch_stock_data(selected_symbol, start_date, end_date)
stock_info = get_stock_info(selected_symbol)
long_name = stock_info['longName'] if stock_info else selected_symbol

# Display stock information
st.header("Stock Detail")
# Company name and symbol
st.markdown(f"**{long_name}** ({stock_info['symbol']})")

# Country, sector, and industry
st.write(f"**Country:** {stock_info['country']}")
st.write(f"**Sector:** {stock_info['sector']}")
st.write(f"**Industry:** {stock_info['industry']}")

# Long business summary
with st.expander("Business Summary"):
    st.write(stock_info["longBusinessSummary"])

if not start_date <= end_date:
    st.error("Please enter a correct date range, End date must be after start date.")
elif not start_date <= default_end_date:
    st.error("Please enter a correct date range, Start date can't be earlier than today.")
elif selected_symbol not in stock_symbols:
    st.error("Invalid stock symbol.")
else:
    st.header("Stock Data")
    st.info("Showing the latest data from stock", icon="ℹ️")

    # Display stock data using dataframe_explorer
    filtered_df = dataframe_explorer(stock_data, case=False)
    st.dataframe(filtered_df, use_container_width=True)

    # Technical Indicator section
    st.subheader("Technical Indicators")
    st.write("Technical indicators help analyze stock price trends and identify potential opportunities. Here's a brief explanation of some indicators used:")

    # Explanation of RSI
    rsi_text = """
    **RSI (Relative Strength Index)**  
    RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100, with values above 70 indicating overbought conditions and values below 30 indicating oversold conditions.
    """

    # Explanation of SMA
    sma_text = """
    **SMA (Simple Moving Average)**  
    SMA calculates the average price of a security over a specified time period. It helps identify trends by smoothing out price fluctuations and can be used to determine support and resistance levels.
    """

    # Explanation of MACD
    macd_text = """
    **MACD (Moving Average Convergence Divergence)**  
    MACD is a trend-following momentum indicator that consists of two lines: the MACD line and the signal line. MACD crossovers and divergences can indicate potential buy or sell signals.
    """

    # Display in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander("RSI"):
            st.markdown(rsi_text)

    with col2:
        with st.expander("SMA"):
            st.markdown(sma_text)

    with col3:
        with st.expander("MACD"):
            st.markdown(macd_text)
    
    fig_general = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # Add candlestick chart
    fig_general.add_trace(
        go.Candlestick(x=stock_data.index,
                       open=stock_data['Open'],
                       high=stock_data['High'],
                       low=stock_data['Low'],
                       close=stock_data['Close'],
                       name="Candlestick"),
        row=1, col=1
    )

    # Update candlestick chart properties
    fig_general.update_traces(decreasing_line_color='red', increasing_line_color='green', row=1, col=1)

    # Add 20-day SMA
    sma_20 = stock_data['Adj Close'].rolling(window=20).mean()
    if(len(stock_data) > 20):
        fig_general.add_trace(
            go.Scatter(x=stock_data.index, y=sma_20, name="20-day SMA", line=dict(color='blue')),
            row=1, col=1
        )

    # Add 100-day SMA
    sma_100 = stock_data['Adj Close'].rolling(window=100).mean()
    if(len(stock_data) > 100):
        fig_general.add_trace(
            go.Scatter(x=stock_data.index, y=sma_100, name="100-day SMA", line=dict(color='yellow')),
            row=1, col=1
        )

    # Add line chart for adjusted close price
    fig_general.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], name="Adjusted Close"),
        row=1, col=1
    )

    fig_general.update_yaxes(title_text="Price", row=1, col=1)

    # Add volume bar chart
    fig_general.add_trace(
        go.Bar(x=stock_data.index, y=stock_data['Volume'], name="Volume"),
        row=2, col=1
    )

    # Update volume chart properties
    fig_general.update_yaxes(title_text="Volume", row=2, col=1)

    # Update layout
    fig_general.update_layout(
        title=f"{long_name} Stock Data and Simple Moving Average",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=800,
        template='plotly_dark'
    )

    # Create a separate figure for RSI
    fig_rsi = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Calculate RSI
    rsi_period = 14  # Number of periods for RSI calculation
    rsi = RSIIndicator(close=stock_data['Adj Close'], window=rsi_period)
    rsi_values = rsi.rsi()

    # Add RSI line chart
    fig_rsi.add_trace(
        go.Scatter(x=stock_data.index, y=rsi_values, name="RSI", line=dict(color='orange')),
        row=1, col=1
    )

    # Update RSI chart properties
    fig_rsi.update_yaxes(title_text="RSI", row=1, col=1)
    
    fig_rsi.update_layout(
        title=f"{long_name} Relative Strength Index",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=500,
        template='plotly_dark'
    )
    
    # Create a separate figure for MACD
    fig_macd = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Calculate MACD
    macd = MACD(close=stock_data['Adj Close'])
    macd_values = macd.macd()
    signal_line = macd.macd_signal()
    histogram = macd.macd_diff()

    # Add MACD line chart
    fig_macd.add_trace(
        go.Scatter(x=stock_data.index, y=macd_values, name="MACD", line=dict(color='purple')),
        row=1, col=1
    )
    # Add signal line
    fig_macd.add_trace(
        go.Scatter(x=stock_data.index, y=signal_line, name="Signal Line", line=dict(color='yellow')),
        row=1, col=1
    )
    # Add histogram
    fig_macd.add_trace(
        go.Bar(x=stock_data.index, y=histogram, name="Histogram", marker_color='rgba(0, 255, 0, 0.5)'),
        row=1, col=1
    )

    # Update MACD chart properties
    fig_macd.update_yaxes(title_text="MACD", row=1, col=1)
    
    fig_macd.update_layout(
        title=f"{long_name} Moving Average Convergence Divergence",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=500,
        template='plotly_dark'
    )

    # Display the charts
    if(len(stock_data) > 1):
        st.plotly_chart(fig_general, use_container_width=True)
        
        if not np.isnan(rsi_values).all():
            st.plotly_chart(fig_rsi, use_container_width=True)
        else :
            st.info("Not enough data to plot RSI", icon="ℹ️")
            
        if not np.isnan(macd_values).all():
            st.plotly_chart(fig_macd, use_container_width=True)
        else :
            st.info("Not enough data to plot MACD", icon="ℹ️")
   
    else :
        st.info("Not enough data to plot data movement", icon="ℹ️")