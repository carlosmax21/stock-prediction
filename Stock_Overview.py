import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator

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

# Calculate ROI
def calculate_roi(stock_data):
    initial_price = stock_data['Adj Close'].iloc[0]
    final_price = stock_data['Adj Close'].iloc[-1]
    roi = (final_price - initial_price) / initial_price
    return roi

# Calculate CAGR
def calculate_cagr(stock_data):
    trading_days = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='B')  # 'B' for business days
    n = len(trading_days)  # Number of trading days in the given date range
    initial_price = stock_data['Adj Close'].iloc[0]
    final_price = stock_data['Adj Close'].iloc[-1]
    cagr = (final_price / initial_price) ** (1 / (n / 252)) - 1
    return cagr

# Set title and sidebar
st.header("Stock Overview")
st.sidebar.markdown("Stock Range")

# Get user input
stock_symbols = ["TLKM.JK", "EXCL.JK", "TBIG.JK", "TOWR.JK"]
selected_symbol = st.sidebar.selectbox('Select a stock symbol:', stock_symbols, index=0)

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
st.write("Symbol: ", stock_info["symbol"])
st.write("Company: ", long_name)
st.write("Sector: ", stock_info["sector"])
st.write("Industry: ", stock_info["industry"])
st.write("Country: ", stock_info["country"])
st.write(stock_info["longBusinessSummary"])

if stock_data is not None:
    st.write("Stock Data:")
    st.dataframe(stock_data.tail())

    # Calculate and display performance metrics
    stock_data['Daily Returns'] = stock_data['Adj Close'].pct_change()
    stock_data['Cumulative Returns'] = (1 + stock_data['Daily Returns']).cumprod() - 1
    volatility = stock_data['Daily Returns'].std()
    roi = calculate_roi(stock_data)
    cagr = calculate_cagr(stock_data)

    st.subheader("Performance Metrics")
    st.write(f"ROI: {roi:.4f}")
    st.write(f"Volatility (Standard Deviation): {volatility:.4f}")
    st.write(f"CAGR: {cagr:.4f}")

    # Calculate and display RSI
    rsi_period = 14  # RSI calculation period
    rsi = RSIIndicator(stock_data['Adj Close'], rsi_period)
    stock_data['RSI'] = rsi.rsi()

    # Create subplots
    fig_peform = make_subplots(rows=1, cols=2, subplot_titles=("Cumulative Returns", "RSI"))

    # Add cumulative returns plot
    fig_peform.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Cumulative Returns'], name="Cumulative Returns"),
        row=1, col=1
    )

    # Add RSI plot
    fig_peform.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['RSI'], name="RSI"),
        row=1, col=2
    )

    # Update layout
    fig_peform.update_layout(
        title=f"{long_name} Performance",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=600,
        template='plotly_dark'
    )

    st.plotly_chart(fig_peform, use_container_width=True)

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
    fig_general.add_trace(
        go.Scatter(x=stock_data.index, y=sma_20, name="20-day SMA", line=dict(color='blue')),
        row=1, col=1
    )

    # Add 100-day SMA
    sma_100 = stock_data['Adj Close'].rolling(window=100).mean()
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
        title=f"{long_name} Stock Data",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=900,
        template='plotly_dark'
    )

    # Display the chart
    st.plotly_chart(fig_general, use_container_width=True)
else:
    st.write("No data available for the selected stock symbol and date range.")