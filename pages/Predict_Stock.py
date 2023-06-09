import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import scipy.stats as stats
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Title
st.title("Stock Prediction")

# Load models and scalers
with open('scaler/tlkm_scaler.pkl', 'rb') as file:
    tlkm_scaler = pickle.load(file)
with open('scaler/xl_scaler.pkl', 'rb') as file:
    xl_scaler = pickle.load(file)
with open('scaler/tbig_scaler.pkl', 'rb') as file:
    tbig_scaler = pickle.load(file)
with open('scaler/towr_scaler.pkl', 'rb') as file:
    towr_scaler = pickle.load(file)

tlkm_model = tf.keras.models.load_model('model/TLKM_best_model.h5')
xl_model = tf.keras.models.load_model('model/EXCL_best_model.h5')
tbig_model = tf.keras.models.load_model('model/TBIG_best_model.h5')
towr_model = tf.keras.models.load_model('model/TOWR_best_model.h5')

# Stock symbols
stock_symbols = ["TLKM.JK", "EXCL.JK", "TBIG.JK", "TOWR.JK"]
selected_symbol = st.sidebar.selectbox('Select a stock symbol:', stock_symbols, index=stock_symbols.index(st.session_state["ticker"]))

# Select model and scaler based on the symbol
def select_model_and_scaler(symbol):
    if symbol == "TLKM.JK":
        return tlkm_model, tlkm_scaler
    elif symbol == "EXCL.JK":
        return xl_model, xl_scaler
    elif symbol == "TBIG.JK":
        return tbig_model, tbig_scaler
    elif symbol == "TOWR.JK":
        return towr_model, towr_scaler
    
model, scaler = select_model_and_scaler(selected_symbol)

# Fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.download(symbol, start=start_date, end=end_date)
    return stock

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=365)

stock_data = fetch_stock_data(selected_symbol, start_date, end_date)

# Scale data
def scale_data(data, scaler):
    data = data.filter(['Adj Close'])
    adj_close_data = data.values
    scaled_data = scaler.fit_transform(adj_close_data)
    return scaled_data

scaled_data = scale_data(stock_data, scaler)

# Predict future stock prices
def predict_future(model, scaler, scaled_data, future_days):
    predictions = []
    with st.spinner("Predicting future stock prices..."):
        for _ in range(future_days):
            input_data = scaled_data[-60:, :]
            input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))
            prediction = model.predict(input_data)
            inverse_prediction = scaler.inverse_transform(prediction)
            predictions.append(inverse_prediction[0][0])
            scaled_data = np.append(scaled_data, [[prediction][0][0]], axis=0)
    return predictions

predict_stock_price = predict_future(model, scaler, scaled_data, 30)

# Create dataframe for predicted stock prices
predict_stock_price_df = pd.DataFrame(predict_stock_price, columns=['Prediction'])
predict_stock_price_df.index = range(1, len(predict_stock_price_df) + 1)

st.header("Stock Data")
st.info("Showing the latest stock data", icon="ℹ️")

fig = go.Figure(data=go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], name="Adjusted Close"))
fig.update_layout(
    title=f"Stock Adjusted Close",
    xaxis_title="Date",
    yaxis_title="Adjusted Close",
    showlegend=True,
    height=400,
    template='plotly_dark'
)

tab_data_1, tab_data_2 = st.tabs(["📈 Chart", "📋 Data"])

with tab_data_1:
    st.plotly_chart(fig, use_container_width=True)
    
with tab_data_2:
    filtered_df = dataframe_explorer(stock_data, case=False)
    st.dataframe(filtered_df, use_container_width=True)


st.header("Stock Prediction")
st.info('Stock prediction result for the next 30 days using LSTM model')

# Plot predicted stock prices
fig_pred = go.Figure(data=go.Scatter(x=predict_stock_price_df.index, y=predict_stock_price_df['Prediction'], name="Prediction"))
fig_pred.update_layout(
    title=f"Stock Prediction",
    xaxis_title="Days",
    yaxis_title="Adjusted Close",
    showlegend=True,
    height=400,
    template='plotly_dark'
)


tab_predict_1, tab_predict_2 = st.tabs(["📈 Chart", "📋 Data"])

with tab_predict_1:
    st.plotly_chart(fig_pred, use_container_width=True)
    
with tab_predict_2:
    st.dataframe(predict_stock_price_df)


mean_returns = np.mean(predict_stock_price)
std_returns = np.std(predict_stock_price)
confidence_level = 0.95
var = abs(stats.norm.ppf(1 - confidence_level, mean_returns, std_returns))

# Display VaR
st.header("Value at Risk")
st.write("Value at risk is a value used to report maximum loss from holding an asset during a certain period at a certain level of probability")
st.info(f"Value at Risk (VaR) for the next 30 days at {confidence_level*100}% confidence level: {round(var, 4)}", icon="💼")