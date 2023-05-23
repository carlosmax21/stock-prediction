import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Prediction")


# Fetch file
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

stock_symbols = ["TLKM.JK", "EXCL.JK", "TBIG.JK", "TOWR.JK"]
selected_symbol = st.sidebar.selectbox('Select a stock symbol:', stock_symbols)

def select_model_and_scaler(symbol):
    if symbol == "TLKM.JK":
        return tlkm_model, tlkm_scaler
    elif symbol == "EXCL.JK":
        return xl_model, xl_scaler
    elif symbol == "TBIG.JK":
        return tbig_model, tbig_scaler
    elif symbol == "TOWR.JK":
        return towr_model, towr_scaler
    
def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.download(symbol, start=start_date, end=end_date)
    return stock

def scale_data(data, scaler):
    data = data.filter(['Adj Close'])
    adj_close_data = data.values
    
    scaled_data = scaler.fit_transform(adj_close_data)
    
    return scaled_data

def predict_future(model, scaler, scaled_data, future_days):    
    predictions = []
    
    for _ in range(future_days):
        # Take the last 60 days' data for prediction
        input_data = scaled_data[-60:, :]
        
        # Reshape the input data for the LSTM model
        input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))
        
        # Make the prediction
        prediction = model.predict(input_data)
        inverse_prediction = scaler.inverse_transform(prediction)
        
        # Append the prediction to the list
        predictions.append(inverse_prediction[0][0])
        
        # Update the data array with the predicted value
        scaled_data = scaled_data[1:]
        scaled_data = np.append(scaled_data, [[prediction][0][0]], axis=0)
        
    return predictions
    
model, scaler = select_model_and_scaler(selected_symbol)

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=365)

stock_data = fetch_stock_data(selected_symbol, start_date, end_date)

fig = go.Figure(data=go.Scatter(x=stock_data.index, y=stock_data['Adj Close']))
fig.update_layout(
    title=f"Stock Adjusted Close",
    xaxis_title="Date",
    yaxis_title="Adjusted Close",
    showlegend=True,
    height=500,
    template='plotly_dark'
)
st.plotly_chart(fig, use_container_width=True)

scaled_data = scale_data(stock_data, scaler)

predict_stock_price = predict_future(model, scaler, scaled_data, 30)

predict_stock_price_df = pd.DataFrame(predict_stock_price, columns=['Prediction'])
predict_stock_price_df.index = range(1, len(predict_stock_price_df) + 1)

st.write(predict_stock_price_df)

fig_pred = go.Figure(data=go.Scatter(x=predict_stock_price_df.index, y=predict_stock_price_df['Prediction']))
fig_pred.update_layout(
    title=f"Stock Prediction",
    xaxis_title="Days",
    yaxis_title="Adjusted Close",
    showlegend=True,
    height=500,
    template='plotly_dark'
)
st.plotly_chart(fig_pred, use_container_width=True)