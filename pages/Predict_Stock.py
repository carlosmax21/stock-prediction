import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
import pickle
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

# Fetch file
with open('scaler/tlkm_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

tlkm_model = tf.keras.models.load_model('model/TLKM_best_model.h5')
st.markdown(tlkm_model.summary())

st.title("Stock Prediction")
st.markdown(scaler.scale_)



