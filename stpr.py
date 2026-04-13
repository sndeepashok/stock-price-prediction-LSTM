import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model= load_model(r'C:\Users\deepa\OneDrive\Desktop\API_My Skills\Stock\Stock Prediction Model.keras')

st.header('Stock Market Predictor')

stock= st.text_input('Enter Stock Symbol', 'GOOG')

start = '2015-01-01'
end = '2025-12-31'

data=yf.download(stock, start, end)

st.subheader('Stock Data')

st.write(data)

data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80) : len(data)])