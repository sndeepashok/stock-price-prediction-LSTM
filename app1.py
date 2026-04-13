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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price Vs MA50')
moving_average_50_days= data.Close.rolling(50).mean()
fig1 =plt.figure(figsize=(8,6))
plt.plot(moving_average_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Price Vs MA50 Vs MA100')
moving_average_100_days= data.Close.rolling(100).mean()
fig2 =plt.figure(figsize=(8,6))
plt.plot(moving_average_50_days, 'r', label='MA50')
plt.plot(moving_average_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Price Vs MA100 Vs MA200')
moving_average_200_days= data.Close.rolling(200).mean()
fig3 =plt.figure(figsize=(8,6))
plt.plot(moving_average_100_days, 'r', label='MA100')
plt.plot(moving_average_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100 : i])
    y.append(data_test_scale[i,0])
             
x,y=np.array(x), np.array(y)

predict=model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale

y = y * scale

st.subheader('Original Price Vs Predicted Price')
fig4= plt.figure(figsize=(8,6))
plt.plot(predict,'r', label='Original Price')
plt.plot(y,'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
