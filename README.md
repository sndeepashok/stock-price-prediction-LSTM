# stock-price-prediction-LSTM
LSTM-based stock price prediction model using TensorFlow with real-time data from yfinance and time-series forecasting.
# 📊 Stock Price Prediction using LSTM (Deep Learning)

## 🔹 Overview
This project implements a deep learning-based stock price prediction system using **LSTM (Long Short-Term Memory)** networks. It fetches historical stock data using the yfinance API, performs preprocessing, and predicts future stock prices based on time-series patterns.

---

## 🔹 Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib  
- TensorFlow / Keras  
- Scikit-learn  
- yfinance  

---

## 🔹 Project Workflow

### 1️⃣ Data Collection
- Retrieved historical stock data (GOOG) using **yfinance API**
- Time range: 2015 – 2025

### 2️⃣ Data Preprocessing
- Removed missing values  
- Applied **MinMaxScaler** for normalization  
- Split data into:
  - 80% Training  
  - 20% Testing  

### 3️⃣ Feature Engineering
- Created time-series sequences using **100-day window**
- Generated input-output pairs for LSTM training  

### 4️⃣ Model Building
- Built a deep LSTM model using TensorFlow/Keras:
  - 4 LSTM layers (50, 60, 80, 120 units)
  - Dropout layers (0.2–0.5) to reduce overfitting  
  - Dense output layer  
- Activation function: ReLU  
- Optimizer: Adam  
- Loss Function: Mean Squared Error  

### 5️⃣ Model Training
- Trained model for **50 epochs**
- Batch size: 32  

### 6️⃣ Prediction & Evaluation
- Predicted stock prices on test dataset  
- Compared:
  - Predicted vs Actual prices  
- Visualized using Matplotlib  

---

## 🔹 Visualizations
- 📈 Moving Average (100-day vs Closing Price)  
- 📉 Moving Average (100-day & 200-day comparison)  
- 📊 Predicted vs Actual Stock Prices  

---

## 🔹 Model Saving & Loading
- Model saved using Keras format:
