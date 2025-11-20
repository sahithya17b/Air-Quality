import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ML libraries
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# =========================
# Load Data
# =========================
df = pd.read_csv(r"C:/Users/Hp/Downloads/air quality dashboard/Air_Quality.csv")
df.columns = df.columns.str.strip().str.replace('.', '', regex=False).str.replace(' ', '_')

st.title("🌍 Air Quality Monitoring Dashboard")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("🔧 Dashboard Controls")
station = st.sidebar.selectbox("Select Station", df['state'].dropna().unique())
pollutant = st.sidebar.selectbox("Select Pollutant", df['pollutant_id'].dropna().unique())
model_choice = st.sidebar.selectbox("Forecasting Model", ["ARIMA", "XGBoost", "LSTM"])

# =========================
# Filter Data
# =========================
filtered_df = df[(df['state'] == station) & (df['pollutant_id'] == pollutant)]
if filtered_df.empty:
    st.error("⚠️ No data available for selected station and pollutant.")
    st.stop()

for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

# =========================
# DASHBOARD 1: Data Explorer
# =========================
st.header("📊 Dashboard 1: Air Quality Data Explorer")

st.subheader("Data Completeness")
st.metric("Completeness", f"{filtered_df.notnull().mean().mean()*100:.2f}%")

st.subheader(f"{pollutant} Average Distribution")
st.plotly_chart(px.histogram(filtered_df, x='pollutant_avg', nbins=30))

st.subheader("Pollutant Min vs Max")
st.plotly_chart(px.scatter(filtered_df, x='pollutant_min', y='pollutant_max'))

st.subheader("📈 Correlation Between Pollutants")
pollutant_cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
corr_data = filtered_df[pollutant_cols].corr()
fig_corr, ax = plt.subplots()
sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# =========================
# DASHBOARD 2: Forecast Engine
# =========================
st.header("📈 Dashboard 2: Forecast Engine")

# Prepare time series
if 'last_update' in filtered_df.columns:
    filtered_df['last_update'] = pd.to_datetime(filtered_df['last_update'], errors='coerce')
    data = filtered_df[['last_update', 'pollutant_avg']].dropna().rename(columns={'last_update': 'ds', 'pollutant_avg': 'y'})
    data = data.sort_values('ds')

# If no valid date data, create dummy sequence
if data.empty or len(data) < 10:
    st.warning("Not enough timestamp data. Generating synthetic series for demo.")
    np.random.seed(42)
    data = pd.DataFrame({
        "ds": pd.date_range("2025-01-01", periods=30, freq="D"),
        "y": np.random.uniform(40, 120, 30)
    })

try:
    # Forecast for next 7 days
    forecast_days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    if model_choice == "ARIMA":
        model = ARIMA(data['y'], order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)

    elif model_choice == "XGBoost":
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['y'].values
        model = XGBRegressor()
        model.fit(X, y)
        future_X = np.arange(len(data), len(data)+7).reshape(-1, 1)
        forecast = model.predict(future_X)

    else:  # LSTM
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(data['y'].values.reshape(-1, 1))
        X_train, y_train = [], []
        for i in range(10, len(y_scaled)):
            X_train.append(y_scaled[i-10:i])
            y_train.append(y_scaled[i])
        X_train, y_train = np.array(X_train), np.array(y_train)

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(10, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, verbose=0)

        X_input = y_scaled[-10:].reshape(1, 10, 1)
        forecast_scaled = []
        for _ in range(7):
            pred = model.predict(X_input, verbose=0)
            forecast_scaled.append(pred[0][0])
            X_input = np.append(X_input[:, 1:, :], [[pred]], axis=1)
        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    forecast_df = pd.DataFrame({"Day": forecast_days, "Predicted Value": forecast})

    st.subheader("🗓️ Weekly Forecast (Sun–Sat)")
    st.write(forecast_df)
    st.plotly_chart(px.line(forecast_df, x="Day", y="Predicted Value", markers=True, title=f"{pollutant} Forecast using {model_choice}"))

except Exception as e:
    st.error(f"Forecasting failed: {e}")

# =========================
# DASHBOARD 3: Alerts
# =========================
st.header("🚨 Dashboard 3: Alert System")
avg_value = filtered_df['pollutant_avg'].mean(skipna=True)
status = "Good" if avg_value < 50 else "Moderate" if avg_value < 100 else "Unhealthy"
st.metric("Average Level", f"{avg_value:.2f} ({status})")

# =========================
# DASHBOARD 4: Summary Gauge
# =========================
st.header("📟 Dashboard 4: Summary Gauge")
fig4 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_value,
    title={'text': "Average Pollutant Level"},
    gauge={'axis': {'range': [0, 150]},
           'steps': [
               {'range': [0, 50], 'color': "green"},
               {'range': [51, 100], 'color': "orange"},
               {'range': [101, 150], 'color': "red"}]})
)
st.plotly_chart(fig4)
