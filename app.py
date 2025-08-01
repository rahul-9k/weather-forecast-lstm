import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from tensorflow.keras.models import load_model
from src.data_loader import load_data, load_scaler
from components import render_forecast_plot

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delhi Weather Forecast", layout="wide")

# --- LOAD DATA AND MODEL ---
@st.cache_data(show_spinner=True)
def load_all():
    df = load_data("data/processed/delhi_weather_clean.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    model = load_model("models/best_model.h5")
    scaler = load_scaler("models/scaler.pkl")
    return df, model, scaler

df, model, scaler = load_all()

# --- CONSTANTS ---
ALL_FEATURES = ['tmin', 'tmax', 'tavg', 'prcp', 'wspd']
INPUT_WINDOW = 30
FORECAST_HORIZON = 7

# --- MAIN UI ---
st.title("🌤️ Delhi Weather Forecast — Next 7 Days")

# Today
today = pd.to_datetime(datetime.now().date())
input_start_date = today - timedelta(days=INPUT_WINDOW)
input_end_date = today - timedelta(days=1)

# Try using recent data first
try:
    input_seq = df.loc[input_start_date:input_end_date, ALL_FEATURES]
    if len(input_seq) < INPUT_WINDOW:
        raise ValueError("Insufficient recent data")
except:
    # Fallback: use same month/days from previous year
    st.warning(f"Using data from same period last year: {input_start_date.year - 1}")
    fallback_start = input_start_date.replace(year=input_start_date.year - 1)
    fallback_end = input_end_date.replace(year=input_end_date.year - 1)
    input_seq = df.loc[fallback_start:fallback_end, ALL_FEATURES]

    if len(input_seq) < INPUT_WINDOW:
        st.error(f"Even previous year's data from {fallback_start} to {fallback_end} is not enough.")
        st.stop()

input_seq_np = input_seq.values.reshape(1, INPUT_WINDOW, len(ALL_FEATURES))

# --- MODEL PREDICTION ---
scaled_preds = model.predict(input_seq_np.astype(np.float32))
dummy_full = np.zeros((FORECAST_HORIZON, len(ALL_FEATURES)))
dummy_full[:, :] = scaled_preds[0, :FORECAST_HORIZON, :]
preds = scaler.inverse_transform(dummy_full)

# Build forecast DataFrame
forecast_dates = pd.date_range(start=today, periods=FORECAST_HORIZON)
forecast_df = pd.DataFrame(preds, columns=ALL_FEATURES, index=forecast_dates)

# --- DISPLAY RESULTS ---
st.subheader(f"Forecast from {today.date()} for next {FORECAST_HORIZON} days")
with st.expander("📊 Show Forecast Data Table", expanded=True):
    st.dataframe(forecast_df.round(2))

# --- PLOT ---
render_forecast_plot(forecast_df)

# --- DOWNLOAD BUTTON ---
csv_bytes = forecast_df.to_csv().encode('utf-8')
st.download_button(
    label="📥 Download forecast CSV",
    data=csv_bytes,
    file_name=f"delhi_weather_forecast_{today.date()}.csv",
    mime='text/csv'
)
