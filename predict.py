import numpy as np
import pandas as pd
import datetime
from src.data_loader import load_data, load_scaler
from src.model import build_model  # import your build_model function

def get_previous_year_same_month_input(df, input_window, numeric_columns):
    today = datetime.date.today()
    current_year = today.year
    current_month = today.month
    prev_year = current_year - 1

    mask = (df.index.year == prev_year) & (df.index.month == current_month)
    month_data = df.loc[mask, numeric_columns]

    if len(month_data) < input_window:
        print(f"⚠️ Not enough data for {prev_year}-{current_month}, using last {input_window} days instead.")
        input_data = df[numeric_columns].iloc[-input_window:].values
    else:
        input_data = month_data.iloc[-input_window:].values
    
    return input_data.reshape(1, input_window, len(numeric_columns))


def predict_next_days(model, input_seq, output_window):
    cleaned_input = np.array([
        [
            [float(x) if isinstance(x, (int, float, np.integer, np.floating)) else np.nan for x in timestep]
            for timestep in sample
        ]
        for sample in input_seq
    ], dtype=np.float32)

    preds = model.predict(cleaned_input)
    return preds[0]


def inverse_scale_predictions(predictions, scaler):
    return scaler.inverse_transform(predictions)


def main():
    print("📈 Predicting next 7 days...")

    # Paths
    model_path = "models/best_model_weights.weights.h5"
    scaler_path = "models/scaler.pkl"
    data_path = "data/processed/delhi_weather_clean.csv"

    # Constants
    INPUT_WINDOW = 30
    OUTPUT_WINDOW = 7

    # Load preprocessed data and scaler
    df = load_data(data_path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    scaler = load_scaler(scaler_path)

    numeric_columns = ['tmin', 'tmax', 'tavg', 'prcp', 'wspd']

    # Build model and load weights
    model = build_model(INPUT_WINDOW, OUTPUT_WINDOW, len(numeric_columns))
    model.load_weights(model_path)

    # Get input sequence
    input_seq = get_previous_year_same_month_input(df, INPUT_WINDOW, numeric_columns)

    # Predict
    scaled_preds = predict_next_days(model, input_seq, OUTPUT_WINDOW)

    # Inverse scale
    preds = inverse_scale_predictions(scaled_preds, scaler)

    # Prepare output
    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=OUTPUT_WINDOW)
    forecast_df = pd.DataFrame(preds, columns=numeric_columns, index=future_dates)

    print("✅ Prediction complete.\n")
    print(forecast_df.round(2))


if __name__ == "__main__":
    main()
