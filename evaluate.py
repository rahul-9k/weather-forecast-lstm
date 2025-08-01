import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.model import build_model
from src.data_loader import load_data, load_scaler
from src.windowing import create_windows
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, scaler, numeric_columns, output_window):
    y_pred = model.predict(X_test)

    # Inverse scale predictions and ground truth
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, len(numeric_columns))).reshape(-1, output_window, len(numeric_columns))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, len(numeric_columns))).reshape(-1, output_window, len(numeric_columns))

    # Overall MAE & RMSE
    overall_mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
    overall_rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))

    print("✅ Evaluation complete.")
    print(f"📏 Overall MAE: {overall_mae:.4f}")
    print(f"📏 Overall RMSE: {overall_rmse:.4f}")

    return y_test_inv, y_pred_inv


def evaluate_per_feature(y_true, y_pred, feature_names):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = y_true.reshape(-1, len(feature_names))
    y_pred = y_pred.reshape(-1, len(feature_names))
    
    print("\n📊 Per-feature MAE and RMSE:")
    for i, feature in enumerate(feature_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        print(f"{feature:>6} → MAE: {mae:.4f} | RMSE: {rmse:.4f}")


def main():
    print("🔍 Evaluating model...")

    # Paths
    model_path = "models/best_model.h5"
    scaler_path = "models/scaler.pkl"
    data_path = "data/processed/delhi_weather_clean.csv"

    # Constants
    INPUT_WINDOW = 30
    OUTPUT_WINDOW = 7
    numeric_columns = ['tmin', 'tmax', 'tavg', 'prcp', 'wspd']

    # Load model, data, scaler
    model = build_model(INPUT_WINDOW, OUTPUT_WINDOW, len(numeric_columns))
    model.load_weights("models/best_model_weights.weights.h5")

    df = load_data(data_path)
    scaler = load_scaler(scaler_path)

    # Prepare windows
    X, y = create_windows(df[numeric_columns], input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)


    # Evaluate
    y_test_inv, y_pred_inv = evaluate_model(model, X, y, scaler, numeric_columns, OUTPUT_WINDOW)

    # Per-feature evaluation
    evaluate_per_feature(y_test_inv, y_pred_inv, numeric_columns)


if __name__ == "__main__":
    main()


