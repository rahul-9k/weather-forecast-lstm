import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads weather data from a CSV file.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and scales the weather data.
    Returns the scaled dataframe and saves the scaler.
    """
    # Keep only relevant columns
    cols_to_keep = ["date", "tmin", "tmax", "tavg", "prcp", "wspd"]
    df = df[cols_to_keep]

    # Drop rows with too many missing values
    df = df.dropna(thresh=4)  # Allow up to 2 NaNs

    # Forward fill and backward fill remaining missing values
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Scale features (excluding 'date')
    features = df.drop(columns=["date"])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Save scaler to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # Create new DataFrame with scaled values and date
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["date"] = df["date"].values
    scaled_df = scaled_df[["date"] + list(features.columns)]  # ensure 'date' first

    return scaled_df

def save_preprocessed_data(df: pd.DataFrame, output_path: str):
    """
    Saves the preprocessed dataframe to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def load_scaler(path="models/scaler.pkl"):
    """
    Load the saved MinMaxScaler object from disk.
    """
    return joblib.load(path)
