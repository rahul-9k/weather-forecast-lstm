# src/windowing.py

import numpy as np
import pandas as pd

def create_windows(data: pd.DataFrame, input_window: int, output_window: int):
    """
    Returns:
    - X: np.array of shape (num_samples, input_window, num_features)
    - y: np.array of shape (num_samples, output_window, num_features)
    """
    data_values = data.values
    X, y = [], []

    total_length = len(data_values)

    for i in range(total_length - input_window - output_window + 1):
        X.append(data_values[i : i + input_window])
        y.append(data_values[i + input_window : i + input_window + output_window])

    return np.array(X), np.array(y)


def train_test_split(X, y, test_ratio=0.2):
    split_index = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test
