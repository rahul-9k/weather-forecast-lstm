
import os
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.windowing import create_windows, train_test_split

# Load clean data
df = load_data("data/processed/delhi_weather_clean.csv")
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# Only keep numeric columns explicitly
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns used:", numeric_cols)

df_numeric = df[numeric_cols]

# Set window sizes
INPUT_WINDOW = 30
OUTPUT_WINDOW = 7

# Create input/output sequences
X, y = create_windows(df, INPUT_WINDOW, OUTPUT_WINDOW)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

# Confirm shapes
print("Train X:", X_train.shape)
print("Train y:", y_train.shape)
print("Test  X:", X_test.shape)
print("Test  y:", y_test.shape)

X = X.astype(np.float32)
y = y.astype(np.float32)

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

