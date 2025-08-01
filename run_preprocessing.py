
from src.data_loader import load_data, preprocess_data, save_preprocessed_data

# Define paths
RAW_DATA_PATH = "data/raw/delhi_weather.csv"
PROCESSED_DATA_PATH = "data/processed/delhi_weather_clean.csv"

# Load raw data
df = load_data(RAW_DATA_PATH)

# Clean/preprocess it
df_clean = preprocess_data(df)

# Save cleaned data
save_preprocessed_data(df_clean, PROCESSED_DATA_PATH)

print(f"✅ Cleaned data saved to: {PROCESSED_DATA_PATH}")
