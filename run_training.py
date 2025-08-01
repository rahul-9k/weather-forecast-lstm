# run_training.py

import os
import numpy as np
from src.model import build_model, train_val_split, train_model

# Paths
X_path = os.path.join("processed", "X.npy")
y_path = os.path.join("processed", "y.npy")
model_save_path = os.path.join("models", "best_model.h5")

# Load data
import numpy as np

X = np.load("data/processed/X.npy", allow_pickle=True)
y = np.load("data/processed/y.npy", allow_pickle=True)



# Split data
X_train, y_train, X_val, y_val = train_val_split(X, y)

# Build model
input_seq_len = X.shape[1]           # e.g., 30
output_seq_len = y.shape[1]          # e.g., 7
num_features = X.shape[2]            # e.g., 5
model = build_model(input_seq_len, output_seq_len, num_features)

model_save_path = "models/best_model.weights.h5"

print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

history = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=50,
    save_path=model_save_path  # Make sure this is not None
)


# After training completes, save weights explicitly
os.makedirs("models", exist_ok=True)
model.save_weights("models/best_model_weights.weights.h5")


print("✅ Model training complete. Best model weights saved at: models/best_model_weights.h5")
