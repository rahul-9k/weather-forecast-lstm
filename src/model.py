# src/model.py

import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

def train_val_split(X, y, val_ratio=0.2):
    split_index = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    return X_train, y_train, X_val, y_val


def build_model(input_seq_len, output_seq_len, num_features, latent_dim=128):
    encoder_inputs = Input(shape=(input_seq_len, num_features))
    encoder_lstm = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = RepeatVector(output_seq_len)(state_h)
    decoder_lstm = LSTM(latent_dim, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    output_layer = TimeDistributed(Dense(num_features))
    decoder_outputs = output_layer(decoder_outputs)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(
    loss=MeanSquaredError(),
    optimizer='adam',
    metrics=[MeanAbsoluteError()]
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, save_path="models/best_model_weights.h5"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss", mode="min", save_weights_only=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    return history

