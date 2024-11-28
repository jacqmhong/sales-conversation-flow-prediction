"""
This script trains LSTM models for predicting response types and conversation stages in sales conversations.
It combines embeddings with metadata features, builds and trains the models, and then evaluates their performance
on the validation set using accuracy, precision, recall, and F1-score. The trained models and their corresponding
metrics are saved for later use and comparison.
"""
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from utils import update_model_registry
import yaml

# Load model config
with open("../config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

train_df = pd.read_csv("../data/processed/train_data.csv")
val_df = pd.read_csv("../data/processed/val_data.csv")

# Combine embeddings with metadata features (speaker and turn)
def prepare_features(df):
    embeddings = np.array(df["embeddings"].apply(eval).tolist())
    speaker_role = df["speaker"].map({"Customer": 0, "Salesman": 1}).to_numpy()
    turn_number = df.groupby("conversation_id")["turn"].transform(lambda x: x / x.max()).to_numpy() # within each convo
    return np.hstack([embeddings, speaker_role.reshape(-1, 1), turn_number.reshape(-1, 1)])

def create_sequences(X, y, convo_ids, sequence_len):
    """
    Forms fixed-length sequences for the LSTM model, shifting the target by one turn in order to predict
    the label of the next turn. It includes the last valid sequence when predicting with new data (app.py).

    Input:
        X = [[A], [B], [C], [D], [E]] for a single convo
        y = [0, 0, 1, 1, 0]
        sequence_len = 3

    First sequence: Input [[A], [B], [C]], Target: 1 (label of D)
    Second sequence: Input [[B], [C], [D]], Target: 0 (label of E)

    Output:
        X_seq = [[[A], [B], [C]], [[B], [C], [D]]]
        y_seq = [1, 0]
    """
    X_seq, y_seq = [], []
    unique_convos = np.unique(convo_ids)
    for convo_id in unique_convos:
        convo_X = X[convo_ids == convo_id]
        convo_y = y[convo_ids == convo_id] if y is not None else None

        # Generate sequences
        for i in range(sequence_len, len(convo_X)):
            X_seq.append(convo_X[i-sequence_len:i])
            if convo_y is not None:
                y_seq.append(convo_y[i])

        # Include the last sequence for prediction mode (y=None)
        if y is None:
            X_seq.append(convo_X[-sequence_len:])

    return np.array(X_seq), (np.array(y_seq) if y is not None else None)

X_train = prepare_features(train_df)
X_val = prepare_features(val_df)

def train_and_save_lstm(target_name):
    # Model parameters based on target
    input_dim = X_train.shape[1] # embedding dim + metadata dim (2)
    lstm_config = config["lstm"][target_name]
    lstm_units = lstm_config["units"]
    dropout_rate = lstm_config["dropout"]
    batch_size = lstm_config["batch_size"]
    max_epochs = lstm_config["epochs"]
    learning_rate = lstm_config["learning_rate"]
    sequence_len = lstm_config["sequence_len"]

    with open(f"../models/label_encoders/label_encoder_{target_name}.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Encode labels
    y_train = label_encoder.transform(train_df[target_name])
    y_val = label_encoder.transform(val_df[target_name])

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, train_df["conversation_id"].to_numpy(), sequence_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, val_df["conversation_id"].to_numpy(), sequence_len)

    # One-hot encode target variables
    y_train_seq = to_categorical(y_train_seq, num_classes=len(label_encoder.classes_))
    y_val_seq = to_categorical(y_val_seq, num_classes=len(label_encoder.classes_))

    # LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(sequence_len, input_dim), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(label_encoder.classes_), activation="softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()]) # class imbalance - accuracy isn't enough

    with mlflow.start_run(run_name=f"Train_{target_name}"): # experiment tracking: log parameters and metrics
        mlflow.log_params({
            "target_name": target_name,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "sequence_len": sequence_len,
        })

        # Train and evaluate model
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True) # for epochs
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping])
        loss, accuracy, precision, recall = model.evaluate(X_val_seq, y_val_seq, verbose=0)
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
        mlflow.log_metrics(metrics)

        # Save model
        initial_model_path = f"../models/lstm_models/{target_name}_model_v1.h5"
        model.save(initial_model_path)
        update_model_registry(target_name=target_name, model_path=initial_model_path, metrics=metrics)
        mlflow.keras.log_model(model, artifact_path="model")


# Main execution block: Train and save response_type and conversation_stage models
if __name__ == "__main__":
    train_and_save_lstm("response_type")
    train_and_save_lstm("conversation_stage")
