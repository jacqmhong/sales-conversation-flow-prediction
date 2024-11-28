# retrain.py: Script for retraining the LSTM models based on detected drift

import logging
import mlflow
import os
import pandas as pd
import pickle
import yaml
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import prepare_features, create_sequences
from utils import save_versioned_model

# Set up logging
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "retrain.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configs
config_path = "../config/config.yaml"
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

thresholds_path = "../config/thresholds.yaml"
with open(thresholds_path, 'r') as threshold_file:
    PERFORMANCE_THRESHOLDS = yaml.safe_load(threshold_file)

TRAIN_DATA_PATH = "../data/processed/train_data.csv" # typically new or appended data
VAL_DATA_PATH = "../data/processed/val_data.csv"
LABEL_ENCODER_PATHS = {
    "response_type": "../models/label_encoders/label_encoder_response_type.pkl",
    "conversation_stage": "../models/label_encoders/label_encoder_conversation_stage.pkl"
}

def evaluate_metrics(model, X_seq, y_seq):
    preds = model.predict(X_seq)
    y_pred = preds.argmax(axis=1)
    y_true = y_seq.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    performance_drift_detected = any(metrics[metric] < PERFORMANCE_THRESHOLDS[metric] for metric in PERFORMANCE_THRESHOLDS)
    return performance_drift_detected, metrics

def retrain_lstm_model(target_name, label_encoder_path):
    # Prepare features and sequences for new data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    val_data = pd.read_csv(VAL_DATA_PATH)
    X_train = prepare_features(train_data)
    X_val = prepare_features(val_data)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    y_train = label_encoder.transform(train_data[target_name])
    y_val = label_encoder.transform(val_data[target_name])

    # Prepare sequences for the LSTM model
    sequence_len = config["lstm"][target_name]["sequence_len"]
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, train_data["conversation_id"].to_numpy(), sequence_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, val_data["conversation_id"].to_numpy(), sequence_len)
    y_train_seq = to_categorical(y_train_seq, num_classes=len(label_encoder.classes_))
    y_val_seq = to_categorical(y_val_seq, num_classes=len(label_encoder.classes_))

    # Model parameters
    lstm_units = config["lstm"][target_name]["units"]
    dropout_rate = config["lstm"][target_name]["dropout"]
    batch_size = config["lstm"][target_name]["batch_size"]
    max_epochs = config["lstm"][target_name]["epochs"]
    learning_rate = config["lstm"][target_name]["learning_rate"]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(sequence_len, X_train.shape[1]), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(label_encoder.classes_), activation="softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])

    with mlflow.start_run(run_name=f"Retrain_{target_name}"): # experiment tracking: log parameters and metrics
        mlflow.log_params({
            "target_name": target_name,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "sequence_len": sequence_len,
        })

        # Train the model
        logging.info(f"Starting model training for {target_name}...")
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
                  epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping])
        logging.info("Model training completed.")

        # Evaluate the model
        performance_drift_detected, metrics = evaluate_metrics(model, X_val_seq, y_val_seq)
        mlflow.log_metrics(metrics)
        logging.info(f"Retrained Model Metrics for {target_name} - {metrics}")

        # Save the model if metrics meet thresholds
        if performance_drift_detected:
            logging.warning(f"Retraining complete, but model metrics for {target_name} did not meet thresholds. No update performed.")
        else:
            save_versioned_model(model, target_name, metrics, logging)
            mlflow.keras.log_model(model, artifact_path="model")
            logging.info(f"Versioned model for {target_name} saved successfully.")


# Execute retraining
if __name__ == "__main__":
    try:
        # Retrain and save the response_type model and the conversation_stage model
        retrain_lstm_model(target_name="response_type", label_encoder_path=LABEL_ENCODER_PATHS["response_type"])
        retrain_lstm_model(target_name="conversation_stage", label_encoder_path=LABEL_ENCODER_PATHS["conversation_stage"])
    except Exception as e:
        logging.error(f"Error during retraining process: {str(e)}")
