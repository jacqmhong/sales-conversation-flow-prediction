# retrain.py: Script for retraining the LSTM models based on detected drift

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

# Load configs
config_path = "../config/config.yaml"
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

thresholds_path = "../config/thresholds.yaml"
with open(thresholds_path, 'r') as threshold_file:
    PERFORMANCE_THRESHOLDS = yaml.safe_load(threshold_file)

TRAIN_DATA_PATH = "../data/processed/train_data.csv" # new data
VAL_DATA_PATH = "../data/processed/val_data.csv"

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

# Retrain the LSTM model for a specific target and save the updated model if metrics meet thresholds.
def retrain_lstm_model(target_name, model_save_path, label_encoder_path):
    # Prepare features and sequences for new data
    train_data = pd.read_csv(TRAIN_DATA_PATH)  # typically new or appended data
    val_data = pd.read_csv(VAL_DATA_PATH)
    X_train = prepare_features(train_data)
    X_val = prepare_features(val_data)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    y_train = label_encoder.transform(train_data[target_name])
    y_val = label_encoder.transform(val_data[target_name])
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

    # LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(sequence_len, X_train.shape[1]), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(label_encoder.classes_), activation="softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])

    # Train
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    print(f"Retraining LSTM model for {target_name}...")
    history = model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
                        epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping])

    # Evaluate
    performance_drift_detected, metrics = evaluate_metrics(model, X_val_seq, y_val_seq)
    print(f"Retrained Model Metrics for {target_name} - {metrics}")
    if performance_drift_detected:
        # To avoid using a subpar model in prod
        print("Retraining complete, but model metrics did not meet thresholds. No update performed.")
    else:
        model.save(model_save_path)
        print(f"Retrained model saved to {model_save_path}.")

# Execute retraining
if __name__ == "__main__":
    # Retrain and save response_type model
    retrain_lstm_model(
        target_name="response_type",
        model_save_path="../models/lstm_models/lstm_response_type_model_with_metadata.h5",
        label_encoder_path="../models/label_encoders/label_encoder_response_type.pkl"
    )

    # Retrain and save conversation_stage model
    retrain_lstm_model(
        target_name="conversation_stage",
        model_save_path="../models/lstm_models/lstm_conversation_stage_model_with_metadata.h5",
        label_encoder_path="../models/label_encoders/label_encoder_conversation_stage.pkl"
    )
