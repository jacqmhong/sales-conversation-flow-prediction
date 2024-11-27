from datetime import datetime
import logging
import numpy as np
import os
import pandas as pd
import pickle
from retrain import retrain_lstm_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from train import prepare_features, create_sequences
import yaml

# Set up logging
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "periodic_evaluation.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load thresholds and configs
thresholds_path = "../config/thresholds.yaml"
with open(thresholds_path, "r") as file:
    PERFORMANCE_THRESHOLDS = yaml.safe_load(file)

DRIFT_THRESHOLD = 0.1  # overall data drift
CONFIG_PATH = "../config/config.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

DRIFT_REPORT_PATH = "../logs/drift_report.csv"
MODEL_PATHS = {
    "response_type": "../models/lstm_models/lstm_response_type_model_with_metadata.h5",
    "conversation_stage": "../models/lstm_models/lstm_conversation_stage_model_with_metadata.h5"
}
LABEL_ENCODER_PATHS = {
    "response_type": "../models/label_encoders/label_encoder_response_type.pkl",
    "conversation_stage": "../models/label_encoders/label_encoder_conversation_stage.pkl"
}
REFERENCE_DATA_PATH = "../data/processed/train_data.csv"  # for data drift baseline
NEW_DATA_PATH = "../data/processed/test_data.csv"  # would instead use newly generated data to detect drift

# Performance Drift Monitoring - Evaluates and logs the model's performance on the new data to monitor performance.
def monitor_performance_drift(model, X_seq, y_seq):
    preds = np.argmax(model.predict(X_seq), axis=1)
    y_true = np.argmax(y_seq, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_true, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, preds, average="weighted", zero_division=0),
    }

    logging.info(f"Performance Metrics - {metrics}")
    performance_drift_detected = any(metrics[metric] < PERFORMANCE_THRESHOLDS[metric] for metric in PERFORMANCE_THRESHOLDS)
    if performance_drift_detected:
        for metric, value in metrics.items():
            if value < PERFORMANCE_THRESHOLDS[metric]:
                logging.warning(f"{metric.capitalize()} below threshold: {value:.4f}")
    return performance_drift_detected, metrics

# Data Drift Monitoring - Compares summary statistics of new data against reference data to detect data drift.
def monitor_data_drift(new_data, reference_data):
    drift_report = []
    total_drift = 0

    for column in reference_data.columns:
        if column in new_data.columns:
            if reference_data[column].dtype in ["float64", "int64"]:
                # Numerical data: compare means
                new_mean = new_data[column].mean()
                ref_mean = reference_data[column].mean()
                diff_mean = abs(new_mean - ref_mean)
                drift_report.append(f"{column} - Mean difference: {diff_mean:.4f}")
                total_drift += diff_mean
            elif reference_data[column].dtype == "object":
                # Categorical data: compare distributions
                ref_counts = reference_data[column].value_counts(normalize=True)
                new_counts = new_data[column].value_counts(normalize=True)
                diff_counts = (new_counts - ref_counts).abs().sum()
                drift_report.append(f"{column} - Categorical distribution difference: {diff_counts:.4f}")
                total_drift += diff_counts

    logging.info("Data Drift Report")
    for entry in drift_report:
        logging.info(entry)

    data_drift_detected = total_drift > DRIFT_THRESHOLD
    return data_drift_detected, drift_report

# Function to log performance and data drift info to a csv file. Always logs regardless of whether drift is detected.
def log_drift_report(target_name, performance_drift, data_drift, performance_metrics, data_drift_details):
    report_exists = os.path.isfile(DRIFT_REPORT_PATH)
    report_data = {
        "timestamp": datetime.now(),
        "target": target_name,
        "performance_drift_detected": performance_drift,
        "data_drift_detected": data_drift,
        "accuracy": performance_metrics.get("accuracy"),
        "precision": performance_metrics.get("precision"),
        "recall": performance_metrics.get("recall"),
        "f1_score": performance_metrics.get("f1_score"),
        "data_drift_details": "; ".join(data_drift_details) if data_drift_details else "No drift detected"
    }

    report_df = pd.DataFrame([report_data])
    if report_exists:
        report_df.to_csv(DRIFT_REPORT_PATH, mode="a", header=False, index=False) # append
    else:
        report_df.to_csv(DRIFT_REPORT_PATH, mode="w", header=True, index=False) # write

    logging.info(f"Drift report updated for {target_name}.")
    if performance_drift or data_drift:
        logging.warning(f"Drift detected for {target_name} and logged.")
    else:
        logging.info(f"No significant drift detected for {target_name}.")


if __name__ == "__main__":
    try:
        # Load and preprocess new data
        new_data = pd.read_csv(NEW_DATA_PATH)
        X_new = prepare_features(new_data)

        for target_name in ["response_type", "conversation_stage"]:
            # Load model and label encoder
            model = load_model(MODEL_PATHS[target_name])
            with open(LABEL_ENCODER_PATHS[target_name], "rb") as f:
                label_encoder = pickle.load(f)

            # Form sequences for LSTM
            sequence_len = config["lstm"][target_name]["sequence_len"]
            y_new = label_encoder.transform(new_data[target_name])
            X_new_seq, y_new_seq = create_sequences(X_new, y_new, new_data["conversation_id"].to_numpy(), sequence_len)
            y_new_seq = to_categorical(y_new_seq, num_classes=len(label_encoder.classes_))

            # Drift Monitoring
            reference_data = pd.read_csv(REFERENCE_DATA_PATH)
            data_drift_detected, data_drift_details = monitor_data_drift(new_data, reference_data)
            performance_drift_detected, performance_metrics = monitor_performance_drift(model, X_new_seq, y_new_seq)
            log_drift_report(target_name, performance_drift_detected, data_drift_detected, performance_metrics, data_drift_details)

            # Trigger retraining if drift detected
            if performance_drift_detected or data_drift_detected:
                logging.info(f"Drift detected for {target_name}. Triggering model retraining.")
                retrain_lstm_model(target_name, MODEL_PATHS[target_name], LABEL_ENCODER_PATHS[target_name])
            else:
                logging.info(f"No significant drift detected for {target_name}. No retraining required.")

    except Exception as e:
        logging.error(f"Error during periodic evaluation: {str(e)}")
