"""
This script evaluates the performance of both the Markov and LSTM models for predicting response types
and conversation stages in sales conversations. It calculates the accuracy, precision, recall, and F1-score
of each model. The evaluation uses test data, pre-trained models, and corresponding label encoders.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from train import create_sequences, prepare_features
import yaml

# Helper for evaluate_markov_model to predict the next response using the markov transition matrix
def get_markov_prediction(current_state, transition_matrix):
    if current_state not in transition_matrix:
        return None  # no data for this state
    return max(transition_matrix[current_state], key=transition_matrix[current_state].get)

# Evaluate the Markov model using accuracy, precision, recall, and F1-score.
def evaluate_markov_model(sequences, transition_matrix, target_name):
    y_true, y_pred = [], []
    for sequence in sequences:
        for i in range(len(sequence) - 2):
            current_state = (sequence[i], sequence[i + 1])
            actual_next = sequence[i + 2]
            predicted_next = get_markov_prediction(current_state, transition_matrix)
            y_true.append(actual_next)
            y_pred.append(predicted_next if predicted_next != -1 else -1)

    # Calculate metrics
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{target_name} Markov Model Results: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}\n")

# Evaluate the LSTM model
def evaluate_lstm_model(model, X_test_seq, y_test_seq, label_encoder, target_name):
    preds = np.argmax(model.predict(X_test_seq), axis=1)
    y_true = np.argmax(y_test_seq, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="weighted", zero_division=0)
    recall = recall_score(y_true, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    print(f"\n{target_name} LSTM Model Results: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}\n")
    report = classification_report(y_true, preds, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_)))
    print(f"\nLSTM Model Performance - Classification Report for {target_name}:\n{report}")

# Evaluate both Markov and LSTM models for a specific target.
def evaluate_target_models(target_name, config, df, X, markov_matrix_path, lstm_model_path, label_encoder_path):
    # Markov Model Evaluation
    with open(markov_matrix_path, "rb") as f:
        markov_matrix = pickle.load(f)

    print(f"\n--- Evaluating {target_name} ---")
    target_sequences = df.groupby("conversation_id")[target_name].apply(list)
    evaluate_markov_model(target_sequences, markov_matrix, target_name)

    # LSTM Model Evaluation
    model = load_model(lstm_model_path)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    y_target = label_encoder.transform(df[target_name])
    sequence_len = config["lstm"][target_name]["sequence_len"]
    X_seq, y_seq = create_sequences(X, y_target, df["conversation_id"].to_numpy(), sequence_len=sequence_len)
    y_seq = to_categorical(y_seq, num_classes=len(label_encoder.classes_))
    evaluate_lstm_model(model, X_seq, y_seq, label_encoder, target_name)

# Main execution block: Load data and evaluate models
if __name__ == "__main__":
    with open("../config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Load and prepare test data: Combine embeddings with metadata features (speaker and turn)
    df = pd.read_csv("../data/processed/test_data.csv")
    X = prepare_features(df)

    # Evaluate Response Type
    evaluate_target_models(target_name="response_type", config=config, df=df, X=X,
        markov_matrix_path="../models/markov_matrices/response_type_markov_transition_matrix.pkl",
        lstm_model_path="../models/lstm_models/response_type_model_v1.h5",
        label_encoder_path="../models/label_encoders/label_encoder_response_type.pkl",
    )

    # Evaluate Conversation Stage
    evaluate_target_models(target_name="conversation_stage", config=config, df=df, X=X,
        markov_matrix_path="../models/markov_matrices/conversation_stage_markov_transition_matrix.pkl",
        lstm_model_path="../models/lstm_models/conversation_stage_model_v1.h5",
        label_encoder_path="../models/label_encoders/label_encoder_conversation_stage.pkl",
    )
