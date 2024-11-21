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
from train import prepare_features

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
            y_pred.append(predicted_next if predicted_next else "Other")

    # Calculate metrics
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{target_name} Markov Model Results: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}\n")

# Evaluate the LSTM model
def evaluate_lstm_model(model, X_test, y_test, label_encoder, target_name):
    preds = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="weighted", zero_division=0)
    recall = recall_score(y_true, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    print(f"\n{target_name} LSTM Model Results: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}\n")
    report = classification_report(y_true, preds, target_names=label_encoder.classes_)
    print(f"\nLSTM Model Performance - Classification Report for {target_name}:\n{report}")

# Main execution block: Load data and evaluate models
if __name__ == "__main__":
    # Load and prepare test data: Combine embeddings with metadata features (speaker and turn)
    df = pd.read_csv("../data/processed/test_data.csv")
    X = prepare_features(df)

    # Load Markov transition matrices
    with open("../models/markov_matrices/response_type_markov_transition_matrix.pkl", "rb") as f:
        response_type_markov_matrix = pickle.load(f)
    with open("../models/markov_matrices/conversation_stage_markov_transition_matrix.pkl", "rb") as f:
        conversation_stage_markov_matrix = pickle.load(f)

    # Load LSTM models and encoders
    response_type_model = load_model("../models/lstm_models/lstm_response_type_model_with_metadata.h5")
    conv_stage_model = load_model("../models/lstm_models/lstm_conversation_stage_model_with_metadata.h5")
    with open("../models/label_encoders/label_encoder_response_type.pkl", "rb") as f:
        response_type_label_encoder = pickle.load(f)
    with open("../models/label_encoders/label_encoder_conversation_stage.pkl", "rb") as f:
        conv_stage_label_encoder = pickle.load(f)

    # Response Type Evaluation: Markov and LSTM
    print("\n--- Evaluating Response Type ---")
    response_sequences = df.groupby("conversation_id")["response_type"].apply(list)
    evaluate_markov_model(response_sequences, response_type_markov_matrix, "response_type") # markov
    y_response = to_categorical(response_type_label_encoder.transform(df["response_type"]))
    evaluate_lstm_model(response_type_model, np.expand_dims(X, axis=2), y_response, response_type_label_encoder, "response_type") # lstm

    # Conversation Stage Evaluation: Markov and LSTM
    print("\n--- Evaluating Conversation Stage ---")
    conversation_sequences = df.groupby("conversation_id")["conversation_stage"].apply(list)
    evaluate_markov_model(conversation_sequences, conversation_stage_markov_matrix, "conversation_stage") # markov
    y_conv_stage = to_categorical(conv_stage_label_encoder.transform(df["conversation_stage"]))
    evaluate_lstm_model(conv_stage_model, np.expand_dims(X, axis=2), y_conv_stage, conv_stage_label_encoder, "conversation_stage") # lstm
