"""
This script trains LSTM models for predicting response types and conversation stages in sales conversations.
It combines embeddings with metadata features, builds and trains the models, and then evaluates their performance
on the validation set using accuracy, precision, recall, and F1-score. The trained models and their corresponding
metrics are saved for later use and comparison.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import yaml

# Load initial config and data -- Task-specific tuning can be done later.
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

# Make fixed-length sequences for LSTM
def create_sequences(X, y, convo_ids, sequence_len):
    # X: [[A], [B], [C], [D], [E]] for a single convo
    # y: [0, 0, 1, 1, 0],  seq_len = 3
    # Output X_seq: [[[A], [B], [C]], [[B], [C], [D]], [[C], [D], [E]]], y_seq: [1, 1, 0]
    X_seq, y_seq = [], []
    unique_convos = np.unique(convo_ids)
    for convo_id in unique_convos:
        convo_X = X[convo_ids == convo_id]
        convo_y = y[convo_ids == convo_id]
        for i in range(sequence_len, len(convo_X)):
            X_seq.append(convo_X[i-sequence_len:i])
            y_seq.append(convo_y[i])
    return np.array(X_seq), np.array(y_seq)

X_train = prepare_features(train_df)
X_val = prepare_features(val_df)

# Model parameters
input_dim = X_train.shape[1] # embedding dim + metadata dim (2)
lstm_units = config["lstm"]["units"]
dropout_rate = config["lstm"]["dropout"]
batch_size = config["lstm"]["batch_size"]
max_epochs = config["lstm"]["epochs"]
learning_rate = config["lstm"]["learning_rate"]
sequence_len = config["lstm"]["sequence_len"]

def train_and_save_lstm(target_name):
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
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "precision", "recall"]) # class imbalance - accuracy isn't enough

    # Train and evaluate model
    print(f"Training LSTM model for {target_name} prediction...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # for epochs
    history = model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping])
    results = model.evaluate(X_val_seq, y_val_seq, verbose=0)
    loss, accuracy, precision, recall = results
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"\nVal Results for {target_name}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1_score:.2f}\n")

    # Generate and save classification report
    y_pred = np.argmax(model.predict(X_val_seq), axis=1)
    y_true = np.argmax(y_val_seq, axis=1)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_)))
    print(f"\nClassification Report for {target_name}:\n{report}")
    with open(f"../models/lstm_models/{target_name}_classification_report.txt", "w") as f:
        f.write(report)

    # Save model and training metrics
    model.save(f"../models/lstm_models/lstm_{target_name}_model_with_metadata.h5")
    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv(f"../models/lstm_models/{target_name}_training_metrics.csv", index=False)

# Main execution block: Train and save response_type and conversation_stage models
if __name__ == "__main__":
    train_and_save_lstm("response_type")
    train_and_save_lstm("conversation_stage")
