from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train import create_sequences
import yaml

app = Flask(__name__)
CORS(app)

# Load config
with open("../config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Load models and encoders
embed_model_name = config["embedding"]["model_name"]
embedding_model = SentenceTransformer(embed_model_name)
response_type_model = load_model("../models/lstm_models/lstm_response_type_model_with_metadata.h5")
conversation_stage_model = load_model("../models/lstm_models/lstm_conversation_stage_model_with_metadata.h5")
with open("../models/label_encoders/label_encoder_response_type.pkl", "rb") as f:
    response_type_label_encoder = pickle.load(f)
with open("../models/label_encoders/label_encoder_conversation_stage.pkl", "rb") as f:
    conversation_stage_label_encoder = pickle.load(f)

# Preprocess incoming convo data
def preprocess_realtime_data(data, sequence_len):
    # Generate embeddings
    df = pd.DataFrame(data) # json to df
    texts = df["response"].tolist()
    embeddings = embedding_model.encode(texts)

    # Add metadata (speaker, turn)
    speaker = df["speaker"].map({"Customer": 0, "Salesman": 1}).to_numpy()
    df["normalized_turn"] = df.groupby("conversation_id")["turn"].transform(lambda x: x / x.max())
    turn_number = df["normalized_turn"].to_numpy()
    X = np.hstack([embeddings, speaker.reshape(-1, 1), turn_number.reshape(-1, 1)]) # single feature arr

    # Create sequences or pad short convos
    convo_ids = df["conversation_id"].to_numpy()
    if len(X) < sequence_len:
        X = pad_sequences([X], maxlen=sequence_len, padding="pre", dtype="float32")[0]
        convo_ids = pad_sequences([convo_ids], maxlen=sequence_len, padding="pre")[0]
        X_seq = np.expand_dims(X, axis=0)  # match LSTM input shape
    else:
        X_seq, _ = create_sequences(X, None, convo_ids, sequence_len=sequence_len)
    return X_seq

# Predict the response_type of the last turn in the json
@app.route("/predict-response-type", methods=["POST"])
def predict_response_type():
    try:
        input_data = request.get_json()
        sequence_len = config["lstm"]["response_type"]["sequence_len"]
        X_seq = preprocess_realtime_data(input_data, sequence_len) # preprocess
        preds = np.argmax(response_type_model.predict(X_seq), axis=1) # lstm pred
        labels = response_type_label_encoder.inverse_transform(preds)
        return jsonify({"lstm_predictions": labels.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Predict the conversation_stage of the last turn in the json
@app.route("/predict-conversation-stage", methods=["POST"])
def predict_conversation_stage():
    try:
        input_data = request.get_json()
        sequence_len = config["lstm"]["conversation_stage"]["sequence_len"]
        X_seq = preprocess_realtime_data(input_data, sequence_len) # preprocess
        preds = np.argmax(conversation_stage_model.predict(X_seq), axis=1) # lstm pred
        labels = conversation_stage_label_encoder.inverse_transform(preds)
        return jsonify({"lstm_predictions": labels.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
