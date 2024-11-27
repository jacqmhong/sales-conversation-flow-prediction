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

# Load configs
with open("../config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
with open("../config/combined_action_mapping.yaml", "r") as mapping_file:
    combined_action_mapping = yaml.safe_load(mapping_file)["combined_action_mapping"]

# Load LSTM models and encoders
embed_model_name = config["embedding"]["model_name"]
embedding_model = SentenceTransformer(embed_model_name)
response_type_model = load_model("../models/lstm_models/lstm_response_type_model_with_metadata.h5")
conversation_stage_model = load_model("../models/lstm_models/lstm_conversation_stage_model_with_metadata.h5")
with open("../models/label_encoders/label_encoder_response_type.pkl", "rb") as f:
    response_type_label_encoder = pickle.load(f)
with open("../models/label_encoders/label_encoder_conversation_stage.pkl", "rb") as f:
    conversation_stage_label_encoder = pickle.load(f)

# Load PCA, KMeans, and cluster transition matrix
with open("../models/pca_model.pkl", "rb") as f:
    pca_model = pickle.load(f)
with open("../models/kmeans_model.pkl", "rb") as f:
    clustering_model = pickle.load(f)
with open("../models/markov_matrices/cluster_markov_transition_matrix.pkl", "rb") as f:
    cluster_transition_matrix = pickle.load(f)

# Preprocess incoming convo data
def preprocess_realtime_data(data, sequence_len):
    # Generate embeddings
    df = pd.DataFrame(data) # json to df
    texts = df["response"].tolist()
    embeddings = embedding_model.encode(texts)

    # Add metadata (speaker, turn)
    speaker = df["speaker"].map({"Customer": 0, "Salesman": 1}).to_numpy()
    turn_number = np.arange(1, len(df) + 1) / len(df)
    X = np.hstack([embeddings, speaker.reshape(-1, 1), turn_number.reshape(-1, 1)]) # single feature arr

    # Create sequences or pad short convos
    if len(X) < sequence_len:
        X = pad_sequences([X], maxlen=sequence_len, padding="pre", dtype="float32")[0]
        X_seq = np.expand_dims(X, axis=0)  # match LSTM input shape
    else:
        X_seq, _ = create_sequences(X, None, np.zeros(len(X)), sequence_len=sequence_len)
    return X_seq

# Top prediction and probabilities of the next response_type
@app.route("/predict-prospect-response", methods=["POST"])
def predict_prospect_response():
    try:
        input_data = request.get_json()
        sequence_len = config["lstm"]["response_type"]["sequence_len"]
        X_seq = preprocess_realtime_data(input_data["history"], sequence_len) # preprocess
        response_type_probs = response_type_model.predict(X_seq)[-1].tolist() # lstm pred probabilities
        response_type_probs = {label: round(prob, 2) for label, prob in zip(response_type_label_encoder.classes_, response_type_probs)}
        top_prediction = max(response_type_probs, key=response_type_probs.get)
        return jsonify({"top_prediction": top_prediction, "response_type_probabilities": response_type_probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Top prediction and probabilities of the next conversation_stage
@app.route("/predict-next-conversation-stage", methods=["POST"])
def predict_conversation_stage():
    try:
        input_data = request.get_json()
        sequence_len = config["lstm"]["conversation_stage"]["sequence_len"]
        X_seq = preprocess_realtime_data(input_data["history"], sequence_len) # preprocess
        conv_stage_probs = conversation_stage_model.predict(X_seq)[-1].tolist() # lstm pred probabilities
        conv_stage_probs = {label: round(prob, 2) for label, prob in zip(conversation_stage_label_encoder.classes_, conv_stage_probs)}
        top_prediction = max(conv_stage_probs, key=conv_stage_probs.get)
        return jsonify({"top_prediction": top_prediction, "conversation_stage_probabilities": conv_stage_probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Suggests a next action according to the resulting response_type and conversation_stage prediction combination
@app.route("/suggest-sales-response", methods=["POST"])
def suggest_sales_response():
    try:
        # Preprocess data
        input_data = request.get_json()
        response_seq_len = config["lstm"]["response_type"]["sequence_len"]
        stage_seq_len = config["lstm"]["conversation_stage"]["sequence_len"]
        X_seq_response = preprocess_realtime_data(input_data["history"], response_seq_len)
        X_seq_stage = preprocess_realtime_data(input_data["history"], stage_seq_len)

        # Predict response_type probabilities
        response_type_probs = response_type_model.predict(X_seq_response)[-1].tolist()
        response_type_probs = {label: round(prob, 2) for label, prob in zip(response_type_label_encoder.classes_, response_type_probs)}
        top_response_type = max(response_type_probs, key=response_type_probs.get)

        # Predict conversation_stage probabilities
        conv_stage_probs = conversation_stage_model.predict(X_seq_stage)[-1].tolist()
        conv_stage_probs = {label: round(prob, 2) for label, prob in zip(conversation_stage_label_encoder.classes_, conv_stage_probs)}
        top_conversation_stage = max(conv_stage_probs, key=conv_stage_probs.get)

        # Determine the next best action
        conv_stage_actions = combined_action_mapping[top_conversation_stage]
        combined_action = conv_stage_actions.get(top_response_type, "No specific action available for this combination.")

        return jsonify({
            "suggested_sales_response": combined_action,
            "top_response_type": top_response_type,
            "top_conversation_stage": top_conversation_stage,
            "response_type_probabilities": response_type_probs,
            "conversation_stage_probabilities": conv_stage_probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Markov model uses cluster transition probabilities to predict the next likely state
@app.route("/predict-conversation-direction", methods=["POST"])
def predict_conversation_direction():
    try:
        # Get the last two responses from the convo history (due to second-order markov matrix)
        input_data = request.get_json()
        convo_history = input_data["history"]
        if len(convo_history) < 2:
            return jsonify({"error": "At least two conversation snippets are required for prediction."}), 400
        last_two_responses = [snippet["response"] for snippet in convo_history[-2:]]

        # Generate embeddings, apply PCA, and assign clusters
        embeddings = embedding_model.encode(last_two_responses)
        reduced_embeddings = pca_model.transform(embeddings)
        clusters = clustering_model.predict(reduced_embeddings)

        # Get the current state in the Markov model
        current_state = tuple(clusters)
        transition_probs = cluster_transition_matrix.get(current_state, {})
        if not transition_probs:
            return jsonify({"error": "No transition data available for the current cluster state."}), 400

        # Predict the most likely next cluster and provide all transition probabilities
        next_cluster = max(transition_probs, key=transition_probs.get)
        sorted_transitions = {int(cluster): round(prob, 2) for cluster, prob in sorted(transition_probs.items(), key=lambda item: item[1], reverse=True)}

        # Can add suggested_response again
        # Can assign attributes to clusters rather than reporting the cluster label
        return jsonify({
            "current_state": [int(c) for c in current_state],
            "predicted_next_cluster": int(next_cluster),
            "transition_probabilities": sorted_transitions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    # app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
