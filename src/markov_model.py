# Second-Order Markov Model

from collections import defaultdict
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

# Load and sort the dataset by conversation_id and turn order
df = pd.read_csv("../data/processed/labeled_with_embeddings.csv")
df = df.sort_values(by=["conversation_id", "turn"])

# Create the 2nd-order transition matrix (1st-order yielded worse results)
def build_transition_matrix(sequences):
    # Count transitions for pairs of consecutive labels, eg. ('Disagreement', 'Question')
    transition_counts = defaultdict(lambda: defaultdict(int))
    for sequence in sequences:
        for i in range(len(sequence) - 2):
            curr_state = (sequence[i], sequence[i + 1])
            next_state = sequence[i + 2]
            transition_counts[curr_state][next_state] += 1

    # Normalize counts to probabilities
    transition_matrix = {}
    for curr_state, next_states in transition_counts.items():
        total_transitions = sum(next_states.values())  # Total transitions from this pair
        probabilities = {next_label: count / total_transitions for next_label, count in next_states.items()}
        transition_matrix[curr_state] = probabilities

    return transition_matrix

def predict_next_response(current_state, transition_matrix):
    if current_state not in transition_matrix:
        return None  # no data for this state
    return max(transition_matrix[current_state], key=transition_matrix[current_state].get)

# Evaluate the model with accuracy, precision, recall, and f1-score
def evaluate_markov_model(sequences, transition_matrix):
    y_true = []
    y_pred = []

    for sequence in sequences:
        for i in range(len(sequence) - 2):
            current_state = (sequence[i], sequence[i + 1])
            actual_next = sequence[i + 2]
            predicted_next = predict_next_response(current_state, transition_matrix)

            y_true.append(actual_next)
            y_pred.append(predicted_next if predicted_next else "Other")

    # Calculate metrics
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

# Process for response_type
response_sequences = df.groupby("conversation_id")["response_type"].apply(list)
response_transition_matrix = build_transition_matrix(response_sequences)
evaluate_markov_model(response_sequences, response_transition_matrix)

with open("../models/response_type_markov_transition_matrix.pkl", "wb") as f:
    pickle.dump(response_transition_matrix, f)

print("Sample state pairs in the response_type transition matrix:")
print(list(response_transition_matrix.keys())[:10])

# Process for conversation_stage
stage_sequences = df.groupby("conversation_id")["conversation_stage"].apply(list)
stage_transition_matrix = build_transition_matrix(stage_sequences)
evaluate_markov_model(stage_sequences, stage_transition_matrix)

with open("../models/conversation_stage_markov_transition_matrix.pkl", "wb") as f:
    pickle.dump(stage_transition_matrix, f)

print("Sample state pairs in the conversation_stage transition matrix:")
print(list(stage_transition_matrix.keys())[:10])
