"""
This script implements a second-order Markov model to predict the next response type,
conversation stage, or cluster in a sales conversation. Using accuracy, precision, recall, and F1-score,
it evaluates the performance of the Markov model as a baseline for the LSTM model in train.py.

Outputs:
    - Transition matrices for response_type, conversation_stage, and clusters.
    - Performance evaluation of the Markov model on the test dataset.
"""
from collections import defaultdict
from evaluate import evaluate_markov_model
import pandas as pd
import pickle

# Load and sort the data by conversation_id and turn order
train_df = pd.read_csv("../data/processed/train_data.csv") # to build the transition matrix
train_df = train_df.sort_values(by=["conversation_id", "turn"])
test_df = pd.read_csv("../data/processed/test_data.csv") # for evaluation
test_df = test_df.sort_values(by=["conversation_id", "turn"])

def build_transition_matrix(sequences):
    """
    Creates a second-order Markov transition matrix from labeled sequences.

    Counts transitions between pairs of consecutive states and the subsequent
    state to build probabilities for each possible next state.

    Parameters:
    - sequences (list of lists): A list of sequences, where each sequence is
      a list of states (e.g., labels or cluster IDs).

    Returns:
    - dict: A transition matrix where keys are tuples representing the current
      state (pair of consecutive states) and values are dictionaries mapping
      next states to their probabilities.
    """
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

def process_markov_model(target_name, train_df, test_df):
    """
    Builds, evaluates, and saves a Markov transition matrix for a target variable.

    Parameters:
    - target_name (str): Name of the target variable.
    - train_df (pd.DataFrame): Training dataframe, grouped by conversation_id,
      containing labeled sequences for the target variable.
    - test_df (pd.DataFrame): Test dataframe, grouped by conversation_id,
      containing labeled sequences for evaluation.
    """
    # Build and evaluate the transition matrix
    train_sequences = train_df.groupby("conversation_id")[target_name].apply(list)
    transition_matrix = build_transition_matrix(train_sequences) # build
    test_sequences = test_df.groupby("conversation_id")[target_name].apply(list)
    evaluate_markov_model(test_sequences, transition_matrix, target_name) # evaluate

    output_path = f"../models/markov_matrices/{target_name}_markov_transition_matrix.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(transition_matrix, f)

    print(f"Sample state pairs in the {target_name} transition matrix:")
    print(list(transition_matrix.keys())[:10])


if __name__ == "__main__":
    process_markov_model("response_type", train_df, test_df)
    process_markov_model("conversation_stage", train_df, test_df)
    process_markov_model("cluster", train_df, test_df) # cluster transitions
