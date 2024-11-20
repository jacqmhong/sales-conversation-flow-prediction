"""
This script splits the dataset into training, validation, and testing sets by conversation IDs, rather than individual rows.
The purpose is to ensure that snippets of conversations are kept together during the split.
In addition, a LabelEncoder is created here to ensure all labels are accounted for consistently.

Outputs:
    train_data.csv: Conversations assigned to the training set.
    val_data.csv: Conversations assigned to the validation set.
    test_data.csv: Conversations assigned to the test set.
    label_encoder_response_type.pkl: LabelEncoder object with all unique labels from response_type
    label_encoder_conversation_stage.pkl: LabelEncoder object with all unique labels from conversation_stage.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Split the full dataset into train and test sets by conversations
df = pd.read_csv("../data/processed/labeled_with_embeddings.csv")
conversation_ids = df["conversation_id"].unique()
train_ids, test_ids = train_test_split(conversation_ids, test_size=0.2, random_state=0)

# Also split train IDs into train and validation sets
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=0)

# Create and save the train, val, and test dfs
train_df = df[df["conversation_id"].isin(train_ids)]
val_df = df[df["conversation_id"].isin(val_ids)]
test_df = df[df["conversation_id"].isin(test_ids)]

train_df.to_csv("../data/processed/train_data.csv", index=False)
val_df.to_csv("../data/processed/val_data.csv", index=False)
test_df.to_csv("../data/processed/test_data.csv", index=False)

print("Data split completed.")
print(f"Train set: {len(train_df)} rows")
print(f"Validation set: {len(val_df)} rows")
print(f"Test set: {len(test_df)} rows")

# Create LabelEncoders for response_type and conversation_stage
for target_name in ["response_type", "conversation_stage"]:
    label_encoder = LabelEncoder()
    label_encoder.fit(df[target_name])
    with open(f"../models/label_encoders/label_encoder_{target_name}.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder for {target_name} saved.")
