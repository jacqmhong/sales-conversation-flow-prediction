import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import yaml

# Embedding configs
with open("../config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

model_name = config["embedding"]["model_name"] # SBERT
output_path = config["embedding"]["output_path"]
batch_size = config["embedding"]["batch_size"]

# Dataset with the three categories: sentiment_label, conversation_stage, response_type
df = pd.read_csv("../data/processed/labeled_sequential_convos.csv")
model = SentenceTransformer(model_name)

# Generate the embeddings in batches
embeddings = []
for i in range(0, len(df), batch_size):
    batch_texts = df['text'][i:(i+batch_size)].tolist()
    batch_embeddings = model.encode(batch_texts)
    embeddings.extend(batch_embeddings)

# Save the embeddings
with open(output_path, 'wb') as f:
    pickle.dump(np.array(embeddings), f)

print("Embedding generation completed and saved.")
