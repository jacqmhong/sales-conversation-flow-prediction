import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import yaml

# Embedding configs
with open("../config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

model_name = config["embedding"]["model_name"]
output_path = config["embedding"]["output_path"]
batch_size = config["embedding"]["batch_size"]
pca_components = config["embedding"]["pca_components"]

# Dataset with the three categories: sentiment_label, conversation_stage, response_type
df = pd.read_csv("../data/processed/labeled_sequential_convos.csv")
model = SentenceTransformer(model_name)

# Generate the embeddings in batches
embeddings = []
for i in range(0, len(df), batch_size):
    batch_texts = df['text'][i:(i+batch_size)].tolist()
    batch_embeddings = model.encode(batch_texts)
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings)

# PCA
print(f"Applying PCA to reduce embeddings to {pca_components} dimensions.")
pca = PCA(n_components=pca_components, random_state=0)
reduced_embeddings = pca.fit_transform(embeddings)

# Save reduced embeddings
reduced_output_path = output_path.replace(".pkl", "_reduced.pkl")
with open(reduced_output_path, 'wb') as f:
    pickle.dump(reduced_embeddings, f)

print(f"Reduced embeddings saved to {reduced_output_path}.")

# Save the original embeddings (still used in cluster_analysis.ipynb to show the difference that PCA made)
with open(output_path, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Original embeddings saved to {output_path}.")

# Add embeddings to df
df['embeddings'] = embeddings.tolist()
df.to_csv("../data/processed/labeled_with_embeddings.csv", index=False)

print("Embedding generation completed and saved.")
