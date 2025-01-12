""" 
RAG GOAL is to retrieve the most relevant documents for a given query and use that document to augment an nput to an LLM so it can generate a more relevant response. 
"""
#  Similarty Search 
import random
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer

device = "cuda"

text_chunks_and_embedding_df = pd.read_csv("./embedded_chunks.csv")

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)

pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist(), axis=0), dtype=torch.float32).to(device)

print(embeddings.shape)

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)

# 1. Define the query
query = "Retail Individual Shareholders"

#2. Embed the query
query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)

#3. get similarity scores with the dot product

start_time = timer()
dot_scores = util.dot_score(a = query_embedding, b = embeddings)
end_timer = timer()
print(f"[INFO] the time taken to get scores on {len(embeddings)} embeddings: {end_timer - start_time:.5f} seconds")

#4. get the top k results (i.e. 5)
top_results_dot_product = torch.topk(dot_scores, k=5)

print(top_results_dot_product)

print(embeddings[102])

print(pages_and_chunks[102]["chunk"])


