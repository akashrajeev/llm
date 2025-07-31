# faiss_index.py

import json
import faiss
import numpy as np

# Load embeddings and text chunks
with open("chunked_embeddings.json", "r", encoding="utf-8") as f:
    embeddings = json.load(f)

with open("chunked_text.json", "r", encoding="utf-8") as f:
    text_chunks = json.load(f)

# Convert to numpy array
embedding_matrix = np.array(embeddings).astype("float32")

# Initialize FAISS index
dimension = len(embedding_matrix[0])
index = faiss.IndexFlatL2(dimension)  # You can use IndexFlatIP for cosine similarity

# Add embeddings to index
index.add(embedding_matrix)

# Save index
faiss.write_index(index, "llm_index.faiss")

# Save the corresponding text chunks
with open("llm_chunks.json", "w", encoding="utf-8") as f:
    json.dump(text_chunks, f)

print("✅ FAISS index built and saved!")
