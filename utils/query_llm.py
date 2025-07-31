# query_llm.py

import json
import faiss
import numpy as np
import cohere

# Load API key
API_KEY = "wcIlRU4zY4baDtEqydme649M1OzKXVPw9SJZpjwQ"  # Replace with env var if needed
co = cohere.Client(API_KEY)

# Load FAISS index and chunks
index = faiss.read_index("llm_index.faiss")
with open("llm_chunks.json", "r", encoding="utf-8") as f:
    text_chunks = json.load(f)

# Take user query
query = input("🔍 Enter your question: ")

# Embed the query
response = co.embed(
    texts=[query],
    model="embed-english-v3.0",
    input_type="search_document"
)
query_vector = np.array(response.embeddings).astype("float32")

# Search in FAISS index
top_k = 3
distances, indices = index.search(query_vector, top_k)

# Get top chunks
retrieved_chunks = [text_chunks[i] for i in indices[0]]

print("\n🎯 Top Relevant Chunks:\n")
for chunk in retrieved_chunks:
    print("👉", chunk)
    print("-" * 80)

# Merge for LLM context
retrieved_text = "\n\n".join(retrieved_chunks)

# Generate final answer with Cohere Chat
llm_response = co.chat(
    model="command-r-plus",
    temperature=0.3,
    chat_history=[],
    message=f"""You are a helpful assistant trained on health insurance policies.

User Question:
{query}

Relevant Policy Extract:
{retrieved_text}

Instructions:
- Start with a short Yes/No/Maybe.
- Then clearly explain why, using evidence from the extract.
- Keep the explanation concise, accurate, and helpful.
"""
)

# Print LLM output
print("\n🧠 Final Answer:\n")
print(llm_response.text.strip())
