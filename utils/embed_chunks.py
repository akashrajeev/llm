import cohere
import json

# Initialize Cohere with your API key
co = cohere.Client("wcIlRU4zY4baDtEqydme649M1OzKXVPw9SJZpjwQ")

# Load your chunked text
with open("chunked_text.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Embed the chunks
response = co.embed(
    texts=chunks,
    model="embed-english-v3.0",  # or try "embed-multilingual-v3.0"
    input_type="search_document"
)

embeddings = response.embeddings

# Save embeddings to file
with open("chunked_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f)

print(f"✅ Embedded {len(embeddings)} chunks and saved to chunked_embeddings.json")
