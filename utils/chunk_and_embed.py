import tiktoken
import json

# Step 1: Function to chunk by token count
def chunk_by_tokens(text, max_tokens=500, model_name="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model_name)  # Tokenizer for the model
    words = text.split()  # Split text into words
    chunks = []  # To store chunks
    current_chunk = []  # Current chunk of words
    current_length = 0  # Current length in tokens

    for word in words:
        word_tokens = len(encoding.encode(word + " "))  # Token count for each word
        if current_length + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))  # Add the chunk to the list
            current_chunk = [word]  # Start a new chunk with the current word
            current_length = word_tokens  # Reset token count
        else:
            current_chunk.append(word)  # Add word to current chunk
            current_length += word_tokens  # Increase token count

    if current_chunk:  # Add the last chunk if any
        chunks.append(" ".join(current_chunk))

    return chunks

# Step 2: Load extracted text from file
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 3: Chunk the text based on token limit (e.g., 500 tokens)
chunks = chunk_by_tokens(text, max_tokens=500)
print(f"✅ Created {len(chunks)} chunks based on token limits.")

# Step 4: Save chunks to a JSON file for future use
with open("chunked_text.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Saved chunks to chunked_text.json.")

# Optional: Display first 3 chunks to verify
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}: \n{chunk}\n---")
