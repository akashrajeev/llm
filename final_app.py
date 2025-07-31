# final_app.py
import os
import json
import requests
import cohere
import tempfile
import faiss
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader

app = FastAPI(
    title="HackRx LLM API",
    version="1.0",
    description="Submit policy PDF and questions to get concise one-line answers."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use environment variable instead of hardcoding the API key
co = cohere.Client(os.environ["COHERE_API_KEY"])

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            reader = PdfReader(tmp_file.name)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

def chunk_text(text: str, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def embed_chunks(chunks: List[str]):
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings

def build_index(embeddings: List[List[float]]):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search(index, query_vector, k=3):
    distances, indices = index.search(np.array(query_vector).astype("float32"), k)
    return indices[0]

def answer_question(question: str, chunks: List[str]) -> str:
    response = co.chat(
        message=question,
        documents=[{"text": chunk} for chunk in chunks]
    )
    return response.text.strip().replace("\n", " ").split(".")[0] + "."

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_hackrx_endpoint(payload: HackRxRequest):
    try:
        text = extract_text_from_url(payload.documents)
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to extract content from document.")
        embeddings = embed_chunks(chunks)
        index = build_index(embeddings)

        answers = []
        for q in payload.questions:
            query_embed = co.embed(texts=[q], model="embed-english-v3.0", input_type="search_query")
            top_indices = search(index, query_embed.embeddings)
            relevant_chunks = [chunks[i] for i in top_indices]
            llm_response = answer_question(q, relevant_chunks)
            answers.append(llm_response)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
