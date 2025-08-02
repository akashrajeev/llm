# final_app.py - FINAL VERSION (Answers All Questions)

import json
import requests
import cohere
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from typing import List, Annotated
import tempfile
import faiss
import numpy as np

app = FastAPI(
    title="HackRx LLM API - Final Submission",
    version="4.0",
    description="Processes a live document URL and answers all questions in the list."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
SECRET_TOKEN = "bc7aba495dfdaa88e718dec0d0fba29a2d45eaecac3268453778d027c7419081"
COHERE_API_KEY = "Xklhtw9AKK4gEItHZSjOskNSf7sbd9DQvi14JDoi"  # IMPORTANT: Paste your real key here

co = cohere.Client(COHERE_API_KEY)

# --- Security Function ---
async def verify_token(authorization: Annotated[str, Header()]):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme.")
    token = authorization.split(" ")[1]
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token or key.")

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Helper Functions (Real-time processing) ---
def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=15) # Increased timeout for larger files
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
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

def answer_question(question: str, chunks: List[str]) -> str:
    prompt = f"""
    Based ONLY on the provided documents, answer the following user question.
    User Question: "{question}"
    Instructions:
    1. Your answer must be a single, concise line.
    2. Be direct and factual. Do not apologize or use conversational fluff.
    3. If the answer is not in the documents, respond with "Information not found in the policy."
    """
    response = co.chat(message=prompt, documents=[{"text": chunk} for chunk in chunks], temperature=0.1)
    return response.text.strip().split('\n')[0]

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx_endpoint(
    payload: HackRxRequest, 
    token: Annotated[str, Depends(verify_token)]
):
    try:
        # Step 1: Real-time document processing
        text = extract_text_from_url(payload.documents)
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to extract content from document.")
        
        embeddings = co.embed(texts=chunks, model="embed-english-v3.0", input_type="search_document").embeddings
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype("float32"))

        # Step 2: Embed all questions in one batch for optimization
        query_embeddings = co.embed(
            texts=payload.questions, 
            model="embed-english-v3.0", 
            input_type="search_query"
        ).embeddings

        # Step 3: Process ALL questions in the list
        answers = []
        for q, q_embed in zip(payload.questions, query_embeddings):
            
            query_vector = np.array([q_embed]).astype("float32")
            distances, indices = index.search(query_vector, k=3)
            
            relevant_chunks = [chunks[i] for i in indices[0]]
            llm_response = answer_question(q, relevant_chunks)
            answers.append(llm_response)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))