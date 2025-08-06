# In functions/main.py

import os
import time
import json
import uuid
import faiss
import uvicorn
import requests
import numpy as np
import tempfile
import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Configuration ---
os.environ['KMP_DUPLICATE_LIB_OK']='True'
load_dotenv()

# Configure APIs using environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
HACKRX_AUTH_TOKEN = os.environ.get("HACKRX_AUTH_TOKEN") # <-- FIXED: Loaded securely

# --- Pydantic Models ---
class HackRxInput(BaseModel):
    documents: str
    questions: list[str]

class HackRxOutput(BaseModel):
    answers: list[str]

# --- Authentication ---
auth_scheme = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not HACKRX_AUTH_TOKEN:
        raise HTTPException(status_code=500, detail="Authentication token not configured on server.")
    if credentials.scheme != "Bearer" or credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Helper Functions ---
def get_embedding(text, model="models/text-embedding-004"): # <-- UPDATED: Using newer embedding model
    try:
        response = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_QUERY")
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def download_pdf(url: str) -> str | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
        with open(temp_filename, "wb") as f:
            f.write(response.content)
        return temp_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None

# ... (Other helper functions like load_and_parse_pdf, chunk_text are fine as they were) ...
def load_and_parse_pdf(pdf_path: str):
    """Loads and parses a single PDF using a hybrid strategy."""
    all_text = []
    print(f"Processing {pdf_path}...")
    
    # Extract text with unstructured
    elements = partition_pdf(pdf_path)
    text = "\n".join([el.text for el in elements])
    all_text.append(text)

    # Extract tables with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(["\t".join(str(cell) if cell is not None else "" for cell in row) for row in table])
                if table_text:
                    all_text.append(f"\n--- TABLE DATA ---\n{table_text}\n--- END TABLE ---\n")
    
    return all_text

def chunk_text(text_elements, chunk_size=1000, chunk_overlap=200):
    """Chunks text elements."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for text in text_elements:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks


# --- FastAPI Application ---
app = FastAPI(
    title="HackRx API",
    description="API for HackRx Hackathon with real-time RAG processing.",
    version="1.0.0"
)

@app.post("/api/v1/hackrx/run", response_model=HackRxOutput)
async def run_hackrx_pipeline(payload: HackRxInput, token: str = Depends(validate_token)):
    pdf_url = payload.documents
    questions = payload.questions
    
    temp_pdf_path = download_pdf(pdf_url)
    if not temp_pdf_path:
        raise HTTPException(status_code=500, detail="Failed to download PDF from URL.")

    try:
        text_elements = load_and_parse_pdf(temp_pdf_path)
        chunks = chunk_text(text_elements)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to extract text from the document.")

        print("Generating document embeddings...")
        doc_embeddings_response = genai.embed_content(
            model="models/text-embedding-004", # <-- UPDATED
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings_array = np.array(doc_embeddings_response['embedding']).astype('float32')
        dimension = embeddings_array.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        answers = []
        generative_model = genai.GenerativeModel('gemini-2.5-flash') # <-- FIXED: Correct model name

        for question in questions:
            question_embedding = get_embedding(question)
            if question_embedding is None:
                answers.append("Error: Could not generate embedding for the question.")
                continue

            D, I = index.search(np.array([question_embedding]).astype('float32'), k=5)
            retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
            **Role:** You are a highly precise AI assistant for analyzing policy documents.
            **Instruction:** Answer the user's question based *exclusively* on the information in the "Context from Policy Document" below.
            If the context does not contain the answer, state: "I cannot answer that question based on the information provided."

            ---
            **Context from Policy Document:**
            {context}
            ---
            **User's Question:** {question}
            ---
            **Answer:**
            """
            
            try:
                response = generative_model.generate_content(prompt)
                answers.append(response.text.strip())
            except Exception as e:
                print(f"Error generating answer: {e}")
                answers.append("Error generating answer from the model.")

    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    return HackRxOutput(answers=answers)


# Block for local testing. Render/Railway will use the Gunicorn command instead.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
