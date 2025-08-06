import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Pydantic Models for API ---
class HackRxInput(BaseModel):
    documents: str
    questions: list[str]

class HackRxOutput(BaseModel):
    answers: list[str]

# --- Authentication ---
auth_scheme = HTTPBearer()
HACKRX_AUTH_TOKEN = "ebdba744e4fbb5939b3dbc74559348314bbc1907834c39bb3103f124ad6ece19"

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Dependency to validate the Bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Helper Functions ---

def get_embedding(text, model="models/embedding-001"):
    """Generates an embedding for the given text."""
    try:
        response = genai.embed_content(model=model, content=text)
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def download_pdf(url: str) -> str:
    """Downloads a PDF from a URL and saves it to a temporary file."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Create a temporary file path in a cross-platform way
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
        
        with open(temp_filename, "wb") as f:
            f.write(response.content)
            
        return temp_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None

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
                    all_text.append(table_text)
    
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
async def run_hackrx_pipeline(
    payload: HackRxInput,
    token: str = Depends(validate_token)
):
    """
    Main endpoint to process a document and answer questions using a real-time RAG pipeline.
    """
    pdf_url = payload.documents
    questions = payload.questions
    
    # 1. Download the PDF
    temp_pdf_path = download_pdf(pdf_url)
    if not temp_pdf_path:
        raise HTTPException(status_code=500, detail="Failed to download PDF from URL.")

    try:
        # 2. Parse and chunk the document
        text_elements = load_and_parse_pdf(temp_pdf_path)
        chunks = chunk_text(text_elements)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to extract and chunk text from the document.")

        # 3. Generate embeddings in real-time (Optimized with Batching)
        print("Generating embeddings for all chunks...")
        batch_size = 100  # The Gemini API can handle up to 100 texts per request
        all_embeddings = []
        valid_chunks_with_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            try:
                # NOTE: The API call might differ slightly based on the library version.
                # This assumes a batch-capable function.
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=batch_chunks,
                    task_type="RETRIEVAL_DOCUMENT" # Specify task type for better performance
                )
                # Store the successful embeddings and their corresponding chunks
                for chunk, embedding in zip(batch_chunks, response['embedding']):
                    all_embeddings.append(embedding)
                    valid_chunks_with_embeddings.append(chunk)

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
            time.sleep(1) # A 1-second delay between large batches is good practice

        if not all_embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate any embeddings.")

        # Update the rest of the code to use the new lists
        chunks = valid_chunks_with_embeddings
        embeddings_array = np.array(all_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        # 4. Build FAISS index in memory
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # 5. Process each question
        answers = []
        # Initialize the model ONCE before the loop
        generative_model = genai.GenerativeModel('gemini-2.5-flash')

        for question in questions:
            # a. Generate question embedding
            question_embedding = get_embedding(question)
            if question_embedding is None:
                answers.append("Could not generate embedding for the question.")
                continue

            # b. Search FAISS index
            D, I = index.search(np.array([question_embedding]).astype('float32'), k=5)
            retrieved_chunks = [chunks[i] for i in I[0] if i != -1]
            context = "\n\n".join(retrieved_chunks)

            # c. Construct prompt and generate answer
            prompt = f"""
            **Role:** You are a highly precise AI assistant for analyzing insurance policy documents.

            **Primary Instruction:** Your task is to answer the user's question based *exclusively* on the information contained within the "Context from Policy Document" provided below. Do not use any external knowledge.

            **Response Requirements:**
            1.  **Accuracy:** Your answer must be factually correct and directly traceable to the provided context.
            2.  **Clarity:** Provide a clear, direct, and unambiguous answer.
            3.  **Justification:** If possible, briefly quote the key phrase or sentence from the context that justifies your answer.
            4.  **Safety:** If the context does not contain the information needed to answer the question, you MUST respond with the exact phrase: "I'm sorry, but I cannot answer that question based on the information in the provided policy documents."

            ---

            **Context from Policy Document:**
            {context}

            ---

            **User's Question:**
            {question}

            ---

            **Answer:**
            """
            
            try:
                # Use the already initialized model
                response = generative_model.generate_content(prompt)
                answers.append(response.text.strip()) # Use .strip() to remove leading/trailing whitespace
            except Exception as e:
                print(f"Error generating answer: {e}")
                answers.append("Error generating answer from the model.")

    finally:
        # 6. Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    return {"answers": answers}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
