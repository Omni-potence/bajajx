import pdfplumber
import google.generativeai as genai
import faiss
import numpy as np
import json
import os
import time
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API (replace with your actual API key)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_embedding(text):
    """Generates an embedding for the given text using the Gemini embedding model."""
    try:
        response = genai.embed_content(model="models/gemini-embedding-001", content=text)
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def load_and_parse_pdfs_hybrid(pdf_paths):
    """Loads and parses PDFs using a hybrid strategy."""
    all_text = []
    for pdf_path in pdf_paths:
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
    """Chunks text elements into fixed-size chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for text in text_elements:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks

def build_faiss_index_ivfpq(chunks):
    """Builds a FAISS IndexIVFPQ index from text chunks."""
    embeddings = []
    valid_chunks = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        if embedding:
            embeddings.append(embedding)
            valid_chunks.append(chunk)
        time.sleep(1)  # Add a 1-second delay
    
    if not embeddings:
        print("No valid embeddings generated. Cannot build FAISS index.")
        return None, None

    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    
    nlist = 5  # Number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
    
    print("Training FAISS index...")
    index.train(embeddings_array)
    print("Adding vectors to FAISS index...")
    index.add(embeddings_array)
    print("FAISS index built successfully.")
    
    return index, valid_chunks

if __name__ == "__main__":
    pdf_dir = os.path.join(os.path.dirname(__file__), "docs")
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'. Please ensure your policy PDFs are in this directory.")
    else:
        text_elements = load_and_parse_pdfs_hybrid(pdf_files)
        if text_elements:
            chunks = chunk_text(text_elements)
            if chunks:
                faiss_index, saved_chunks = build_faiss_index_ivfpq(chunks)
                if faiss_index and saved_chunks:
                    faiss.write_index(faiss_index, "policy_docs.faiss")
                    with open("policy_docs_chunks.json", "w") as f:
                        json.dump(saved_chunks, f)
                    print("FAISS index and chunks saved successfully.")
                else:
                    print("Failed to build or save FAISS index and chunks.")
            else:
                print("No chunks generated from the text elements.")
        else:
            print("No text elements extracted from PDFs.")
