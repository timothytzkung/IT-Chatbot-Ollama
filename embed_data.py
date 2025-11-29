import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# --- Configuration ---
FILE_PATH = "data.json" # Your original source file
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_FILE = "rag_data.parquet"

def create_and_save_embeddings():
    print(f"1. Loading source data from {FILE_PATH}...")
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found.")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [item["clean_body_text"] for item in data]
    
    print(f"2. Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # The M1 chip will use the Apple MPS backend for faster processing
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("3. Generating embeddings for all documents...")
    # Generate the embeddings
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    
    # Create the DataFrame
    df = pd.DataFrame({
        "Document": documents,
        # Convert the NumPy array to a list of floats so it can be saved/loaded easily
        "Embedding": embeddings.tolist(), 
    })
    
    print(f"4. Saving DataFrame to {OUTPUT_FILE}...")
    # Save the DataFrame to a compressed Parquet file
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"Yay! Success! Embeddings saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    create_and_save_embeddings()