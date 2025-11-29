import numpy as np
import pandas as pd
import gradio as gr
import ollama 
from sentence_transformers import SentenceTransformer
import os
import torch 

# --- Configuration ---
EMBEDDINGS_FILE = "rag_data.parquet"

# --- Global Data Loading (Run Once) ---
print("Loading RAG system...")

# 1. Check for and load pre-computed embeddings
if os.path.exists(EMBEDDINGS_FILE):
    print(f"1. Loading pre-computed embeddings from {EMBEDDINGS_FILE}...")
    # Load the DataFrame
    df = pd.read_parquet(EMBEDDINGS_FILE)
    # Convert the list of floats back into a NumPy array for fast calculation
    df['Embedding'] = df['Embedding'].apply(np.array)
else:
    # If the file is missing, we must exit or print an error
    print(f"Error! Ran into an oopsie: Embeddings file '{EMBEDDINGS_FILE}' not found.")
    print("Please run the 'embed_data.py' script first to generate the embeddings.")
    df = None 

# 2. Load the embedding model (only used for embedding the user's new query)
print("2. Loading sentence transformer model for query embedding...")
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Explicitly move the model to the M1's GPU (MPS) for speed
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model.to(device)
    print(f"   Model loaded on device: {device}")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

# --- RAG Logic ---

if df is not None and embedding_model is not None:
    
    def retrieve_with_pandas(query: str, top_k: int = 2):
        # Embed the new query
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    
        def cosine_sim(x):
            x = np.array(x)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(x)
            if norm_product == 0: return 0.0
            return float(np.dot(query_embedding, x) / norm_product)
    
        # Calculate similarity against all pre-computed document embeddings
        # This is fast because it only involves NumPy array comparisons
        df["Similarity"] = df["Embedding"].apply(cosine_sim)
        return df.sort_values(by="Similarity", ascending=False).head(top_k)
    
    def generate_with_rag(query, history):
        # 1. Retrieve
        results = retrieve_with_pandas(query)
        context_str = "\n\n---\n\n".join(results["Document"].tolist())
    
        # 2. Construct Prompt
        prompt_content = f"""You are an IT helpdesk assistant.
If the user did not ask a question, be friendly and ask how you can help them.
If the user asked a question, answer the user's question with detailed step by step instructions: consider all the articles below.
If the user asked a question and the answer is not in the contexts, say you don't know and suggest contacting SFU IT.


Question:
{query}

-- Start of Articles --
{context_str}

-- End of Articles --

Answer:"""
    
        # 3. Call Local Model (Ollama)
        response_stream = ollama.chat(
            model='gemma3:4b', 
            messages=[{'role': 'user', 'content': prompt_content}],
            stream=True
        )
        
        # 4. Stream response back to UI
        partial_message = ""
        for chunk in response_stream:
            partial_message += chunk['message']['content']
            yield partial_message

    # --- Interface ---
    demo = gr.ChatInterface(
        fn=generate_with_rag,
        title="SFU IT Chatbot (Optimized Mac M1 Host)",
        description="Embeddings loaded from cache for fast startup.",
    )
    
    if __name__ == "__main__":
        demo.launch(share=True)
else:
    # Fallback interface if setup failed
    def setup_error(message, history):
        return "Chatbot setup failed. Please check the terminal for setup instructions."
    
    error_demo = gr.ChatInterface(
        fn=setup_error,
        title="Chatbot Setup Error",
        description="The system failed to load data. Ensure 'embed_data.py' was run.",
    )
    if __name__ == "__main__":
        error_demo.launch()