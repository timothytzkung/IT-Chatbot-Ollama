import json
import numpy as np
import pandas as pd
import gradio as gr
import ollama  # pip install ollama
from sentence_transformers import SentenceTransformer
import os

# --- Configuration ---
print("Loading RAG system on your device...")

# Load Knowledge base
FILE_PATH = "data.json"
if not os.path.exists(FILE_PATH):
    # Dummy data for testing if you don't have the file yet
    print(f"Warning: {FILE_PATH} not found. Creating dummy data.")
    data = [{"clean_body_text": "To reset your password, visit password.sfu.ca and click 'Forgot Password'."}]
else:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

documents = [item["clean_body_text"] for item in data]

# Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

df = pd.DataFrame({
    "Document": documents,
    "Embedding": list(embeddings),
})

# --- RAG Logic ---
def retrieve_with_pandas(query: str, top_k: int = 2):
    query_embedding = embedding_model.encode([query])[0]

    def cosine_sim(x):
        x = np.array(x)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(x)
        if norm_product == 0: return 0.0
        return float(np.dot(query_embedding, x) / norm_product)

    df["Similarity"] = df["Embedding"].apply(cosine_sim)
    return df.sort_values(by="Similarity", ascending=False).head(top_k)

def generate_with_rag(query, history):
    # 1. Retrieve
    results = retrieve_with_pandas(query)
    context_str = "\n\n---\n\n".join(results["Document"].tolist())

    # 2. Construct Prompt
    # We send this to Ollama, which is running efficiently in the background
      # Build a clean prompt
    prompt_content = f"""You are an IT helpdesk assistant.
If the user asked a question, answer the user's question with detailed step by step instructions: consider all the articles below.
If the user asked a question and the answer is not in the contexts, say you don't know and suggest contacting SFU IT.
If the user DID NOT ask a question, be friendly and ask how you can help them.


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
    title="SFU IT Chatbot (Locally Hosted)",
    description="Running locally on your device~!",
)

if __name__ == "__main__":
    # share=True creates a public link (e.g., https://12345.gradio.live) valid for 72 hours
    demo.launch(share=True)