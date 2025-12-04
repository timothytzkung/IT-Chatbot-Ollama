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
FILE_PATH = "data.jsonl"
PRELOAD_FILE_PATH = "preload-data"

# File path readings
if not os.path.exists(FILE_PATH):
    # Dummy data for testing if you don't have the file yet
    print(f"Warning: {FILE_PATH} not found. Creating dummy data.")
    data = [{"text": "To reset your password, visit password.sfu.ca and click 'Forgot Password'."}]
elif os.path.exists(PRELOAD_FILE_PATH):
    print(f"Found Preloaded Data! Using {PRELOAD_FILE_PATH}...")
    with open(PRELOAD_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        print(f"No Preloaded Data Found. Using {FILE_PATH}...")
        data = pd.read_json(path_or_buf=f, lines=True)

# Writes in data embedding
if not os.path.exists(PRELOAD_FILE_PATH):
    documents = list(data["text"])
    print(f"Creating {PRELOAD_FILE_PATH}...")
    with open("preload-data", "w") as fp:
        json.dump(documents, fp)
else:
    documents = data

# Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

df = pd.DataFrame({
    "Document": documents,
    "Embedding": list(embeddings),
})

# --- RAG Logic ---
def retrieve_with_pandas(query: str, top_k: int = 10):
    query_embedding = embedding_model.encode([query])[0]

    def cosine_sim(x):
        x = np.array(x)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(x)
        if norm_product == 0: return 0.0
        return float(np.dot(query_embedding, x) / norm_product)

    df["Similarity"] = df["Embedding"].apply(cosine_sim)
    return df.sort_values(by="Similarity", ascending=False).head(top_k)


def clean_query_with_llm(query, history):
    prompt_content = f"""    
    Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
    You have access to SFU IT Knowledge Base index with 100's of chunked documents.
    Generate a search query based on the conversation and the new question.
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Do not include any special characters like '+'.
    If the question is not in English, translate the question to English before generating the search query.
    If you cannot generate a search query, return just the number 0.

    History:
    {history}

    Question:
    {query}

    Search Query:
    """

    response_stream = ollama.chat(
        model='gemma3:4b', 
        messages=[{'role': 'user', 'content': prompt_content}],
        stream=True
    )
    return response_stream


def generate_with_rag(query, history):
    # Retrieve
    search_query = clean_query_with_llm(query)
    print(search_query)
    results = retrieve_with_pandas(search_query)

    context_str = "\n\n---\n\n".join(results["Document"].tolist())

    # Construct Prompt
    # We send this to Ollama, which is running efficiently in the background
      # Build a clean prompt
    prompt_content = f"""
    You are an SFU IT helpdesk chatbot.
    Below is a history of the conversation so far, a new question asked by the user, and related article chunks to the user question.
    If the user asked a question, answer the user's question with detailed step by step instructions: consider all the articles below.
    If the user asked a question and the answer is not in the contexts, say you don't know and suggest contacting SFU IT.
    If the user DID NOT ask a question, be friendly and ask how you can help them.

History:
{history}

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