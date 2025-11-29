# SFU IT Chatbot w/ Ollama
This chatbot runs locally on your computer! Yay! This RAG Chatbot uses Gemma3-4b LLM and all-MiniLM-L6-v2 for vector embedding.

To run the app, you need Ollama installed which can be found here:
https://ollama.com/download

Then, you need to download Gemma3-4b from your terminal:
`ollama pull gemma3:4b`

Now, first set up virtual environment:
`python3 -m venv venv`
`source venv/bin/activate`

Then install requirements:
`pip install -r requirements.txt`

Now run the app (Note: This uses about 3-4 GB of RAM):
`python app.py`