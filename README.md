# SFU IT Chatbot w/ Ollama
This chatbot runs locally on your computer! Yay! This RAG Chatbot uses Gemma3-4b LLM and all-MiniLM-L6-v2 for vector embedding.

To run the app, you need Ollama installed which can be found here: <br>
https://ollama.com/download<br>

Then, you need to download Gemma3-4b from your terminal:<br>
`ollama pull gemma3:4b`<br>

Now, first set up virtual environment:<br>
`python3 -m venv venv`<br>
`source venv/bin/activate`<br>

Then install requirements:<br>
`pip install -r requirements.txt`<br>

Now run the app (Note: This uses about 3-4 GB of RAM):<br>
`python app.py`<br>