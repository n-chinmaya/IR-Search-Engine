# IR Search Engine

A semantic search engine that uses `sentence-transformers` (MiniLM) for generating embeddings and `FAISS` for efficient vector similarity search. It includes a Streamlit-based web interface for uploading documents and searching through them.

## Features

- **Semantic Search:** Finds relevant documents based on meaning, not just keyword matching.
- **Document Indexing:** Chunks and indexes text files (`.txt`).
- **Web Interface:** User-friendly interface built with Streamlit (`app/app.py`).
- **Vector Store:** Persists embeddings and metadata using FAISS and Pickle (`vector_store/`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/n-chinmaya/IR-Search-Engine.git
   cd IR-Search-Engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app/app.py
   ```

2. Navigate to the "Librarian's Desk" in the sidebar to upload `.txt` files.

3. Click "Catalog & Index" to process the files.

4. Use the search bar to query the indexed documents.

## Project Structure

- `src/indexer.py`: Core logic for chunking, embedding, and indexing.
- `app/app.py`: Streamlit frontend application.
- `data/`: Folder for raw text files.
- `vector_store/`: Folder for the FAISS index and metadata.
