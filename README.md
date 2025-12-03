# IR Search Engine / Digital Library Search

A semantic search engine that uses AI-powered embeddings to search through your document collection. Built with Python, Streamlit, FAISS, and Sentence Transformers.

## Description

This Digital Library Search tool enables intelligent, semantic search across text documents. Unlike traditional keyword-based search, it understands the meaning and context of your queries to find the most relevant passages in your document collection. The application chunks documents, generates vector embeddings using the `all-MiniLM-L6-v2` model, and indexes them with FAISS for fast similarity search.

## Features

- **Semantic Search**: Find relevant content based on meaning, not just keywords, using state-of-the-art sentence embeddings
- **Document Uploading**: Easily upload and process text files through the web interface
- **Interactive UI**: Clean, modern Streamlit interface with real-time search results and relevance scores
- **Persistent Index**: Document embeddings are saved to disk, so you don't need to re-index on each restart

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

2. Using the web interface:
   - **Upload Documents**: Use the sidebar ("Librarian's Desk") to upload `.txt` files. Click "Catalog & Index" to process and index the documents.
   - **Search**: Enter your query in the search bar to find relevant passages. Results are displayed with relevance scores.
   - **View Stats**: The sidebar shows the number of indexed books and passages.

## Project Structure

```
IR-Search-Engine/
├── app/
│   └── app.py           # Streamlit web application
├── src/
│   └── indexer.py       # SemanticIndexer class for embedding and FAISS indexing
├── data/                # Directory for storing uploaded text documents
├── vector_store/        # Directory for persisted FAISS index and metadata
└── requirements.txt     # Project dependencies
```
