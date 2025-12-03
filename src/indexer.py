import os
# Reorder imports to avoid segfault on macOS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class SemanticIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path="vector_store"):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.dimension = 384  # Dimension for MiniLM-L6-v2
        self.chunk_size = 500 # Characters per chunk
        
        # Storage
        self.chunks = []      # Text content
        self.metadata = []    # Source file names
        self.index = None

    def chunk_text(self, text, source_file):
        """Splits long text into smaller, meaningful chunks with overlap."""
        overlap = 50
        text_chunks = []
        if len(text) <= self.chunk_size:
            text_chunks = [text]
        else:
            for i in range(0, len(text), self.chunk_size - overlap):
                text_chunks.append(text[i:i + self.chunk_size])
        
        # Extract a "title" from the filename
        title = os.path.splitext(source_file)[0].replace('_', ' ').title()
        
        meta = [{"source": source_file, "title": title} for _ in text_chunks]
        return text_chunks, meta

    def save_index(self):
        """Saves the FAISS index and metadata to disk."""
        if not self.index:
            return
        
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
            
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        
        # Save metadata and chunks
        with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
            pickle.dump((self.chunks, self.metadata), f)
        print(f"Index saved to {self.index_path}")

    def load_index(self):
        """Loads the FAISS index and metadata from disk."""
        if not os.path.exists(os.path.join(self.index_path, "index.faiss")):
            print("No saved index found.")
            return False
            
        print("Loading index from disk...")
        self.index = faiss.read_index(os.path.join(self.index_path, "index.faiss"))
        
        with open(os.path.join(self.index_path, "metadata.pkl"), "rb") as f:
            self.chunks, self.metadata = pickle.load(f)
        print(f"Index loaded with {self.index.ntotal} vectors.")
        return True

    def build_index(self, data_folder):
        """Reads files, embeds them, and builds a FAISS index."""
        print("--- Starting Indexing Process ---")
        new_chunks = []
        new_meta = []

        # 1. Process Files
        if not os.path.exists(data_folder):
            print(f"Data folder {data_folder} does not exist.")
            return

        for file in os.listdir(data_folder):
            if file.endswith(".txt"):
                path = os.path.join(data_folder, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks, meta = self.chunk_text(content, file)
                    new_chunks.extend(chunks)
                    new_meta.extend(meta)

        self.chunks = new_chunks
        self.metadata = new_meta
        
        if not self.chunks:
            print("No documents found.")
            return

        # 2. Create Embeddings (Batch processing is faster)
        print(f"Embedding {len(self.chunks)} chunks...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        
        # 3. Initialize FAISS Index (L2 Distance ~ Euclidean)
        # We normalize vectors so L2 distance acts like Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension) # IP = Inner Product
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors.")
        self.save_index()

    def search(self, query, top_k=5):
        """Encodes query and searches FAISS."""
        if not self.index:
            return []

        # Encode and Normalize Query
        query_vec = self.model.encode([query])
        faiss.normalize_L2(query_vec)
        
        # Search
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if not enough neighbors
                meta = self.metadata[idx]
                results.append({
                    "content": self.chunks[idx],
                    "source": meta['source'],
                    "title": meta.get('title', 'Unknown'),
                    "score": float(distances[0][i])
                })
        return results