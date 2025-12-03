import os
# --- MAC OS FIX: MUST BE AT THE VERY TOP ---
# This allows PyTorch and FAISS to run simultaneously on Mac without crashing
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------------------------

import streamlit as st
import sys

# Fix import path
# Use absolute paths to be robust against CWD changes
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
data_path = os.path.join(project_root, "data")
index_path = os.path.join(project_root, "vector_store")

if src_path not in sys.path:
    sys.path.append(src_path)

# Only import the backend AFTER setting the fix above
from indexer import SemanticIndexer 

# ... (Rest of your code remains the same)

# ... rest of your code ...

# Setup Page
st.set_page_config(page_title="Library Neural Search", page_icon="🏛️", layout="wide")

# Custom CSS for a clean, neutral look
st.markdown("""
<style>
    /* Card Container */
    .book-card {
        background-color: transparent;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* Typography */
    .book-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .book-meta {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-bottom: 10px;
    }
    
    .book-content {
        font-size: 1rem;
        line-height: 1.5;
        opacity: 0.9;
        padding-left: 15px;
        border-left: 3px solid #FF4B4B; /* Streamlit Red for accent */
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Digital Library Search")
st.markdown("Explore your document collection with AI-powered semantic search.")
st.divider()

# ---------------------------
# State Management
# ---------------------------
if 'engine' not in st.session_state:
    with st.spinner("Initializing Library System..."):
        st.session_state.engine = SemanticIndexer(index_path=index_path)
        # Try to load existing index first
        if not st.session_state.engine.load_index():
            # If no index, build it if data exists
            if os.path.exists(data_path) and os.listdir(data_path):
                 st.session_state.engine.build_index(data_path)

engine = st.session_state.engine

# ---------------------------
# Sidebar: Librarian Desk
# ---------------------------
with st.sidebar:
    st.markdown("### 📚 Librarian's Desk")
    
    # Library Stats
    if engine.index:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Books", len(set(m['source'] for m in engine.metadata)))
        with col_b:
            st.metric("Passages", engine.index.ntotal)
    
    st.divider()
    st.markdown("#### 📥 Add to Collection")
    uploaded_files = st.file_uploader("Upload Text Files", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Catalog & Index", type="primary"):
            with st.spinner("Reading, Chunking, and Vectorizing..."):
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                    
                for file in uploaded_files:
                    save_path = os.path.join(data_path, file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.read())
                
                # Rebuild Index
                engine.build_index(data_path)
                st.success(f"Added {len(uploaded_files)} new books to the shelves!")
                st.rerun()

# ---------------------------
# Main Search UI
# ---------------------------
# Centered Search Layout
col_spacer_l, col_search, col_spacer_r = st.columns([1, 2, 1])

with col_search:
    query = st.text_input("", placeholder="🔍 Search the archives... (e.g., 'History of AI')")

if query:
    if not engine.index or engine.index.ntotal == 0:
        st.warning("The library is empty. Please upload some books first.")
    else:
        with st.spinner("Searching the archives..."):
            results = engine.search(query, top_k=5)
        
        st.markdown(f"### 🎯 Results for: *'{query}'*")
        
        if not results:
            st.info("No matching records found.")
        
        for res in results:
            # Calculate confidence percentage
            score = res['score']
            confidence = min(max(score * 100, 0), 100)
            
            # Create a "Book Card"
            st.markdown(f"""
            <div class="book-card">
                <div class="book-title">📄 {res.get('title', 'Unknown Title')}</div>
                <div class="book-meta">Source: {res['source']}</div>
                <div class="book-content">
                    "{res['content']}..."
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Use native Streamlit components for the bar to ensure theme compatibility
            st.caption(f"Relevance Score: {score:.2f}")
            st.progress(int(confidence))
            st.markdown("---")
