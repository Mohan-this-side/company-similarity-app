import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import altair as alt
import os
import gdown
import gc
import torch
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import logging
import time
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory optimization settings
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None

# Set page configuration
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

def handle_system_limits():
    """Configure system limits for better performance."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard), hard))
    except Exception as e:
        logger.warning(f"Could not configure system limits: {e}")

def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if 'embeddings' in st.session_state:
        del st.session_state.embeddings
        gc.collect()

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load data with improved memory management."""
    try:
        data_path = 'data_with_embeddings.pkl'
        
        if not os.path.exists(data_path):
            with st.status("📥 Downloading database...", expanded=True) as status:
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                status.write("Starting download...")
                gdown.download(url, data_path, quiet=False)
                
                if not os.path.exists(data_path):
                    raise FileNotFoundError("Download failed")
                
                status.update(label="✅ Download complete!", state="complete")

        # Load data in chunks
        df = pd.read_pickle(data_path)
        
        # Process embeddings efficiently
        embeddings = np.array(df['Embeddings'].tolist(), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_normalized = (embeddings / norms).astype(np.float32)
        
        # Build FAISS index
        dimension = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_normalized)
        
        # Clear memory
        del embeddings
        clear_memory()
        
        return df, embeddings_normalized, index
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        st.error("Failed to load data. Please refresh the page.")
        return None, None, None

def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """Find similar companies with optimized processing."""
    try:
        company_mask = df['Name'].str.lower() == company_name.lower()
        if not company_mask.any():
            st.error(f"Company '{company_name}' not found.")
            return None, None
            
        company_index = company_mask.idxmax()
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        
        # Search with padding
        search_n = min(top_n + 5, len(df))
        distances, indices = index.search(query_vector, search_n)
        
        # Filter results
        mask = indices[0] != company_index
        similar_indices = indices[0][mask][:top_n]
        similar_distances = distances[0][mask][:top_n]
        
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        
        return similar_companies, company_index
        
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        st.error("Error finding similar companies.")
        return None, None

def display_results(similar_companies, query_company):
    """Display results in an optimized way."""
    # Display query company
    st.markdown("### 📌 Query Company")
    st.markdown(f"""
        <div class="company-card">
            <h4>{query_company['Name']}</h4>
            <p><strong>Employees:</strong> {query_company['Employee Count']}</p>
            <p>{query_company['Combined_Description'][:500]}...</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=similar_companies['Similarity Score'],
        y=similar_companies['Name'],
        orientation='h',
        marker_color='rgb(37, 99, 235)'
    ))
    fig.update_layout(
        title="Similarity Scores",
        height=min(400, len(similar_companies) * 30 + 100),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display similar companies in batches
    for idx in range(0, len(similar_companies), 2):
        cols = st.columns(2)
        for i, col in enumerate(cols):
            if idx + i < len(similar_companies):
                company = similar_companies.iloc[idx + i]
                with col:
                    st.markdown(f"""
                        <div class="company-card">
                            <h4>{company['Name']}</h4>
                            <p><strong>Similarity:</strong> {company['Similarity Score']:.2f}</p>
                            <p><strong>Employees:</strong> {company['Employee Count']}</p>
                            <p>{company['Combined_Description'][:300]}...</p>
                        </div>
                    """, unsafe_allow_html=True)

def main():
    """Main application with optimized performance."""
    handle_system_limits()
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        df, embeddings_normalized, index = load_data()
        if df is not None:
            st.session_state.data = df
            st.session_state.embeddings = embeddings_normalized
            st.session_state.index = index
            st.session_state.data_loaded = True
    
    try:
        display_banner()
        
        st.title("🔍 Company Similarity Finder")
        st.markdown("""
            Find companies similar to your target using our AI-powered analysis.
            Enter a company name below to explore related companies.
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            company_name = st.text_input("🔎 Enter company name:", placeholder="e.g., Microsoft")
        with col2:
            top_n = st.slider("Number of similar companies:", 1, 15, 5)
        
        if company_name and st.session_state.data_loaded:
            similar_companies, company_index = get_similar_companies(
                st.session_state.data,
                st.session_state.embeddings,
                st.session_state.index,
                company_name,
                top_n
            )
            
            if similar_companies is not None:
                display_results(similar_companies, st.session_state.data.iloc[company_index])
                
                # Add export option
                if st.button("📥 Export Results"):
                    csv = similar_companies.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"similar_companies_{company_name}.csv",
                        "text/csv"
                    )
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please refresh the page.")
        clear_memory()

if __name__ == "__main__":
    main()