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
import warnings

# Suppress warnings and config watchdog
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None

# Configure page
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS from file to reduce memory usage
def load_css():
    st.markdown("""
        <style>
        .main {padding: 0; max-width: 100%; background-color: #0f172a;}
        .company-card {
            background-color: #1e293b;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
            border-left: 4px solid #3b82f6;
            color: #f8fafc;
        }
        .metric-card {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 8px;
            flex: 1;
            text-align: center;
            color: #f8fafc;
        }
        .stButton > button {
            background-color: #3b82f6;
            color: #f8fafc;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def init_resources():
    """Initialize CUDA and other resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

@st.cache_data(ttl=3600, show_spinner=False)
def download_data():
    """Download data file if not present."""
    try:
        data_path = 'data_with_embeddings.pkl'
        if not os.path.exists(data_path):
            with st.status("📥 Downloading database...", expanded=True) as status:
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                status.write("Starting download...")
                gdown.download(url, data_path, quiet=False)
        return data_path
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load and process data with optimized memory usage."""
    try:
        data_path = download_data()
        if not data_path:
            return None, None, None

        # Load data in chunks
        df = pd.read_pickle(data_path)
        
        # Process embeddings
        embeddings = np.array(df['Embeddings'].tolist(), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_normalized = (embeddings / norms).astype(np.float32)
        
        # Create index
        dimension = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_normalized)
        
        del embeddings, norms
        gc.collect()
        
        return df, embeddings_normalized, index
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return None, None, None

def find_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """Find similar companies with optimized search."""
    try:
        company_mask = df['Name'].str.lower() == company_name.lower()
        if not company_mask.any():
            return None, None
            
        company_index = company_mask.idxmax()
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        
        # Search with minimal padding
        search_n = min(top_n + 2, len(df))
        distances, indices = index.search(query_vector, search_n)
        
        # Filter results
        mask = indices[0] != company_index
        similar_indices = indices[0][mask][:top_n]
        similar_distances = distances[0][mask][:top_n]
        
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        
        return similar_companies, company_index
    except Exception as e:
        logger.error(f"Search error: {e}")
        return None, None

def display_company(company, is_query=False):
    """Display company information efficiently."""
    prefix = "📌" if is_query else "🎯"
    st.markdown(f"""
        <div class="company-card">
            <h4>{prefix} {company['Name']}</h4>
            <div style="display:flex;gap:1rem;margin-bottom:1rem">
                <div class="metric-card">
                    <strong>{'Query Company' if is_query else 'Similar Company'}</strong>
                </div>
                <div class="metric-card">
                    <strong>Employees:</strong> {company['Employee Count']}
                </div>
                {f'''<div class="metric-card">
                    <strong>Similarity:</strong> {company['Similarity Score']:.2f}
                </div>''' if not is_query else ''}
            </div>
            <p>{company['Combined_Description'][:300]}...</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    """Main application with optimized performance."""
    init_resources()
    load_css()
    
    try:
        # Load data
        if not st.session_state.data_loaded:
            df, embeddings_normalized, index = load_data()
            if df is not None:
                st.session_state.data = df
                st.session_state.embeddings = embeddings_normalized
                st.session_state.index = index
                st.session_state.data_loaded = True
        
        st.title("🔍 Company Similarity Finder")
        st.markdown(
            "Find companies similar to your target using our AI-powered analysis."
        )
        
        # Input section
        col1, col2 = st.columns([2, 1])
        with col1:
            company_name = st.text_input("🔎 Company name:", placeholder="e.g., Microsoft")
        with col2:
            top_n = st.slider("Similar companies to show:", 1, 10, 5)
        
        if company_name and st.session_state.data_loaded:
            similar_companies, company_index = find_similar_companies(
                st.session_state.data,
                st.session_state.embeddings,
                st.session_state.index,
                company_name,
                top_n
            )
            
            if similar_companies is not None:
                # Display companies
                query_company = st.session_state.data.iloc[company_index]
                display_company(query_company, is_query=True)
                
                for _, company in similar_companies.iterrows():
                    display_company(company)
                
                # Export option
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

if __name__ == "__main__":
    main()