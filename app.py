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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for loading status
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Set page configuration
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with loading animation
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0;
        max-width: 100%;
    }
    
    /* Banner styling */
    .banner-container {
        background-color: #0a192f;
        padding: 1rem;
        margin-bottom: 2rem;
        border-radius: 0;
        text-align: center;
    }
    
    .banner-text {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Card styling */
    .company-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        border: 2px solid #e2e8f0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def display_banner():
    """Display the Innovius banner."""
    try:
        image = Image.open('Innovius Capital Cover.jpeg')
        st.image(image, use_column_width=True)
    except Exception as e:
        logger.error(f"Error loading banner: {str(e)}")
        # Fallback banner if image fails to load
        st.markdown("""
            <div class="banner-container">
                <h1 class="banner-text">INNOVIUS</h1>
                <p style="color: white;">VENTURE DIFFERENTLY</p>
            </div>
        """, unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def download_data():
    """Download data with proper error handling and user feedback."""
    data_path = 'data_with_embeddings.pkl'
    if not os.path.exists(data_path):
        try:
            with st.status("📥 Downloading company database...", expanded=True) as status:
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                
                # Show progress message
                status.write("Starting download...")
                
                # Download with progress tracking
                gdown.download(url, data_path, quiet=False)
                
                if os.path.exists(data_path):
                    status.update(label="✅ Download complete!", state="complete")
                    time.sleep(1)  # Give users time to see the completion message
                else:
                    status.update(label="❌ Download failed!", state="error")
                    return None
        except Exception as e:
            st.error(f"Error downloading file: {str(e)}")
            logger.error(f"Download error: {str(e)}")
            return None
    return data_path

@st.cache_data(show_spinner=True)
def load_data():
    """Load and process data with proper error handling."""
    try:
        data_path = download_data()
        if data_path is None:
            return None, None, None

        with st.spinner('Processing company data...'):
            # Load DataFrame
            df = pd.read_pickle(data_path)
            
            # Process embeddings
            embeddings = np.array(df['Embeddings'].tolist())
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings_normalized.astype('float32')
            
            # Create FAISS index
            dimension = embeddings_normalized.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_normalized)
            
            # Clean up
            del embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            st.session_state.data_loaded = True
            return df, embeddings_normalized, index
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading error: {str(e)}")
        return None, None, None

def main():
    """Main application logic with improved error handling and user feedback."""
    # Display banner
    display_banner()
    
    # Initialize loading state
    if not st.session_state.data_loaded:
        st.info("🚀 Initializing the application...")
    
    # Load data
    df, embeddings_normalized, index = load_data()
    
    if df is None:
        st.error("Unable to load company database. Please try refreshing the page.")
        st.stop()
    
    # Main app content
    st.title("🔍 Company Similarity Finder")
    st.markdown("""
    Discover companies similar to your target using our AI-powered analysis engine.
    Simply enter a company name below to explore related companies and understand their relationships.
    """)
    
    # Create two columns for inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_name_input = st.text_input(
            "🔎 Enter a company name:",
            placeholder="e.g., Microsoft, Apple, Tesla..."
        )
    
    with col2:
        top_n = st.slider(
            "Number of similar companies:",
            min_value=1,
            max_value=20,
            value=5
        )
    
    # Process search
    if company_name_input:
        try:
            similar_companies, company_index = get_similar_companies(
                df, embeddings_normalized, index, company_name_input, top_n
            )
            
            if similar_companies is not None:
                # Display results...
                # [Rest of your display code remains the same]
                pass
                
        except Exception as e:
            st.error(f"Error processing search: {str(e)}")
            logger.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)