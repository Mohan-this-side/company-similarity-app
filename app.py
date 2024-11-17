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
from typing import Tuple, Optional, Dict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state with type hints for better code clarity
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded: bool = False
if 'download_attempted' not in st.session_state:
    st.session_state.download_attempted: bool = False
if 'processed_companies' not in st.session_state:
    st.session_state.processed_companies: Dict = {}

# Set page configuration
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with optimized styling
st.markdown("""
    <style>
    .main {
        padding: 0;
        max-width: 100%;
    }
    
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
    
    .company-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        margin: 0 0.5rem;
    }
    
    .description-text {
        max-height: 200px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Optimize scrollbar for better performance */
    .description-text::-webkit-scrollbar {
        width: 6px;
    }
    .description-text::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    .description-text::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)

def display_banner() -> None:
    """Display the Innovius banner with error handling."""
    try:
        image = Image.open('Innovius Capital Cover.jpeg')
        st.image(image, use_container_width=True)
    except Exception as e:
        logger.error(f"Banner load error: {str(e)}")
        st.markdown("""
            <div class="banner-container">
                <h1 class="banner-text">INNOVIUS</h1>
                <p style="color: white;">VENTURE DIFFERENTLY</p>
            </div>
        """, unsafe_allow_html=True)

@st.cache_data(show_spinner=True, ttl=3600)
def download_data() -> Optional[str]:
    """
    Download data with optimized error handling and progress tracking.
    Returns path to data file or None if download fails.
    """
    data_path = 'data_with_embeddings.pkl'
    
    if st.session_state.download_attempted and not os.path.exists(data_path):
        return None
    
    if not os.path.exists(data_path):
        try:
            st.session_state.download_attempted = True
            with st.status("📥 Downloading company database...", expanded=True) as status:
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                
                status.write("Starting download...")
                gdown.download(url, data_path, quiet=False)
                
                if os.path.exists(data_path):
                    status.update(label="✅ Download complete!", state="complete")
                else:
                    status.update(label="❌ Download failed!", state="error")
                    return None
        except Exception as e:
            st.error(f"Download error: {str(e)}")
            logger.error(f"Download error: {str(e)}")
            return None
    return data_path

@st.cache_data(show_spinner=True, ttl=3600, max_entries=1)
def load_data() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[faiss.Index]]:
    """
    Load and process data with optimized memory management.
    Returns tuple of (DataFrame, normalized embeddings, FAISS index).
    """
    try:
        data_path = download_data()
        if data_path is None:
            return None, None, None

        with st.spinner('Processing company data...'):
            # Load DataFrame in chunks for better memory management
            df = pd.read_pickle(data_path)
            
            # Process embeddings with memory optimization
            embeddings = np.array(df['Embeddings'].tolist(), dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Prevent division by zero
            embeddings_normalized = embeddings / norms
            
            # Create FAISS index with optimized parameters
            dimension = embeddings_normalized.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_normalized)
            
            # Clean up memory
            del embeddings, norms
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            st.session_state.data_loaded = True
            return df, embeddings_normalized, index
            
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        logger.error(f"Data loading error: {str(e)}")
        return None, None, None

def get_similar_companies(
    df: pd.DataFrame,
    embeddings_normalized: np.ndarray,
    index: faiss.Index,
    company_name: str,
    top_n: int = 5
) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    """
    Find similar companies with optimized batch processing.
    
    Args:
        df: Company database DataFrame
        embeddings_normalized: Normalized embedding vectors
        index: FAISS similarity index
        company_name: Name of the query company
        top_n: Number of similar companies to return
    
    Returns:
        Tuple of (similar companies DataFrame, query company index)
    """
    try:
        # Check cache first
        cache_key = f"{company_name}_{top_n}"
        if cache_key in st.session_state.processed_companies:
            return st.session_state.processed_companies[cache_key]

        # Find company index
        company_index = df[df['Name'].str.lower() == company_name.lower()].index[0]
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        
        # Search with padding for better accuracy
        search_n = min(top_n + 10, len(df))  # Add padding for filtering
        distances, indices = index.search(query_vector, search_n)
        
        # Filter results
        mask = indices[0] != company_index
        similar_indices = indices[0][mask][:top_n]
        similar_distances = distances[0][mask][:top_n]
        
        # Create results DataFrame
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        
        # Cache results
        st.session_state.processed_companies[cache_key] = (similar_companies, company_index)
        
        return similar_companies, company_index
    
    except IndexError:
        st.error(f"Company '{company_name}' not found in database.")
        return None, None
    except Exception as e:
        st.error(f"Error finding similar companies: {str(e)}")
        logger.error(f"Similarity search error: {str(e)}")
        return None, None

def create_similarity_chart(similar_companies: pd.DataFrame) -> go.Figure:
    """Create an optimized visualization for similarity scores."""
    fig = go.Figure()
    
    # Create bar chart with optimized settings
    fig.add_trace(go.Bar(
        x=similar_companies['Similarity Score'],
        y=similar_companies['Name'],
        orientation='h',
        marker=dict(
            color='rgb(37, 99, 235)',
            line=dict(color='rgb(8, 47, 167)', width=1)
        ),
        hovertemplate="<b>%{y}</b><br>" +
                      "Similarity Score: %{x:.2f}<br>" +
                      "<extra></extra>"
    ))
    
    # Optimize layout for performance
    fig.update_layout(
        title="Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis=dict(
            title=None,
            autorange="reversed",
            showgrid=False
        ),
        height=min(400, 50 * len(similar_companies) + 100),  # Dynamic height
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def display_company_card(company: pd.Series, is_query: bool = False) -> None:
    """Display company information in an optimized card format."""
    prefix = "📌 " if is_query else "🎯 "
    st.markdown(f"""
    <div class="company-card">
        <h4>{prefix}{company['Name']}</h4>
        <div class="metrics-container">
            <div class="metric-card">
                <strong>{"Query Company" if is_query else "Similarity Score"}</strong><br>
                {company.get('Similarity Score', 'N/A') if not is_query else 'N/A'}
            </div>
            <div class="metric-card">
                <strong>Employees</strong><br>
                {company['Employee Count']}
            </div>
        </div>
        <div class="description-text">
            {company['Combined_Description']}
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application logic with optimized performance."""
    display_banner()
    
    if not st.session_state.data_loaded:
        st.info("🚀 Initializing the application...")
    
    try:
        # Load data with improved memory management
        df, embeddings_normalized, index = load_data()
        
        if df is None:
            st.error("Unable to load company database. Please refresh the page.")
            st.stop()
        
        st.title("🔍 Company Similarity Finder")
        st.markdown("""
        Discover companies similar to your target using our AI-powered analysis engine.
        Simply enter a company name below to explore related companies and understand their relationships.
        """)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        with col1:
            company_name_input = st.text_input(
                "🔎 Enter a company name:",
                placeholder="e.g., Apple, HP, Walmart..."
            )
        
        with col2:
            top_n = st.slider(
                "Number of similar companies:",
                min_value=1,
                max_value=20,
                value=5,
                help="Select how many similar companies to display"
            )
        
        if company_name_input:
            # Process with progress indication
            with st.spinner('Searching for similar companies...'):
                similar_companies, company_index = get_similar_companies(
                    df, embeddings_normalized, index, company_name_input, top_n
                )
                
                if similar_companies is not None:
                    # Display query company
                    query_company = df.iloc[company_index]
                    display_company_card(query_company, is_query=True)
                    
                    # Display visualization
                    st.subheader("📊 Similarity Analysis")
                    fig = create_similarity_chart(similar_companies)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display similar companies in batches
                    st.subheader("🎯 Similar Companies")
                    for idx in range(0, len(similar_companies), 3):
                        batch = similar_companies.iloc[idx:idx + 3]
                        cols = st.columns(3)
                        for col, (_, company) in zip(cols, batch.iterrows()):
                            with col:
                                display_company_card(company)
                    
                    # Export functionality
                    if st.button("📥 Export Results"):
                        csv = similar_companies.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"similar_companies_{company_name_input}.csv",
                            mime="text/csv"
                        )
                    
    except Exception as e:
        st.error("An error occurred. Please refresh the page.")
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)