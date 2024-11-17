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

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure CUDA settings for better performance
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None

# Set page configuration with optimized settings
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

# Custom CSS for dark theme and improved visibility
st.markdown("""
    <style>
    /* Main container and background */
    .main {
        padding: 0;
        max-width: 100%;
        background-color: #0f172a;
    }
    
    /* Banner styling */
    .banner-container {
        background-color: #1e293b;
        padding: 1rem;
        margin-bottom: 2rem;
        border-radius: 0;
        text-align: center;
    }
    
    /* Company card styling */
    .company-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
        color: #f8fafc;
    }
    
    /* Card headers */
    .company-card h3, .company-card h4 {
        color: #f8fafc;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Card text content */
    .company-card p {
        color: #cbd5e1;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Emphasis for important information */
    .company-card strong {
        color: #60a5fa;
        font-weight: 600;
    }
    
    /* Metrics container layout */
    .metrics-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Individual metric cards */
    .metric-card {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 8px;
        flex: 1;
        text-align: center;
        color: #f8fafc;
    }
    
    /* Description text container */
    .description-text {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        color: #cbd5e1;
        max-height: 200px;
        overflow-y: auto;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #2d3748;
        color: #f8fafc;
        border: 1px solid #4b5563;
        border-radius: 6px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: #f8fafc;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

def handle_system_limits():
    """Configure system limits for better performance."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Set file handle limit to minimum of 8192 or hard limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard), hard))
    except Exception as e:
        logger.warning(f"Could not configure system limits: {e}")

def clear_memory():
    """Aggressively clear memory to prevent OOM issues."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if 'embeddings' in st.session_state:
        del st.session_state.embeddings
        gc.collect()

def display_banner():
    """Display the Innovius banner with fallback."""
    try:
        image = Image.open('Innovius Capital Cover.jpeg')
        st.image(image, use_container_width=True)
    except Exception as e:
        logger.error(f"Banner load error: {e}")
        # Fallback banner if image fails to load
        st.markdown("""
            <div class="banner-container">
                <h1 class="banner-text">INNOVIUS</h1>
                <p style="color: #f8fafc;">VENTURE DIFFERENTLY</p>
            </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load and process data with optimized memory management."""
    try:
        data_path = 'data_with_embeddings.pkl'
        
        # Download data if not exists
        if not os.path.exists(data_path):
            with st.status("📥 Downloading database...", expanded=True) as status:
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                status.write("Starting download...")
                gdown.download(url, data_path, quiet=False)
                
                if not os.path.exists(data_path):
                    raise FileNotFoundError("Download failed")
                
                status.update(label="✅ Download complete!", state="complete")

        # Load and process data
        df = pd.read_pickle(data_path)
        
        # Process embeddings with optimized memory usage
        embeddings = np.array(df['Embeddings'].tolist(), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        embeddings_normalized = (embeddings / norms).astype(np.float32)
        
        # Create FAISS index
        dimension = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_normalized)
        
        # Clean up memory
        del embeddings
        clear_memory()
        
        return df, embeddings_normalized, index
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        st.error("Failed to load data. Please refresh the page.")
        return None, None, None

def display_company_details(company, is_query=False):
    """Display a single company's details in a styled card."""
    prefix = "📌" if is_query else "🎯"
    
    metrics_html = f"""
        <div class="metrics-container">
            <div class="metric-card">
                <strong>Type</strong><br>
                {"Query Company" if is_query else "Similar Company"}
            </div>
            <div class="metric-card">
                <strong>Employee Count</strong><br>
                {company['Employee Count']}
            </div>
            {f'''
            <div class="metric-card">
                <strong>Similarity Score</strong><br>
                {company.get('Similarity Score', 0):.2f}
            </div>
            ''' if not is_query else ''}
        </div>
    """
    
    description_html = f"""
        <div class="description-text">
            {company['Combined_Description'][:500]}...
        </div>
    """
    
    st.markdown(f"""
        <div class="company-card">
            <h4>{prefix} {company['Name']}</h4>
            {metrics_html}
            {description_html}
        </div>
    """, unsafe_allow_html=True)

def create_similarity_chart(similar_companies):
    """Create an enhanced visualization for similarity scores."""
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=similar_companies['Similarity Score'],
        y=similar_companies['Name'],
        orientation='h',
        marker=dict(
            color='#3b82f6',
            line=dict(color='#2563eb', width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Similarity Score: %{x:.2f}<extra></extra>"
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title={
            'text': "Similarity Scores",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(color='#f8fafc')
        },
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font=dict(color='#f8fafc'),
        height=min(400, len(similar_companies) * 40 + 100),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#374151',
            title=dict(text="Similarity Score", font=dict(color='#f8fafc'))
        ),
        yaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#374151',
            autorange="reversed"
        )
    )
    
    return fig

def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """Find similar companies using FAISS index."""
    try:
        # Find the query company
        company_mask = df['Name'].str.lower() == company_name.lower()
        if not company_mask.any():
            st.error(f"Company '{company_name}' not found.")
            return None, None
            
        company_index = company_mask.idxmax()
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        
        # Search with padding for better results
        search_n = min(top_n + 5, len(df))
        distances, indices = index.search(query_vector, search_n)
        
        # Filter out the query company
        mask = indices[0] != company_index
        similar_indices = indices[0][mask][:top_n]
        similar_distances = distances[0][mask][:top_n]
        
        # Create results DataFrame
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        
        return similar_companies, company_index
        
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        st.error("Error finding similar companies.")
        return None, None

def main():
    """Main application logic."""
    # Configure system limits
    handle_system_limits()
    
    try:
        # Load data if not already loaded
        if not st.session_state.data_loaded:
            df, embeddings_normalized, index = load_data()
            if df is not None:
                st.session_state.data = df
                st.session_state.embeddings = embeddings_normalized
                st.session_state.index = index
                st.session_state.data_loaded = True
        
        # Display banner and title
        display_banner()
        st.title("🔍 Company Similarity Finder")
        st.markdown("""
            <p style="color: #cbd5e1; font-size: 1.1rem;">
            Discover companies similar to your target using our AI-powered analysis.
            Enter a company name below to explore related companies.
            </p>
        """, unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        with col1:
            company_name = st.text_input(
                "🔎 Enter company name:",
                placeholder="e.g., Microsoft, Apple, Tesla..."
            )
        with col2:
            top_n = st.slider(
                "Number of similar companies:",
                1, 15, 5,
                help="Select how many similar companies to display"
            )
        
        # Process search
        if company_name and st.session_state.data_loaded:
            with st.spinner('Finding similar companies...'):
                similar_companies, company_index = get_similar_companies(
                    st.session_state.data,
                    st.session_state.embeddings,
                    st.session_state.index,
                    company_name,
                    top_n
                )
                
                if similar_companies is not None:
                    # Display query company
                    query_company = st.session_state.data.iloc[company_index]
                    display_company_details(query_company, is_query=True)
                    
                    # Display similarity chart
                    st.markdown("### 📊 Similarity Analysis")
                    fig = create_similarity_chart(similar_companies)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display similar companies
                    st.markdown("### 🎯 Similar Companies")
                    for _, company in similar_companies.iterrows():
                        display_company_details(company)
                    
                    # Export functionality
                    if st.button("📥 Export Results"):
                        csv = similar_companies.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"similar_companies_{company_name}.csv",
                            mime="text/csv"
                        )
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please refresh the page.")
        clear_memory()

if __name__ == "__main