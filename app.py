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

# Initialize session state for loading status and cache control
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Add cache control to prevent infinite loading
if 'download_attempted' not in st.session_state:
    st.session_state.download_attempted = False

# Set page configuration
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# [Previous CSS styles remain the same...]

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

@st.cache_data(show_spinner=True, ttl=3600)  # Cache for 1 hour
def download_data():
    """Download data with proper error handling and user feedback."""
    data_path = 'data_with_embeddings.pkl'
    
    # Check if download was already attempted
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
            st.error(f"Error downloading file: {str(e)}")
            logger.error(f"Download error: {str(e)}")
            return None
    return data_path

@st.cache_data(show_spinner=True, ttl=3600)
def load_data():
    """Load and process data with proper error handling."""
    try:
        data_path = download_data()
        if data_path is None:
            return None, None, None

        with st.spinner('Processing company data...'):
            df = pd.read_pickle(data_path)
            
            # Process embeddings with progress updates
            embeddings = np.array(df['Embeddings'].tolist())
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings_normalized.astype('float32')
            
            dimension = embeddings_normalized.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_normalized)
            
            # Clean up memory
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

def create_similarity_chart(similar_companies):
    """Create an interactive bar chart for similarity scores."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=similar_companies['Similarity Score'],
        y=similar_companies['Name'],
        orientation='h',
        marker=dict(
            color='rgb(37, 99, 235)',
            line=dict(color='rgb(8, 47, 167)', width=1)
        )
    ))
    
    fig.update_layout(
        title="Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis=dict(title=None, autorange="reversed"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    """Main application logic with improved error handling and user feedback."""
    # Display banner
    display_banner()
    
    # Initialize loading state
    if not st.session_state.data_loaded:
        st.info("🚀 Initializing the application...")
    
    # Load data with timeout handling
    try:
        with st.spinner('Loading company database...'):
            df, embeddings_normalized, index = load_data()
    except Exception as e:
        st.error("Loading timeout. Please refresh the page.")
        return
    
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
        with st.spinner('Searching for similar companies...'):
            similar_companies, company_index = get_similar_companies(
                df, embeddings_normalized, index, company_name_input, top_n
            )
            
            if similar_companies is not None:
                # Display query company details
                query_company = df.iloc[company_index]
                st.subheader("📌 Query Company")
                with st.container():
                    st.markdown(f"""
                    <div class="company-card">
                        <h3>{query_company['Name']}</h3>
                        <p><strong>Employee Count:</strong> {query_company['Employee Count']}</p>
                        <p>{query_company['Combined_Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display similarity visualization
                st.subheader("🎯 Similar Companies")
                fig = create_similarity_chart(similar_companies)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display similar companies details
                for _, company in similar_companies.iterrows():
                    st.markdown(f"""
                    <div class="company-card">
                        <h4>{company['Name']}</h4>
                        <p><strong>Similarity Score:</strong> {company['Similarity Score']:.2f}</p>
                        <p><strong>Employee Count:</strong> {company['Employee Count']}</p>
                        <p>{company['Combined_Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add export functionality
                if st.button("📥 Export Results"):
                    csv = similar_companies.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"similar_companies_{company_name_input}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)