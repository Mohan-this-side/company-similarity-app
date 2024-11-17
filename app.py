# Required imports
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

# Configure basic logging to track errors and info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables to persist data between reruns
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None

# Configure the Streamlit page
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data(show_spinner=False)
def load_data():
    """
    Load and process the company data from pickle file.
    Uses caching to prevent reloading on each rerun.
    Returns:
        tuple: (DataFrame, normalized embeddings, FAISS index) or (None, None, None) on error
    """
    try:
        data_path = 'data_with_embeddings.pkl'
        
        # Download data if not present
        if not os.path.exists(data_path):
            with st.status("📥 Downloading company database...", expanded=True) as status:
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                status.write("Starting download...")
                gdown.download(url, data_path, quiet=False)
                
                if not os.path.exists(data_path):
                    raise FileNotFoundError("Download failed")
                
                status.update(label="✅ Download complete!", state="complete")

        # Load the DataFrame
        df = pd.read_pickle(data_path)
        
        # Convert embeddings list to numpy array and normalize
        embeddings = np.array(df['Embeddings'].tolist(), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        embeddings_normalized = (embeddings / norms).astype(np.float32)
        
        # Create FAISS index for similarity search
        dimension = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_normalized)
        
        # Clean up memory
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return df, embeddings_normalized, index
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        st.error("Failed to load data. Please refresh the page.")
        return None, None, None

def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """
    Find similar companies using FAISS similarity search.
    Args:
        df: DataFrame containing company data
        embeddings_normalized: Normalized embedding vectors
        index: FAISS index for similarity search
        company_name: Name of company to find similarities for
        top_n: Number of similar companies to return
    Returns:
        tuple: (Similar companies DataFrame, query company index) or (None, None) on error
    """
    try:
        # Find the query company
        company_mask = df['Name'].str.lower() == company_name.lower()
        if not company_mask.any():
            st.error(f"Company '{company_name}' not found in database.")
            return None, None
            
        company_index = company_mask.idxmax()
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        
        # Search for similar companies
        distances, indices = index.search(query_vector, top_n + 1)
        
        # Filter out the query company itself
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

def display_company_details(company, is_query=False):
    """
    Display company information in a clean format.
    Args:
        company: Series containing company information
        is_query: Boolean indicating if this is the query company
    """
    st.markdown(f"### {'📌 Query Company' if is_query else '🎯 Similar Company'}")
    st.write(f"**Company Name:** {company['Name']}")
    st.write(f"**Employee Count:** {company['Employee Count']}")
    if not is_query:
        st.write(f"**Similarity Score:** {company['Similarity Score']:.2f}")
    st.write(f"**Description:**\n{company['Combined_Description'][:500]}...")
    st.markdown("---")

def create_similarity_chart(similar_companies):
    """
    Create a bar chart visualization of similarity scores.
    Args:
        similar_companies: DataFrame containing similar companies
    Returns:
        plotly.graph_objects.Figure: Bar chart of similarity scores
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=similar_companies['Similarity Score'],
        y=similar_companies['Name'],
        orientation='h',
        marker_color='rgb(37, 99, 235)'
    ))
    
    fig.update_layout(
        title="Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis_title="Company Name",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def display_banner():
    """
    Display the Innovius banner image.
    Uses local image file with error handling and fallback.
    """
    try:
        # Try to load and display the banner image
        image = Image.open('Innovius Capital Cover.jpeg')
        st.image(image, use_container_width=True)
    except Exception as e:
        # Fallback text if image fails to load
        logger.error(f"Banner load error: {e}")
        st.markdown("""
            # INNOVIUS
            ## VENTURE DIFFERENTLY
        """)
        
def main():
    """Main application logic."""
    try:
        # Display header
        st.title("🔍 Company Similarity Finder")
        st.write("Find companies similar to your target using our AI-powered analysis.")
        
        # Display banner
        display_banner()
        
        # Load data if not already loaded
        if not st.session_state.data_loaded:
            df, embeddings_normalized, index = load_data()
            if df is not None:
                st.session_state.data = df
                st.session_state.embeddings = embeddings_normalized
                st.session_state.index = index
                st.session_state.data_loaded = True
        
        # Create input section
        col1, col2 = st.columns([2, 1])
        with col1:
            company_name = st.text_input(
                "🔎 Enter company name:",
                placeholder="e.g., Microsoft, Apple, Tesla..."
            )
        with col2:
            top_n = st.slider(
                "Number of similar companies:",
                1, 10, 5,
                help="Select how many similar companies to display"
            )
        
        # Process search when company name is entered
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
                    st.subheader("📊 Similarity Analysis")
                    fig = create_similarity_chart(similar_companies)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display similar companies
                    st.subheader("🎯 Similar Companies")
                    for _, company in similar_companies.iterrows():
                        display_company_details(company)
                    
                    # Add export option
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

if __name__ == "__main__":
    main()