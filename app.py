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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'download_attempted' not in st.session_state:
    st.session_state.download_attempted = False

# Set page configuration
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for enhanced visual appeal
st.markdown("""
    <style>
    /* Modern card styling */
    .company-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1f77b4;
    }
    
    /* Category badge styling */
    .category-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    
    /* Metric container styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Enhanced header styling */
    .stTitle {
        color: #1976d2 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

def display_banner():
    """Display the Innovius banner."""
    try:
        image = Image.open('Innovius Capital Cover.jpeg')
        # Fixed deprecated warning by using use_container_width
        st.image(image, use_container_width=True)
    except Exception as e:
        logger.error(f"Error loading banner: {str(e)}")
        st.markdown("""
            <div class="banner-container">
                <h1 class="banner-text">INNOVIUS</h1>
                <p style="color: white;">VENTURE DIFFERENTLY</p>
            </div>
        """, unsafe_allow_html=True)

@st.cache_data(show_spinner=True, ttl=3600)
def download_data():
    """Download data with proper error handling and user feedback."""
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
            
            embeddings = np.array(df['Embeddings'].tolist())
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings_normalized.astype('float32')
            
            dimension = embeddings_normalized.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_normalized)
            
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

def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """Find similar companies based on embedding similarity."""
    try:
        company_index = df[df['Name'].str.lower() == company_name.lower()].index[0]
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        distances, indices = index.search(query_vector, top_n + 1)
        
        # Filter out the query company
        mask = indices[0] != company_index
        similar_indices = indices[0][mask][:top_n]
        similar_distances = distances[0][mask][:top_n]
        
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        
        return similar_companies, company_index
    except IndexError:
        st.error(f"Company '{company_name}' not found in database.")
        return None, None
    except Exception as e:
        st.error(f"Error finding similar companies: {str(e)}")
        return None, None

def create_radar_chart(company_data):
    """Create a radar chart comparing key metrics."""
    categories = ['Employee Count', 'Similarity Score']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[company_data['Employee Count'], company_data['Similarity Score']],
        theta=categories,
        fill='toself',
        name=company_data['Name']
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(company_data['Employee Count'], 1)]
            )),
        showlegend=True,
        title="Company Metrics Comparison"
    )
    return fig

def create_category_distribution(similar_companies):
    """Create a pie chart showing category distribution."""
    category_counts = similar_companies['Top Level Category'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title="Category Distribution",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def display_company_details(company):
    """Display detailed company information in a formatted card."""
    st.markdown(f"""
    <div class="company-card">
        <h3>{company['Name']}</h3>
        <div class="metric-container">
            <p><strong>Organization ID:</strong> {company['Organization Id']}</p>
            <p><strong>Employees:</strong> {company['Employee Count']}</p>
            {f'<span class="category-badge">{company["Top Level Category"]}</span>' if pd.notna(company["Top Level Category"]) else ''}
            {f'<span class="category-badge">{company["Secondary Category"]}</span>' if pd.notna(company["Secondary Category"]) else ''}
        </div>
        <p>{company['Combined_Description']}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application logic with enhanced visualizations."""
    display_banner()
    
    if not st.session_state.data_loaded:
        st.info("🚀 Initializing the application...")
    
    try:
        df, embeddings_normalized, index = load_data()
        
        if df is None:
            st.error("Unable to load company database. Please refresh the page.")
            st.stop()
        
        st.title("🔍 Company Similarity Finder")
        st.markdown("""
        Discover companies similar to your target using our AI-powered analysis engine.
        Simply enter a company name below to explore related companies and understand their relationships.
        """)
        
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
        
        if company_name_input:
            similar_companies, company_index = get_similar_companies(
                df, embeddings_normalized, index, company_name_input, top_n
            )
            
            if similar_companies is not None:
                # Display query company
                query_company = df.iloc[company_index]
                st.subheader("📌 Query Company")
                display_company_details(query_company)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Similar Companies", "Category Analysis", "Metrics Comparison"])
                
                with tab1:
                    # Display similarity scores chart
                    fig_similarity = create_similarity_chart(similar_companies)
                    st.plotly_chart(fig_similarity, use_container_width=True)
                    
                    # Display similar companies with enhanced details
                    for _, company in similar_companies.iterrows():
                        display_company_details(company)
                
                with tab2:
                    # Display category distribution
                    fig_categories = create_category_distribution(similar_companies)
                    st.plotly_chart(fig_categories, use_container_width=True)
                
                with tab3:
                    # Display radar chart for the first similar company
                    if not similar_companies.empty:
                        fig_radar = create_radar_chart(similar_companies.iloc[0])
                        st.plotly_chart(fig_radar, use_container_width=True)
                
                # Export functionality
                col1, col2 = st.columns([1, 4])
                with col1:
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