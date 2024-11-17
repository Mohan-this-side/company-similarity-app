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

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables for data management
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'download_attempted' not in st.session_state:
    st.session_state.download_attempted = False

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Innovius - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and enhanced styling
st.markdown("""
<style>
    /* Dark theme for main app background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Modern dark card styling */
    .company-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #1f77b4;
        color: #FAFAFA;
    }
    
    /* Dark theme category badge */
    .category-badge {
        background-color: #2C3E50;
        color: #ECF0F1;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    
    /* Dark theme metric container */
    .metric-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #FAFAFA;
    }
    
    /* Enhanced header styling for dark theme */
    .stTitle {
        color: #3498DB !important;
        font-weight: 600 !important;
    }
    
    /* Dark theme for input fields */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Dark theme for selectbox */
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Dark theme for number input */
    .stNumberInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Banner styling */
    .banner-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .banner-text {
        color: #3498DB;
        font-size: 2.5em;
        font-weight: bold;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

def display_banner():
    """Display the application banner with logo."""
    try:
        image = Image.open('Innovius Capital Cover.jpeg')
        st.image(image, use_container_width=True)
    except Exception as e:
        logger.error(f"Error loading banner: {str(e)}")
        # Fallback banner if image fails to load
        st.markdown("""
        <div class="banner-container">
            <h1 class="banner-text">INNOVIUS</h1>
            <p style="color: white;">VENTURE DIFFERENTLY</p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(show_spinner=True, ttl=3600)
def download_data():
    """
    Download the company database from Google Drive.
    Returns:
        str: Path to the downloaded data file or None if download fails
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
            st.error(f"Error downloading file: {str(e)}")
            logger.error(f"Download error: {str(e)}")
            return None
    return data_path

def create_radar_chart(company_data):
    """
    Create an interactive radar chart comparing key metrics.
    
    Args:
        company_data (pd.Series): Data for a single company
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    fig = go.Figure()
    
    # Define metrics to compare
    metrics = ['Employee Count', 'Similarity Score']
    values = [company_data['Employee Count'], company_data['Similarity Score']]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name=company_data['Name'],
        line=dict(color='rgb(37, 99, 235)'),
        fillcolor='rgba(37, 99, 235, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor='rgba(128,128,128,0.2)',
                range=[0, max(values)]
            ),
            angularaxis=dict(
                gridcolor='rgba(128,128,128,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        title=dict(
            text="Company Metrics Comparison",
            font=dict(color='#FAFAFA')
        ),
        showlegend=True
    )
    return fig

def display_company_details(company):
    """
    Display detailed company information in a formatted card.
    
    Args:
        company (pd.Series): Company information
    """
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
    """Main application logic with enhanced error handling and user experience."""
    display_banner()
    
    if not st.session_state.data_loaded:
        with st.status("🚀 Initializing application...", expanded=True) as status:
            df, embeddings_normalized, index = load_data()
            if df is None:
                status.update(label="❌ Failed to load database", state="error")
                st.stop()
            status.update(label="✅ Application ready!", state="complete")
    
    st.title("🔍 Company Similarity Finder")
    
    # Create two columns for input controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_name_input = st.text_input(
            "🔎 Enter a company name:",
            placeholder="e.g., Microsoft, Apple, Tesla...",
            key="company_search"
        )
    
    with col2:
        top_n = st.slider(
            "Number of similar companies:",
            min_value=1,
            max_value=20,
            value=5,
            key="similarity_count"
        )
    
    if company_name_input:
        with st.spinner("🔍 Finding similar companies..."):
            similar_companies, company_index = get_similar_companies(
                df, embeddings_normalized, index, company_name_input, top_n
            )
            
        if similar_companies is not None:
            # Display query company
            query_company = df.iloc[company_index]
            st.subheader("📌 Query Company")
            display_company_details(query_company)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs([
                "Similar Companies",
                "Category Analysis",
                "Metrics Comparison"
            ])
            
            with tab1:
                fig_similarity = create_similarity_chart(similar_companies)
                st.plotly_chart(fig_similarity, use_container_width=True)
                
                for _, company in similar_companies.iterrows():
                    display_company_details(company)
            
            with tab2:
                fig_categories = create_category_distribution(similar_companies)
                st.plotly_chart(fig_categories, use_container_width=True)
            
            with tab3:
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)