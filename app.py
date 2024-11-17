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

# Add this CSS at the beginning of your custom CSS section
st.markdown("""
<style>
    /* Dark theme for main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Dark theme for company cards */
    .company-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #1f77b4;
        color: #FAFAFA;
    }
    
    /* Dark theme for metric containers */
    .metric-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #FAFAFA;
    }
    
    /* Style for input fields and widgets */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Style for selectbox */
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Style for number input */
    .stNumberInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Add custom CSS for enhanced visual appeal
# st.markdown("""
#     <style>
#     /* Modern card styling */
#     .company-card {
#         background-color: #ffffff;
#         border-radius: 10px;
#         padding: 20px;
#         margin: 10px 0;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         border-left: 5px solid #1f77b4;
#     }
    
#     /* Category badge styling */
#     .category-badge {
#         background-color: #e3f2fd;
#         color: #1976d2;
#         padding: 5px 10px;
#         border-radius: 15px;
#         font-size: 0.8em;
#         margin-right: 5px;
#     }
    
#     /* Metric container styling */
#     .metric-container {
#         background-color: #f8f9fa;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 10px 0;
#     }
    
#     /* Enhanced header styling */
#     .stTitle {
#         color: #1976d2 !important;
#         font-weight: 600 !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

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
def clean_description(text):
    """
    Clean HTML tags and unnecessary elements from text.
    
    Args:
        text (str): Raw text that may contain HTML tags
    Returns:
        str: Cleaned text with HTML tags removed
    """
    if pd.isna(text):
        return ""
    
    # Remove common HTML tags and artifacts
    replacements = {
        '</div>': '',
        '<div>': '',
        '</p>': '',
        '<p>': '',
        '\n': ' '
    }
    
    cleaned_text = text
    for old, new in replacements.items():
        cleaned_text = cleaned_text.replace(old, new)
    
    # Remove multiple spaces
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def format_website_url(website):
    """
    Format website URL to ensure it has proper http/https prefix.
    
    Args:
        website (str): Raw website URL
    Returns:
        str: Properly formatted URL with https prefix
    """
    if pd.isna(website):
        return ""
        
    # Clean the website string
    website = str(website).strip().lower()
    
    # Add https:// if no protocol specified
    if not website.startswith(('http://', 'https://')):
        website = f'https://{website}'
        
    return website

@st.cache_data(show_spinner=True, ttl=3600)
def load_data():
    """Load and process data with improved error handling and status updates."""
    try:
        data_path = download_data()
        if data_path is None:
            return None, None, None

        # Use status instead of spinner for better feedback
        with st.status('Processing company data...', expanded=True) as status:
            status.write("Loading data file...")
            df = pd.read_pickle(data_path)
            
            # Clean company descriptions
            status.write("Cleaning company descriptions...")
            df['Combined_Description'] = df['Combined_Description'].apply(clean_description)
            
            # Format website URLs
            status.write("Processing website URLs...")
            if 'Website' in df.columns:
                df['Website'] = df['Website'].fillna('')
            
            status.write("Processing embeddings...")
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
            
            status.update(label="✅ Data loading complete!", state="complete")
            st.session_state.data_loaded = True
            return df, embeddings_normalized, index
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading error: {str(e)}")
        return None, None, None

def display_company_details(company):
    """
    Display detailed company information in a formatted card with website link.
    
    Args:
        company (pd.Series): Company data including name, description, website etc.
    """
    # Format website URL and create HTML link if website exists
    website_html = ""
    if pd.notna(company.get('Website')):
        formatted_url = format_website_url(company['Website'])
        website_html = f"""
        <p><strong>Website:</strong> 
            <a href="{formatted_url}" target="_blank" style="color: #1976d2; text-decoration: underline;">
                {company['Website']}
            </a>
        </p>
        """
    
    # Create the company card with all details
    st.markdown(f"""
    <div class="company-card">
        <h3>{company['Name']}</h3>
        <div class="metric-container">
            <p><strong>Organization ID:</strong> {company['Organization Id']}</p>
            <p><strong>Employees:</strong> {company['Employee Count']}</p>
            {website_html}
            {f'<span class="category-badge">{company["Top Level Category"]}</span>' if pd.notna(company["Top Level Category"]) else ''}
            {f'<span class="category-badge">{company["Secondary Category"]}</span>' if pd.notna(company["Secondary Category"]) else ''}
        </div>
        <p>{clean_description(company['Combined_Description'])}</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=True, ttl=3600)
def download_data():
    """Download data with enhanced error handling and progress feedback."""
    data_path = 'data_with_embeddings.pkl'
    
    if st.session_state.download_attempted and not os.path.exists(data_path):
        return None
    
    if not os.path.exists(data_path):
        try:
            st.session_state.download_attempted = True
            
            # Create download status indicator with detailed feedback
            with st.status("📥 Downloading company database...", expanded=True) as status:
                status.write("Initiating download...")
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                
                status.write("Downloading data file (this may take a few minutes)...")
                
                # Set download timeout and implement retry logic
                start_time = time.time()
                timeout = 300  # 5 minutes timeout
                max_retries = 3
                retry_count = 0
                
                while not os.path.exists(data_path) and retry_count < max_retries:
                    if time.time() - start_time > timeout:
                        status.update(label="❌ Download timed out!", state="error")
                        return None
                        
                    try:
                        gdown.download(url, data_path, quiet=False)
                        break
                    except Exception as e:
                        retry_count += 1
                        status.write(f"Download attempt {retry_count} failed, retrying... ({str(e)})")
                        time.sleep(5)  # Wait 5 seconds before retrying
                
                if os.path.exists(data_path):
                    status.update(label="✅ Download complete!", state="complete")
                else:
                    status.update(label="❌ Download failed after maximum retries!", state="error")
                    return None
                
        except Exception as e:
            st.error(f"Error downloading file: {str(e)}")
            logger.error(f"Download error: {str(e)}")
            return None
    return data_path

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