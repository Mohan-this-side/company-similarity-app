# Import required libraries
import streamlit as st  # For creating web application
import pandas as pd    # For data manipulation
import numpy as np     # For numerical operations
from sentence_transformers import SentenceTransformer  # For text embeddings
import faiss          # For efficient similarity search
import plotly.express as px      # For interactive visualizations
import plotly.graph_objects as go # For custom interactive plots
import logging        # For application logging
import time          # For timing operations
from PIL import Image # For image processing
import os            # For file operations
import gdown         # For downloading from Google Drive
import gc            # For garbage collection
import torch         # For GPU memory management
import re            # For regular expressions

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables to maintain state between reruns
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

# Custom CSS styling for dark theme and UI components
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
    
    /* Style for hyperlinks in company cards */
    .company-website {
        color: #4CAF50;
        text-decoration: none;
        padding: 5px 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    
    .company-website:hover {
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    /* Input field styles */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    .stNumberInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

def clean_description(text):
    """
    Clean HTML tags and unwanted elements from company description.
    
    Args:
        text (str): Raw description text containing HTML and special characters
    Returns:
        str: Cleaned and formatted description text
    """
    if pd.isna(text):
        return ""
    
    # Remove malformed or incomplete HTML tags
    text = re.sub(r'</div>\s*(?!<)', ' ', text)  # Remove orphaned closing div tags
    text = re.sub(r'<p>\s*(?!</p>)', ' ', text)  # Remove orphaned opening p tags
    
    # Remove all complete HTML tag pairs
    text = re.sub(r'<[^>]*>', '', text)
    
    # Clean up special characters and normalize whitespace
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()

def format_website_link(website):
    """
    Format website URL into a proper HTML hyperlink.
    
    Args:
        website (str): Raw website URL
    Returns:
        str: Formatted HTML hyperlink or empty string if URL is invalid
    """
    if pd.isna(website):
        return ""
    
    # Ensure URL has proper format with http/https
    if not website.startswith(('http://', 'https://')):
        website = 'https://' + website
    
    return f'<a href="{website}" target="_blank" class="company-website">🌐 Visit Website</a>'

def display_company_details(company):
    """
    Display detailed company information in a formatted card.
    
    Args:
        company (pd.Series): Company information including name, description, website, etc.
    """
    description = clean_description(company.get('Combined_Description', ''))
    website = company.get('Website', '')
    website_html = format_website_link(website) if pd.notna(website) else ''
    
    # Handle category display
    category_html = []
    if pd.notna(company.get("Top Level Category")):
        category_html.append(f'<span class="category-badge">{company["Top Level Category"]}</span>')
    if pd.notna(company.get("Secondary Category")):
        category_html.append(f'<span class="category-badge">{company["Secondary Category"]}</span>')
    
    categories_display = ' '.join(category_html) if category_html else 'Not Provided'
    
    # Create and display company card
    card_html = f"""
    <div class="company-card">
        <h3>{company['Name']}</h3>
        <div class="metric-container">
            <p><strong>Organization ID:</strong> {company['Organization Id']}</p>
            <p><strong>Employees:</strong> {company['Employee Count']}</p>
            {website_html}
            {categories_display}
        </div>
        <div class="description-container">
            <p>{description}</p>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def display_banner():
    """
    Display the Innovius banner image or fallback to text header.
    """
    try:
        image = Image.open('Innovius Capital Cover.jpeg')
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
    """
    Download company database from Google Drive with caching.
    
    Returns:
        str: Path to downloaded data file or None if download fails
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

@st.cache_data(show_spinner=True, ttl=3600)
def load_data():
    """
    Load and process company data, creating FAISS index for similarity search.
    
    Returns:
        tuple: (DataFrame, normalized embeddings, FAISS index) or (None, None, None) if loading fails
    """
    try:
        data_path = download_data()
        if data_path is None:
            return None, None, None

        with st.spinner('Processing company data...'):
            # Load data and create FAISS index
            df = pd.read_pickle(data_path)
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

def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """
    Find similar companies using FAISS similarity search.
    
    Args:
        df (pd.DataFrame): Company database
        embeddings_normalized (np.array): Normalized company embeddings
        index (faiss.Index): FAISS similarity index
        company_name (str): Name of query company
        top_n (int): Number of similar companies to return
    
    Returns:
        tuple: (DataFrame of similar companies, index of query company) or (None, None) if search fails
    """
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
    """
    Create horizontal bar chart showing similarity scores.
    
    Args:
        similar_companies (pd.DataFrame): DataFrame containing similar companies and their scores
    
    Returns:
        go.Figure: Plotly figure object for similarity chart
    """
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

def create_category_distribution(similar_companies):
    """
    Create pie chart showing distribution of company categories.
    
    Args:
        similar_companies (pd.DataFrame): DataFrame containing similar companies
    
    Returns:
        go.Figure: Plotly figure object for category distribution chart
    """
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

def main():
    """
    Main application logic and UI layout.
    """
    display_banner()
    
    if not st.session_state.data_loaded:
        st.info("🚀 Initializing the application...")
    
    try:
        # Load data and initialize FAISS index
        df, embeddings_normalized, index = load_data()
        
        if df is None:
            st.error("Unable to load company database. Please refresh the page.")
            st.stop()
        
        # Application header and description
        st.title("🔍 Company Similarity Finder")
        st.markdown("""
        Discover companies similar to your target using our AI-powered analysis engine.
        Simply enter a company name below to explore related companies and understand their relationships.
        """)
        
        # User input section
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
            # Find and display similar companies
            similar_companies, company_index = get_similar_companies(
                df, embeddings_normalized, index, company_name_input, top_n
            )
            
            if similar_companies is not None:
                # Display query company details
                query_company = df.iloc[company_index]
                st.subheader("📌 Query Company")
                display_company_details(query_company)
                
                # Create visualization tabs
                tab1, tab2 = st.tabs(["Similar Companies", "Category Analysis"])
                
                with tab1:
                    fig_similarity = create_similarity_chart(similar_companies)
                    st.plotly_chart(fig_similarity, use_container_width=True)
                    
                    for _, company in similar_companies.iterrows():
                        display_company_details(company)
                
                with tab2:
                    fig_categories = create_category_distribution(similar_companies)
                    st.plotly_chart(fig_categories, use_container_width=True)
                
                
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