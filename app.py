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

# ============================================================================
# PAGE CONFIGURATION AND STYLING
# ============================================================================

# Configure the Streamlit page with custom settings
st.set_page_config(
    page_title="Innovius Capital - Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the visual appearance of the app
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 0rem;
    }
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header and title styling */
    .stTitle {
        font-size: 2.5rem !important;
        color: #1a365d !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Card styling for company information */
    .company-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
    }
    
    /* Metric card styling for key statistics */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Search box enhancement */
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 2rem;
        background-color: #2563eb;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_banner():
    """
    Load and display the banner image from the repository.
    Returns None if the image cannot be loaded.
    """
    try:
        # Attempt to load the banner image
        image = Image.open('Innovius Capital Cover.jpeg')
        # Display the banner with full width
        st.image(image, use_column_width=True)
    except Exception as e:
        st.warning("Banner image not loaded. Using default header.")
        logger.error(f"Error loading banner: {str(e)}")

@st.cache_data(show_spinner=True)
def download_data():
    """
    Download the data file from Google Drive if not present locally.
    Returns the path to the data file or None if download fails.
    """
    data_path = 'data_with_embeddings.pkl'
    if not os.path.exists(data_path):
        try:
            with st.status("📥 Downloading company database...", expanded=True) as status:
                st.write("Initializing download...")
                file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, data_path, quiet=False)
                st.write("✅ Download complete!")
                status.update(label="Database ready!", state="complete")
        except Exception as e:
            st.error(f"❌ Error downloading file: {str(e)}")
            return None
    return data_path

@st.cache_data(show_spinner=True)
def load_data():
    """
    Load and process the data from the pickle file.
    Returns tuple of (DataFrame, normalized embeddings, FAISS index)
    """
    data_path = download_data()
    if data_path is None:
        return None, None, None
    
    try:
        with st.spinner('🔄 Processing company data...'):
            # Load the DataFrame
            df = pd.read_pickle(data_path)
            
            # Process embeddings
            embeddings = np.array(df['Embeddings'].tolist())
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings_normalized.astype('float32')
            
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
        st.error(f"❌ Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None

def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    """
    Find similar companies based on embedding similarity.
    Returns tuple of (similar companies DataFrame, query company index)
    """
    try:
        # Find the index of the query company
        company_index = df[df['Name'].str.lower() == company_name.lower()].index[0]
        
        # Get the embedding vector for the query company
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        
        # Search for similar companies
        distances, indices = index.search(query_vector, top_n + 1)
        
        # Filter out the query company and get top N similar companies
        similar_indices = indices[0][indices[0] != company_index][:top_n]
        similar_distances = distances[0][indices[0] != company_index][:top_n]
        
        # Create DataFrame with similar companies
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        
        return similar_companies, company_index
    except IndexError:
        st.error(f"Company '{company_name}' not found in database.")
        return None, None
    except Exception as e:
        st.error(f"Error finding similar companies: {str(e)}")
        return None, None

def create_similarity_visualization(similar_companies):
    """
    Create an interactive visualization of similarity scores using Plotly.
    """
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=similar_companies['Similarity Score'],
        y=similar_companies['Name'],
        orientation='h',
        marker_color='rgb(37, 99, 235)',
        hovertemplate="<b>%{y}</b><br>" +
                      "Similarity Score: %{x:.2f}<br>" +
                      "<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title="Company Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis_title="Company Name",
        plot_bgcolor='white',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def main():
    """
    Main application function that orchestrates the app's functionality.
    """
    # Display banner
    load_banner()
    
    # Title and description
    st.title("🔍 Company Similarity Finder")
    st.markdown("""
    Find companies similar to your target based on our advanced AI-powered similarity analysis. 
    Enter a company name below to discover related companies and understand their relationships.
    """)
    
    # Load data
    with st.spinner('Loading company database...'):
        df, embeddings_normalized, index = load_data()

    if df is None:
        st.error("Failed to load company database. Please try again later.")
        st.stop()

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Company search input
        company_name_input = st.text_input(
            "🔎 Enter a company name:",
            placeholder="e.g., Microsoft, Apple, Tesla..."
        )
    
    with col2:
        # Number of similar companies selector
        top_n = st.slider(
            "Number of similar companies:",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many similar companies you want to see"
        )

    # Process company search
    if company_name_input:
        with st.spinner('🔄 Finding similar companies...'):
            similar_companies, company_index = get_similar_companies(
                df, embeddings_normalized, index, company_name_input, top_n
            )
        
        if similar_companies is not None:
            # Display query company details
            query_company = df.iloc[company_index]
            
            st.markdown("### 🎯 Query Company")
            with st.container():
                st.markdown(f"""
                <div class="company-card">
                    <h3>{query_company['Name']}</h3>
                    <p><strong>Employee Count:</strong> {query_company['Employee Count']}</p>
                    <p>{query_company['Combined_Description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display similarity visualization
            st.markdown("### 📊 Similar Companies")
            fig = create_similarity_visualization(similar_companies)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed company information
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
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)