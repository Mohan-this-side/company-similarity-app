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

# Set page configuration
st.set_page_config(
    page_title="Company Similarity Finder",
    page_icon="🔍",
    layout="wide",
)

# Configure logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data and models
@st.cache_data(show_spinner=True)
def download_data():
    data_path = 'data_with_embeddings.pkl'
    if not os.path.exists(data_path):
        try:
            st.info("Downloading data file... This might take a few minutes.")
            # Google Drive file ID
            file_id = '1Lw9Ihrf0tz7MnWA-dO_q0fGFyssddTlI'
            # Construct the download URL
            url = f'https://drive.google.com/uc?id={file_id}'
            # Download the file
            gdown.download(url, data_path, quiet=False)
            if not os.path.exists(data_path):
                st.error("Failed to download the data file.")
                return None
        except Exception as e:
            st.error(f"Error downloading file: {str(e)}")
            return None
    return data_path

@st.cache_data(show_spinner=True)
def load_data():
    data_path = download_data()
    if data_path is None:
        return None, None, None
    
    try:
        # Read the pickle file in chunks
        df = pd.read_pickle(data_path)
        
        # Convert embeddings to numpy array and normalize
        embeddings = np.array(df['Embeddings'].tolist())
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings_normalized.astype('float32')
        
        # Build FAISS index
        dimension = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_normalized)
        
        # Clear some memory
        del embeddings
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return df, embeddings_normalized, index
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None

# Function to get similar companies
def get_similar_companies(df, embeddings_normalized, index, company_name, top_n=5):
    try:
        company_index = df[df['Name'].str.lower() == company_name.lower()].index[0]
    except IndexError:
        return None, None
    except Exception as e:
        st.error(f"Error finding company: {str(e)}")
        return None, None
    
    try:
        query_vector = embeddings_normalized[company_index].reshape(1, -1)
        distances, indices = index.search(query_vector, top_n + 1)
        similar_indices = indices[0][indices[0] != company_index][:top_n]
        similar_distances = distances[0][indices[0] != company_index][:top_n]
        similar_companies = df.iloc[similar_indices].copy()
        similar_companies['Similarity Score'] = similar_distances
        return similar_companies, company_index
    except Exception as e:
        st.error(f"Error finding similar companies: {str(e)}")
        return None, None

def main():
    # Title and description
    st.title("🔍 Company Similarity Finder")
    st.markdown("""
    Enter a company name to find the top N companies similar to it based on their descriptions.
    """)

    # Load data
    with st.spinner('Loading data... This might take a few minutes on first run.'):
        df, embeddings_normalized, index = load_data()

    if df is None:
        st.error("Failed to load data. Please try again later.")
        st.stop()

    # Input company name
    company_name_input = st.text_input("Enter a company name:")

    # Number of similar companies to display
    top_n = st.slider("Number of similar companies to display:", min_value=1, max_value=20, value=5)

    if company_name_input:
        with st.spinner('Searching for similar companies...'):
            similar_companies, company_index = get_similar_companies(df, embeddings_normalized, index, company_name_input, top_n)
        
        if similar_companies is not None:
            st.success(f"Top {top_n} companies similar to '{company_name_input}':")
            
            # Display the query company details
            query_company = df.iloc[company_index]
            st.subheader(f"Query Company: {query_company['Name']}")
            st.write(f"**Employee Count:** {query_company['Employee Count']}")
            st.write(f"**Description:** {query_company['Combined_Description']}")
            st.write("---")
            
            # Display similar companies
            st.subheader("Similar Companies:")
            # Visualize similarity scores
            chart_data = similar_companies[['Name', 'Similarity Score']]
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Similarity Score',
                y=alt.Y('Name', sort='-x')
            ).properties(title='Similarity Scores')
            st.altair_chart(chart, use_container_width=True)
            
            # Display the DataFrame
            st.write(similar_companies[['Name', 'Employee Count', 'Similarity Score', 'Combined_Description']].reset_index(drop=True))
        else:
            st.error(f"Company '{company_name_input}' not found in the database.")

if __name__ == "__main__":
    main()