import streamlit as st
from atlassian import Confluence
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import CharacterTextSplitter
import chromadb
import numpy as np
import tempfile
import os

# Set the Streamlit theme to light mode
st.set_page_config(layout="wide", page_title='Confluence Page Content Fetcher', initial_sidebar_state='auto')

# Define your Streamlit UI components
st.title('Confluence Page Content Fetcher')
confluence_page_id = st.text_input('Enter the Confluence page ID or relative URL path:')

# Initialize Confluence client with your actual Confluence credentials and URL
CONFLUENCE_URL = ''
CONFLUENCE_USERNAME = ''
CONFLUENCE_API_TOKEN = ''
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len
)
try:
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        password=CONFLUENCE_API_TOKEN
    )
    st.write("Confluence client initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Confluence client: {e}")

# Define functions to interact with Confluence
def get_page_id_or_path(url_or_id):
    if url_or_id.isdigit():
        return url_or_id
    else:
        match = re.search(r'pages/(\d+)', url_or_id)
        if match:
            return match.group(1)
        else:
            return None

def get_confluence_page_content(page_id):
    try:
        content = confluence.get_page_by_id(page_id, expand='body.storage')
        st.write(f"Fetched content for Confluence page ID: {page_id}")
        return content
    except Exception as e:
        st.write(f"Error fetching content for Confluence page ID {page_id}: {e}")
        return None

def html_to_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

if st.button('Submit'):
    if confluence_page_id:
        page_id = get_page_id_or_path(confluence_page_id)
        if page_id:
            content = get_confluence_page_content(page_id)
            if content:
                html_content = content.get('body', {}).get('storage', {}).get('value', '')
                plain_text = html_to_plain_text(html_content)
                st.text_area('Page Content:', plain_text, height=1000)
                chunks = text_splitter.split_text(plain_text)

                # Use a temporary directory for ChromaDB storage
                temp_dir = tempfile.gettempdir()
                db_path = os.path.join(temp_dir, "chromadb")
                
                client = chromadb.PersistentClient(path=db_path)
                collection_name = "confluenceDocuments"
                collection = client.get_or_create_collection(name=collection_name)

                chunk_ids = [str(i) for i in range(len(chunks))]

                # Split the chunks and metadata into batches
                batch_size = 1000
                num_batches = int(np.ceil(len(chunks) / batch_size))

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(chunks))
                    
                    batch_chunks = chunks[start_idx:end_idx]
                    batch_metadatas = [{"id": id} for id in chunk_ids[start_idx:end_idx]]
                    
                    collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metadatas,
                        ids=chunk_ids[start_idx:end_idx]
                    )
                st.write("Chunks have been successfully added to the ChromaDB collection.")
                print(collection.peek(2))

            else:
                st.error("There is no content with the given ID, or the calling user does not have permission to view the content.")
        else:
            st.error("Invalid Confluence page ID or URL path.")
    else:
        st.error('Please enter the Confluence page ID or URL path.')

st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #FFFFFF;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)