from dotenv import load_dotenv
import streamlit as st
from confluence_utils import (
    get_confluence_client, 
    get_page_id_or_path, 
    get_confluence_page_content, 
    html_to_plain_text
)
from langchain_utils import classify_query, retrieve_relevant_page
from explanation import explain_word
from question_answering import answer_question
from summarization import summarize_text
from atlassian import Confluence
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import CharacterTextSplitter
import chromadb
import numpy as np
import tempfile
import os

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    layout="centered",     
    page_title='Confluence Page Content Assistant',
    page_icon="https://img.icons8.com/?size=96&id=FpVS74LDFYc2&format=png"
)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'processed_page_ids' not in st.session_state:
    st.session_state.processed_page_ids = set()

# Streamlit UI components
st.title('Confluence Page Content Assistant')
confluence_page_ids = st.text_area('Enter the Confluence page IDs or relative URL paths (comma-separated):')
user_prompt = st.text_area('Enter your query:', height=100)

# Load environment variables for Confluence credentials
CONFLUENCE_URL = os.getenv('CONFLUENCE_URL')
CONFLUENCE_USERNAME = os.getenv('CONFLUENCE_USERNAME')
CONFLUENCE_API_TOKEN = os.getenv('CONFLUENCE_API_TOKEN')

# Initialize Confluence client
try:
    confluence = get_confluence_client(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
    st.success("Confluence client initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Confluence client: {e}")
    st.stop()

# Initialize ChromaDB
temp_dir = tempfile.gettempdir()
db_path = os.path.join(temp_dir, "chromadb")
client = chromadb.PersistentClient(path=db_path)
collection_name = "confluenceDocuments"
collection = client.get_or_create_collection(name=collection_name)

# Text splitter initialization
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len
)

# Sidebar for chat history
st.sidebar.subheader("Chat History")
selected_chat = None
for idx, chat in enumerate(st.session_state.chat_history):
    if st.sidebar.button(f"Query {idx + 1}: {chat['user']}"):
        selected_chat = chat

if selected_chat:
    st.subheader("Selected Chat")
    st.write(f"**User:** {selected_chat['user']}")
    st.write(f"**Assistant:** {selected_chat['assistant']}")
    st.write("---")

# Form submission handling
if st.button('Submit'):
    if confluence_page_ids and user_prompt.strip():
        page_ids = [pid.strip() for pid in confluence_page_ids.split(',')]
        valid_page_ids = [get_page_id_or_path(pid) for pid in page_ids if get_page_id_or_path(pid)]

        if valid_page_ids:
            try:
                pages_content = []
                for page_id in valid_page_ids:
                    if page_id in st.session_state.processed_page_ids:
                        continue  # Skip already processed pages
                    
                    # Fetch the page content from Confluence
                    content = get_confluence_page_content(confluence, page_id)
                    if content:
                        html_content = content.get('body', {}).get('storage', {}).get('value', '')
                        plain_text = html_to_plain_text(html_content)
                        pages_content.append((page_id, plain_text))
                        
                        # Split the page content into chunks
                        chunks = text_splitter.split_text(plain_text)
                        chunk_ids = [f"{page_id}_{i}" for i in range(len(chunks))]
                        
                        # Generate Confluence page URL
                        page_url = f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={page_id}"
                        # Add chunks and metadata to ChromaDB in batches
                        batch_size = 1000
                        num_batches = int(np.ceil(len(chunks) / batch_size))
                        
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
                            
                            batch_chunks = chunks[start_idx:end_idx]
                            batch_metadatas = [{"id": id, "url": page_url} for id in chunk_ids[start_idx:end_idx]]
                            
                            collection.add(
                                documents=batch_chunks,
                                metadatas=batch_metadatas,
                                ids=chunk_ids[start_idx:end_idx]
                            )
                        st.write(f"Page ID {page_id}: Successfully added to ChromaDB.")

                        # Mark the page as processed
                        st.session_state.processed_page_ids.add(page_id)
                    else:
                        st.error(f"Failed to fetch content for page ID {page_id}. Check permissions or validity.")

                # Retrieve relevant content for user's query
                relevant_page_content = retrieve_relevant_page(user_prompt, collection)

                # Classify the user's query
                query_type = classify_query(user_prompt)

                # Route query to the appropriate function
                if query_type == "Summarization":
                    response,url = summarize_text(relevant_page_content)
                    st.subheader('Page Summary')
                elif query_type == "Question Answering":
                    response,url = answer_question(user_prompt, relevant_page_content)
                    st.subheader('Answer')
                elif query_type == "Word Explanation":
                    response,url = explain_word(user_prompt, relevant_page_content)
                    st.subheader('Word Explanation')
                else:
                    response = "Unable to classify the query. Please try again."
                    st.error(response)

                # Display response
                st.text_area('', response, height=500)
                st.link_button("refer to the confluence page", url)
                # Add chat to session history
                st.session_state.chat_history.append({"user": user_prompt, "assistant": response})

            except Exception as e:
                st.error(f"Error processing request: {e}")
        else:
            st.error("Invalid Confluence page IDs or URL paths.")
    else:
        st.error('Please enter the Confluence page IDs and a query.')

# Custom Streamlit theme (light mode)
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
