import streamlit as st
from confluence_utils import get_confluence_client, get_page_id_or_path, get_confluence_page_content, html_to_plain_text
from langchain_utils import classify_query
from explanation import explain_word
from question_answering import answer_question
from summarization import summarize_text

# Set the Streamlit page configuration
st.set_page_config(layout="centered",     
    page_title='Confluence Page Content Assistant',
    page_icon="https://img.icons8.com/?size=96&id=FpVS74LDFYc2&format=png", )

# Initialize session state variables for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define your Streamlit UI components
st.title('Confluence Page Content Assistant')
confluence_page_id = st.text_input('Enter the Confluence page ID or relative URL path:')
user_prompt = st.text_area('Enter your query:', height=100)

CONFLUENCE_URL = ''
CONFLUENCE_USERNAME = ''
CONFLUENCE_API_TOKEN = ''
try:
    confluence = get_confluence_client(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
    st.success("Confluence client initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Confluence client: {e}")

# Interactive sidebar for chat history
st.sidebar.subheader("Chat History")
selected_chat = None
for idx, chat in enumerate(st.session_state.chat_history):
    if st.sidebar.button(f"Query {idx + 1}: {chat['user']}"):
        selected_chat = chat

# Display selected chat
if selected_chat:
    st.subheader("Selected Chat")
    st.write(f"**User:** {selected_chat['user']}")
    st.write(f"**Assistant:** {selected_chat['assistant']}")
    st.write("---")

# Handle the form submission
if st.button('Submit'):
    if confluence_page_id:
        page_id = get_page_id_or_path(confluence_page_id)
        if page_id:
            try:
                content = get_confluence_page_content(confluence, page_id)
                html_content = content.get('body', {}).get('storage', {}).get('value', '')
                plain_text = html_to_plain_text(html_content)
                
                # Classify the user's query
                query_type = classify_query(user_prompt)

                # Route the query to the appropriate agent
                if query_type == "Summarization":
                    response = summarize_text(plain_text)
                    st.subheader('Page Summary')
                elif query_type == "Question Answering":
                    response = answer_question(user_prompt, plain_text)
                    st.subheader('Answer')
                elif query_type == "Word Explanation":
                    response = explain_word(user_prompt, plain_text)
                    st.subheader('Word Explanation')
                else:
                    st.error("Unable to classify the query. Please try again.")
                    response = "Error: Unable to classify the query."
                
                st.text_area('', response, height=500)
                
                # Store the chat in session state
                st.session_state.chat_history.append({"user": user_prompt, "assistant": response})
            except Exception as e:
                st.error(f"Error processing the request: {e}")
        else:
            st.error("Invalid Confluence page ID or URL path.")
    else:
        st.error('Please enter the Confluence page ID or URL path.')
