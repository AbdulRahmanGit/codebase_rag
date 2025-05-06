import streamlit as st
from datetime import datetime
import re
from utils import (
    initialize_pinecone,
    initialize_gemini,
    clone_repository,
    get_main_files_content,
    store_in_pinecone,
    query_pinecone,
    generate_response,
    get_repo_name
)

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'pinecone': '', 'gemini': ''}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_apis():
    """Initialize APIs and return Pinecone client"""
    try:
        pc = initialize_pinecone(st.session_state.api_keys['pinecone'])
        initialize_gemini(st.session_state.api_keys['gemini'])
        return pc, True
    except Exception as e:
        st.error(f"API initialization error: {str(e)}")
        return None, False

# UI Setup
st.set_page_config(page_title="Codebase RAG", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Settings")
    pinecone_key = st.text_input("Pinecone API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    if pinecone_key and gemini_key:
        st.session_state.api_keys['pinecone'] = pinecone_key
        st.session_state.api_keys['gemini'] = gemini_key
    
    st.subheader("Configuration")
    st.slider("Chunk Size", 500, 2000, key="chunk_size")
    st.slider("Chunk Overlap", 100, 500, key="chunk_overlap")
    st.checkbox("Debug Mode", key="debug_mode")

# Main Content
st.title("Codebase RAG System")
st.write("Analyze repositories and ask questions about their code.")

# Initialize APIs
pc, api_initialized = initialize_apis()

# Repository Analysis
st.subheader("Repository Analysis")
repo_url = st.text_input("Enter GitHub repository URL", placeholder="https://github.com/user/repo")

if repo_url and api_initialized:
    if not re.match(r"^https://github\.com/[^/]+/[^/]+/?$", repo_url):
        st.error("Invalid GitHub URL format")
    else:
        index_name = get_repo_name(repo_url)
        namespace = index_name
        
        if st.button("Analyze Repository"):
            with st.spinner("Processing repository..."):
                repo_path = clone_repository(repo_url)
                if repo_path:
                    files_content = get_main_files_content(repo_path)
                    if store_in_pinecone(files_content, index_name, namespace, pc):
                        st.success("Analysis complete!")
                        st.session_state.uploaded_files.extend(files_content)

# Chat Interface
st.subheader("Chat")
question = st.text_area("Ask a question about the codebase", height=100, 
                       placeholder="Type your question here...")

if st.button("Submit Question", key="submit_question") and question and 'index_name' in locals() and api_initialized:
    with st.spinner("Generating response..."):
        context_chunks = query_pinecone(question, index_name, namespace, pc)
        response = generate_response(question, context_chunks)
        st.session_state.chat_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Chat History
if st.session_state.chat_history:
    st.subheader("History")
    for chat in reversed(st.session_state.chat_history):
        with st.expander(f"Q: {chat['question']} ({chat['timestamp']})"):
            st.write(f"A: {chat['response']}")
            if st.button("Export", key=f"export_{chat['timestamp']}"):
                markdown_content = f"# Q&A\n\n**Question:** {chat['question']}\n\n**Answer:** {chat['response']}\n\n*Timestamp: {chat['timestamp']}*"
                st.download_button(
                    label="Download",
                    data=markdown_content,
                    file_name=f"qa_{chat['timestamp']}.md",
                    mime="text/markdown"
                )

# Debug Info
if st.session_state.debug_mode:
    st.subheader("Debug")
    st.json({
        'files': st.session_state.uploaded_files,
        'api_status': {
            'pinecone': bool(st.session_state.api_keys['pinecone']),
            'gemini': bool(st.session_state.api_keys['gemini'])
        }
    }) 