import streamlit as st
import pinecone
from pinecone import ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import tempfile
import json
from pathlib import Path
import magic
from typing import List, Dict, Any
import time
from git import Repo
import re

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'pinecone': '', 'gemini': ''}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Constants
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java', '.md',
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h', '.md'}
IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.gitignore', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INDEX_NAME = "assignment"

def initialize_apis():
    """Initialize APIs and return Pinecone client"""
    try:
        pc = pinecone.Pinecone(api_key=st.session_state.api_keys['pinecone'])
        genai.configure(api_key=st.session_state.api_keys['gemini'])
        return pc, True
    except Exception as e:
        st.error(f"API initialization error: {str(e)}")
        return None, False
def get_embeddings(text: str) -> List[float]:
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(text).tolist()

def get_file_content(file_path: str, repo_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"name": os.path.relpath(file_path, repo_path), "content": content}
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
        return None

def get_main_files_content(repo_path: str) -> List[Dict[str, Any]]:
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue
            for file in files:
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(os.path.join(root, file), repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        st.error(f"Repository walk error: {str(e)}")
    return files_content

def clone_repository(repo_url: str) -> str:
    try:
        repo_name = repo_url.split("/")[-1]
        repo_path = os.path.join(tempfile.gettempdir(), repo_name)
        if not os.path.exists(repo_path):
            Repo.clone_from(repo_url, repo_path)
            st.success(f"Repository {repo_name} cloned successfully!")
        return repo_path
    except Exception as e:
        st.error(f"Repository cloning error: {str(e)}")
        return None

def store_in_pinecone(files_content: List[Dict[str, Any]], namespace: str, pc: pinecone.Pinecone):
    try:
        if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME,
                spec=ServerlessSpec(cloud='aws',region='us-east-1'),
                dimension=768,
                metric="cosine"
            )
        
        index = pc.Index(INDEX_NAME)
        for file_data in files_content:
            chunks = [file_data['content'][i:i+CHUNK_SIZE] for i in range(0, len(file_data['content']), CHUNK_SIZE-CHUNK_OVERLAP)]
            for i, chunk in enumerate(chunks):
                embedding = get_embeddings(chunk)
                index.upsert(
                    vectors=[{
                        'id': f"{file_data['name']}_{i}",
                        'values': embedding,
                        'metadata': {'file_name': file_data['name'], 'chunk_index': i, 'text': chunk}
                    }],
                    namespace=namespace
                )
        return True
    except Exception as e:
        st.error(f"Pinecone storage error: {str(e)}")
        return False

def query_rag(question: str, namespace: str, pc: pinecone.Pinecone) -> str:
    try:
        question_embedding = get_embeddings(question)
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=question_embedding,
            top_k=10,
            include_metadata=True,
            namespace=namespace
        )
        context = "\n\n".join([match['metadata']['text'] for match in results['matches']])
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer based on the code context."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Query error: {str(e)}"

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
    st.slider("Chunk Size", 500, 2000, CHUNK_SIZE, key="chunk_size")
    st.slider("Chunk Overlap", 100, 500, CHUNK_OVERLAP, key="chunk_overlap")
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
        namespace = repo_url.split("/")[-1]
        if st.button("Analyze Repository"):
            with st.spinner("Processing repository..."):
                repo_path = clone_repository(repo_url)
                if repo_path:
                    files_content = get_main_files_content(repo_path)
                    if store_in_pinecone(files_content, namespace, pc):
                        st.success("Analysis complete!")
                        st.session_state.uploaded_files.extend(files_content)

# Chat Interface
st.subheader("Chat")
question = st.text_area("Ask a question about the codebase", height=100, 
                       placeholder="Type your question here...")

if st.button("Submit Question", key="submit_question") and question and 'namespace' in locals() and api_initialized:
    with st.spinner("Generating response..."):
        response = query_rag(question, namespace, pc)
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