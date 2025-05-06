import os
import tempfile
from git import Repo
from typing import Dict, Any, List
import streamlit as st

# Constants
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java', '.md',
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h', '.md'}
IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.gitignore', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

def get_repo_name(repo_url: str) -> str:
    """Extract repository name from URL.
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Repository name
    """
    return repo_url.split("/")[-1].lower()

def get_file_content(file_path: str, repo_path: str) -> Dict[str, Any]:
    """Get content of a single file.
    
    Args:
        file_path: Path to the file
        repo_path: Root path of the repository
        
    Returns:
        Dictionary containing file name and content
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"name": os.path.relpath(file_path, repo_path), "content": content}
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
        return None

def get_main_files_content(repo_path: str) -> List[Dict[str, Any]]:
    """Get content of all supported code files from repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of dictionaries containing file names and contents
    """
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
    """Clone a GitHub repository.
    
    Args:
        repo_url: URL of the GitHub repository
        
    Returns:
        Path to the cloned repository
    """
    try:
        repo_name = get_repo_name(repo_url)
        repo_path = os.path.join(tempfile.gettempdir(), repo_name)
        if not os.path.exists(repo_path):
            Repo.clone_from(repo_url, repo_path)
            st.success(f"Repository {repo_name} cloned successfully!")
        return repo_path
    except Exception as e:
        st.error(f"Repository cloning error: {str(e)}")
        return None 