from typing import List, Dict, Any
import streamlit as st
import pinecone
from pinecone import ServerlessSpec
from .embeddings import get_embeddings
# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def initialize_pinecone(api_key: str) -> pinecone.Pinecone:
    """Initialize Pinecone client.
    
    Args:
        api_key: Pinecone API key
        
    Returns:
        Initialized Pinecone client
    """
    return pinecone.Pinecone(api_key=api_key)

def store_in_pinecone(files_content: List[Dict[str, Any]], index_name: str, namespace: str, pc: pinecone.Pinecone) -> bool:
    """Store file chunks in Pinecone.
    
    Args:
        files_content: List of file contents to store
        index_name: Name of the Pinecone index
        namespace: Namespace for the vectors
        pc: Pinecone client
        
    Returns:
        Success status
    """
    try:
        if index_name not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                spec=ServerlessSpec(cloud='aws',region='us-east-1'),
                dimension=768,
                metric="cosine"
            )
        
        index = pc.Index(index_name)
        for file_data in files_content:
            chunks = [file_data['content'][i:i+CHUNK_SIZE] 
                     for i in range(0, len(file_data['content']), CHUNK_SIZE-CHUNK_OVERLAP)]
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

def query_pinecone(question: str, index_name: str, namespace: str, pc: pinecone.Pinecone) -> List[Dict[str, Any]]:
    """Query Pinecone for relevant chunks.
    
    Args:
        question: Query text
        index_name: Name of the Pinecone index
        namespace: Namespace to search in
        pc: Pinecone client
        
    Returns:
        List of relevant chunks with metadata
    """
    try:
        question_embedding = get_embeddings(question)
        index = pc.Index(index_name)
        results = index.query(
            vector=question_embedding,
            top_k=10,
            include_metadata=True,
            namespace=namespace
        )
        return results.matches
    except Exception as e:
        st.error(f"Pinecone query error: {str(e)}")
        return [] 