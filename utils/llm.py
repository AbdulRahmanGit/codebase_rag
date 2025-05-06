import google.generativeai as genai
from typing import List, Dict, Any
import streamlit as st

def initialize_gemini(api_key: str) -> None:
    """Initialize Gemini API.
    
    Args:
        api_key: Gemini API key
    """
    genai.configure(api_key=api_key)

def generate_response(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate response using Gemini.
    
    Args:
        question: User's question
        context_chunks: Relevant context chunks from vector store
        
    Returns:
        Generated response
    """
    try:
        context = "\n\n".join([chunk['metadata']['text'] for chunk in context_chunks])
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer based on the code context."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"LLM error: {str(e)}")
        return "Error generating response. Please try again." 