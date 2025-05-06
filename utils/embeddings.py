from sentence_transformers import SentenceTransformer
from typing import List

def get_embeddings(text: str) -> List[float]:
    """Generate embeddings using SentenceTransformer.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of embedding values
    """
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(text).tolist() 