
from sentence_transformers import SentenceTransformer

# Load a lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(texts: list[str]):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings
