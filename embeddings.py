import json
import numpy as np
from config import MODEL_NAME
from utils import retry

@retry(retries=4, backoff=3)
def get_query_embedding(question: str, co) -> np.ndarray:
    """Generates an embedding for a given query string."""
    resp = co.embed(
        model=MODEL_NAME,
        input_type="search_query",
        embedding_types=["float"],
        texts=[question],
    )
    return np.asarray(resp.embeddings.float[0])

def load_image_embeddings(embeddings_path: str):
    """Load image embeddings and corresponding filenames."""
    with open(embeddings_path, "r") as f:
        image_embeddings = json.load(f)

    filenames = list(image_embeddings.keys())
    embeddings = np.vstack(list(image_embeddings.values()))
    
    return embeddings, filenames
