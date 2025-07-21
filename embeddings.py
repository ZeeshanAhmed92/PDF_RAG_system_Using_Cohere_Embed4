import json
import numpy as np
from config import MODEL_NAME
from utils import retry

@retry(retries=4, backoff=3)
def get_query_embedding(query: str, co):
    response = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        inputs=[{"text": query}]
    )
    return np.asarray(response.embeddings.float[0])

