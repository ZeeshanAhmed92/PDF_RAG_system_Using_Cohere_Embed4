import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from config import FAISS_INDEX_PATH, FILENAME_MAP_PATH

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def load_faiss_index():
    if FAISS_INDEX_PATH.exists() and FILENAME_MAP_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FILENAME_MAP_PATH, "rb") as f:
            filenames = pickle.load(f)
    else:
        # Load dynamic dimension based on Cohere embed model
        dummy_vec = np.zeros(1536, dtype="float32")  # Change 1024 to actual size if needed
        index = faiss.IndexFlatIP(len(dummy_vec))
        filenames = []

    return index, filenames


def save_faiss_index(index, filenames):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FILENAME_MAP_PATH, "wb") as f:
        pickle.dump(filenames, f)

def add_embedding(index, filenames, embedding, image_name):
    norm_embedding = normalize(embedding).astype("float32")
    index.add(norm_embedding[np.newaxis, :])
    filenames.append(image_name)
