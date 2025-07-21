import os
import cohere
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


# ==== CONFIG ====
PDF_FOLDER = Path("source_docs")
IMG_FOLDER = Path("images")
HASHES_FOLDER = Path("hashes")
PDF_HASH_FILE = "pdf_hashes.json"
FAISS_INDEX_PATH = Path("store/image_index.faiss")  # for FAISS index
FILENAME_MAP_PATH = Path("store/image_filenames.pkl")  # for image path -> index mapping
MODEL_NAME = "embed-v4.0"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(HASHES_FOLDER, exist_ok=True)

# Load API keys
load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))