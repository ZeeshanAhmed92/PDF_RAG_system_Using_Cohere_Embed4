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
IMG_EMB_FILE = "image_embeddings.json"
MODEL_NAME = "embed-v4.0"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(HASHES_FOLDER, exist_ok=True)

# Load API keys
load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))