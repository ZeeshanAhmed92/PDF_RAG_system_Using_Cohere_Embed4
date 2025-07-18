import os
import streamlit as st
from dotenv import load_dotenv








from config import HASHES_FOLDER, PDF_HASH_FILE, IMG_EMB_FILE, PDF_FOLDER, IMG_FOLDER, MODEL_NAME, co, Client
from utils import load_json, hash_file, convert_pdf_to_images, embed_image, save_json, retry, base64_from_image



os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(HASHES_FOLDER, exist_ok=True)

load_dotenv()