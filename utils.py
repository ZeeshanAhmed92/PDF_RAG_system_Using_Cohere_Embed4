import os
import time
import json
import base64
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import mimetypes
from config import MODEL_NAME
from pdf2image import convert_from_path

# ==== HELPERS ====
def hash_file(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def base64_from_image(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(img_path)
    if not mime_type:
        mime_type = "image/png"  # fallback
    return f"data:{mime_type};base64,{b64_data}"



def embed_image(co, img_path: str):
    api_input_document = {
        "content": [{"type": "image", "image": base64_from_image(img_path)}]
    }
    api_response = co.embed(
        model=MODEL_NAME,
        input_type="search_document",
        embedding_types=["float"],
        inputs=[api_input_document],
    )
    return np.asarray(api_response.embeddings.float[0])


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list:
    pdf_name = Path(pdf_path).stem
    images = convert_from_path(pdf_path, dpi=200)
    image_paths = []

    for i, img in enumerate(images):
        img_filename = f"{pdf_name}_page{i + 1}.png"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths


def load_json(path: str) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}


def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def retry(retries=3, backoff=2):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            delay = backoff
            for attempt in range(retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed — retrying in {delay}s: {e}")
                    time.sleep(delay)
                    delay *= 2
            raise RuntimeError("All retries failed")
        return wrapper
    return decorator