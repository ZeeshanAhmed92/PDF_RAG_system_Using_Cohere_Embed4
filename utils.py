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
    img_path = Path(img_path)
    with img_path.open("rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(img_path.name)
    if not mime_type:
        mime_type = "image/png"
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
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem
    images = convert_from_path(str(pdf_path), dpi=200)
    image_paths = []

    for i, img in enumerate(images):
        img_filename = output_dir / f"{pdf_name}_page{i + 1}.png"
        img.save(img_filename, "PNG")
        image_paths.append(str(img_filename))

    return image_paths


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


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