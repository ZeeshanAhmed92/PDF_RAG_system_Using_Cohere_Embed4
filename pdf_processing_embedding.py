from config import HASHES_FOLDER, PDF_HASH_FILE, PDF_FOLDER, IMG_FOLDER
from utils import load_json, hash_file, convert_pdf_to_images, embed_image, save_json
from faiss_utils import load_faiss_index, save_faiss_index, add_embedding
from pathlib import Path
from tqdm import tqdm
import os

def process_pdfs_and_embed_pages(co, specific_pdf_path: Path = None):
    pdf_hash_path = os.path.join(HASHES_FOLDER, PDF_HASH_FILE)
    pdf_hashes = load_json(pdf_hash_path)

    index, filenames = load_faiss_index()
    new_embeddings = 0

    pdf_files = [specific_pdf_path] if specific_pdf_path else [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    for pdf_path in tqdm(pdf_files):
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        current_hash = hash_file(pdf_path)

        if pdf_name in pdf_hashes and pdf_hashes[pdf_name] == current_hash:
            print(f"✅ Skipping (unchanged): {pdf_path.name}")
            continue

        print(f"🔄 Processing: {pdf_path.name}")
        image_paths = convert_pdf_to_images(pdf_path, IMG_FOLDER)

        for img_path in image_paths:
            img_name = os.path.basename(img_path)

            if img_name in filenames:
                continue

            emb = embed_image(co, img_path)
            add_embedding(index, filenames, emb, img_name)
            new_embeddings += 1

        pdf_hashes[pdf_name] = current_hash

    save_faiss_index(index, filenames)
    save_json(pdf_hash_path, pdf_hashes)

    print(f"\n✅ Total FAISS entries: {index.ntotal}")
    print(f"🆕 New embeddings added: {new_embeddings}")
