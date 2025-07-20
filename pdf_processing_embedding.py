import os
from tqdm import tqdm
from pathlib import Path
from config import HASHES_FOLDER, PDF_HASH_FILE, IMG_EMB_FILE, PDF_FOLDER, IMG_FOLDER
from utils import load_json, hash_file, convert_pdf_to_images, embed_image, save_json

# ==== MAIN PIPELINE ====
def process_pdfs_and_embed_pages(co, specific_pdf_path: Path = None):
    pdf_hash_path = os.path.join(HASHES_FOLDER, PDF_HASH_FILE)
    img_emb_path = os.path.join(HASHES_FOLDER, IMG_EMB_FILE)

    pdf_hashes = load_json(pdf_hash_path)
    image_embeddings = load_json(img_emb_path)

    new_embeddings = 0

    # Only process a specific PDF if provided
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
            if img_name in image_embeddings:
                continue

            emb = embed_image(co, img_path)
            image_embeddings[img_name] = emb.tolist()
            new_embeddings += 1

        pdf_hashes[pdf_name] = current_hash

    save_json(pdf_hash_path, pdf_hashes)
    save_json(img_emb_path, image_embeddings)

    print(f"\n✅ Total embeddings stored: {len(image_embeddings)}")
    print(f"🆕 New embeddings added: {new_embeddings}")
