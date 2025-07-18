import os
from tqdm import tqdm
from pathlib import Path
from config import HASHES_FOLDER, PDF_HASH_FILE, IMG_EMB_FILE, PDF_FOLDER, IMG_FOLDER
from utils import load_json, hash_file, convert_pdf_to_images, embed_image, save_json

# ==== MAIN PIPELINE ====
def process_pdfs_and_embed_pages(co):
    pdf_hash_path = os.path.join(HASHES_FOLDER, PDF_HASH_FILE)
    img_emb_path = os.path.join(HASHES_FOLDER, IMG_EMB_FILE)

    pdf_hashes = load_json(pdf_hash_path)
    image_embeddings = load_json(img_emb_path)

    new_embeddings = 0

    for pdf_file in tqdm(os.listdir(PDF_FOLDER)):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        pdf_name = Path(pdf_file).stem
        current_hash = hash_file(pdf_path)

        if pdf_name in pdf_hashes and pdf_hashes[pdf_name] == current_hash:
            print(f"✅ Skipping (unchanged): {pdf_file}")
            continue

        # New or changed PDF
        print(f"🔄 Processing: {pdf_file}")
        image_paths = convert_pdf_to_images(pdf_path, IMG_FOLDER)

        for img_path in image_paths:
            img_name = os.path.basename(img_path)

            if img_name in image_embeddings:
                continue  # image already embedded (unlikely for new PDF)

            emb = embed_image(co, img_path)
            image_embeddings[img_name] = emb.tolist()
            new_embeddings += 1

        # Save new hash
        pdf_hashes[pdf_name] = current_hash

    # Save updated JSONs
    save_json(pdf_hash_path, pdf_hashes)
    save_json(img_emb_path, image_embeddings)

    print(f"\n✅ Total embeddings stored: {len(image_embeddings)}")
    print(f"🆕 New embeddings added: {new_embeddings}")