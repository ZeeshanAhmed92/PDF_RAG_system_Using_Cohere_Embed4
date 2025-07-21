import os
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from openai import OpenAI
from utils import embed_image
from faiss_utils import load_faiss_index, normalize

def search_image_by_question(question, co, top_k=4):
    # Embed the question correctly
    response = co.embed(
        texts=[question],
        input_type="search_query",
        model="embed-v4.0"
    )
    query_emb = response.embeddings.float[0]  # ✅ Correct access

    index, filenames = load_faiss_index()
    norm_query = normalize(np.array(query_emb)).astype("float32")
    
    D, I = index.search(norm_query[np.newaxis, :], top_k)
    matched_paths = [str(Path("images") / filenames[i]) for i in I[0] if i < len(filenames)]
    print("📂 matched_paths:", matched_paths)
    return matched_paths

def encode_image_to_base64(img_path: str) -> str:
    """Encodes an image to base64 for embedding in a prompt."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def answer_question_about_images(question: str, matched_paths: list, client: OpenAI,
                                 model="gpt-4.1-mini", verbose=True) -> str:
    """
    Sends a multimodal prompt (text + multiple images) to the LLM and returns the answer.

    Parameters:
    - question (str): User query
    - matched_paths (list): List of local image paths
    - client: OpenAI or AzureOpenAI client
    - model (str): Model to use (e.g., gpt-4.1-mini, gpt-4o)
    - verbose (bool): Whether to print the response

    Returns:
    - response text
    """
    try:
        # Encode each image to base64 and build image_url blocks
        image_contents = []
        for img_path in matched_paths:
            b64 = encode_image_to_base64(img_path)
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        # Combine the text question and the images
        message_content = [{"type": "text", "text": f"Answer clearly: {question}"}] + image_contents

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message_content},
            ],
            max_tokens=1000,
        )

        answer_text = response.choices[0].message.content.strip()
        if verbose:
            print("🧠 LLM Response:", answer_text)

        return answer_text

    except Exception as e:
        print(f"❌ Error processing images or getting response: {e}")
        return "Error occurred during processing."
