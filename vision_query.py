import os
import base64
import numpy as np
from PIL import Image
from openai import OpenAI
from embeddings import get_query_embedding, load_image_embeddings

def search_image_by_question(question: str, co, max_size=800,
                             embeddings_path="hashes/image_embeddings.json",
                             image_folder="images",
                             n_results=3) -> list:
    """
    Embeds the query, searches for the top-N most similar image pages,
    and returns the matched image paths.
    """
    # Step 1: Get query embedding
    query_emb = get_query_embedding(question, co)

    # Step 2: Load embeddings and filenames
    embeddings, filenames = load_image_embeddings(embeddings_path)

    # Step 3: Compute cosine similarity (dot product for normalized vectors)
    scores = np.dot(query_emb, embeddings.T)
    top_indices = np.argsort(scores)[::-1][:n_results]

    matched_paths = []
    for idx in top_indices:
        img_path = os.path.join(image_folder, filenames[idx])

        # Display image (optional)
        img = Image.open(img_path)
        img.thumbnail((max_size, max_size))
        img.show()

        matched_paths.append(img_path)

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
