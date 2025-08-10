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
                                 model="gpt-4.1-mini", verbose=True, context_cache: list = None) -> str:
    """
    Sends a multimodal prompt (text + multiple images + recent context) to the LLM and returns the answer.
    """
    try:
        # Encode each image to base64 and build image_url blocks
        image_contents = []
        for img_path in matched_paths:
            b64 = encode_image_to_base64(img_path)
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        # 🧠 Build context from previous Q&A
        context_text = ""
        if context_cache:
            for i, item in enumerate(context_cache[-4:]):
                context_text += f"Previous Q{i+1}: {item['question']}\n"
                context_text += f"Answer: {item['answer']}\n\n"

        # 📝 Build prompt with context + current question
        prompt_text = f"""
You are a helpful assistant answering questions about World Bank trust fund reports based on images and prior discussion.

{f"Recent context:\n{context_text}" if context_text else ""}
Now answer this question: {question}
""".strip()

        # 👤 Build message content
        message_content = [{"type": "text", "text": prompt_text}] + image_contents

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

