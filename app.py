import os
import traceback
import shutil
import base64
import time
import cohere
import tempfile
import mimetypes
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from utils import load_json, hash_file, save_json
import streamlit.components.v1 as components
from pdf_processing_embedding import process_pdfs_and_embed_pages
from vision_query import search_image_by_question, answer_question_about_images
from chat_history import generate_session_id, save_chat_history, load_chat_history, CHAT_DATA_DIR
from config import HASHES_FOLDER, PDF_HASH_FILE, PDF_FOLDER, IMG_FOLDER, co

# Ensure paths exist
HASHES_FOLDER.mkdir(parents=True, exist_ok=True)
PDF_FOLDER.parent.mkdir(parents=True, exist_ok=True)
IMG_FOLDER.parent.mkdir(parents=True, exist_ok=True)


load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_hash_path = os.path.join(HASHES_FOLDER, PDF_HASH_FILE)
file_hashes = load_json(pdf_hash_path)


# Initialize or restore session
if "chat_id" not in st.session_state:
    st.session_state.chat_id = generate_session_id()

current_chat_path = CHAT_DATA_DIR / f"{st.session_state.chat_id}.json"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(current_chat_path)

# UI
st.set_page_config("Multimodal RAG")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #ffffff; font-weight: bold; margin-top: 10px;'>
        📄 Multimodal RAG on World Bank Trust Funds Reports
    </h1>
""", unsafe_allow_html=True)


# ✅ Background Image Function (Full Page, Clean)
def set_background(image_path):
    if not os.path.exists(image_path):
        st.warning("⚠️ Background image not found.")
        return

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <style>
        html, body, .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .stApp {{
            backdrop-filter: blur(0px);
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Apply background
set_background("world_bank.jpg")

# 📁 Custom Header in Sidebar (bold + color)
st.sidebar.markdown(
    "<h3 style='color: #000840; font-weight: bold;'>📁 Upload PDF</h3>",
    unsafe_allow_html=True
)

# 📄 Custom Label for File Uploader (bold + color)
st.sidebar.markdown(
    "<span style='font-weight: bold; color: #000840;'>Choose a PDF file</span>",
    unsafe_allow_html=True
)

# Hide default label
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file:
    try:
        # ✅ Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = Path(tmp_file.name)

        # ✅ Check file size
        if temp_path.stat().st_size < 1000:
            st.sidebar.error("❌ Uploaded file is too small or corrupted.")
            temp_path.unlink(missing_ok=True)
        else:
            filename = uploaded_file.name
            file_stem = Path(filename).stem
            file_hash = hash_file(temp_path)

            if file_hashes.get(file_stem) == file_hash:
                st.sidebar.success("✅ Already processed.")
                temp_path.unlink(missing_ok=True)
            else:
                st.sidebar.info("🔄 Processing...")

                # ✅ Move to final path
                final_path = PDF_FOLDER / filename
                shutil.move(str(temp_path), str(final_path))

                # ✅ Process PDF
                process_pdfs_and_embed_pages(co, specific_pdf_path=final_path)

                # ✅ Save hash
                file_hashes[file_stem] = file_hash
                save_json(pdf_hash_path, file_hashes)

                st.sidebar.success("✅ File processed.")

    except Exception as e:
        st.sidebar.error(f"❌ Failed: {e}")
        st.sidebar.code(traceback.format_exc(), language="python")

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# RAG Q&A
# ✅ Styled subheader
st.markdown("""
    <h3 style='font-weight: 700; color: #ffffff;'>
        🔍 Ask a Question
    </h3>
""", unsafe_allow_html=True)

# ✅ Styled label for input
st.markdown("""
    <label style='font-weight: 700; color: #ffffff; font-size: 1rem;'>
        Enter your question
    </label>
""", unsafe_allow_html=True)

# Add this CSS before your input box
st.markdown("""
    <style>
    /* Style the placeholder text */
    input::placeholder {
        color: #000840 !important; /* Light blue */
        opacity: 1; /* Firefox */
    }

    /* Style the actual input text */
    input[type="text"] {
        color: #000840 !important; /* Darker blue for input text */
        font-weight: 500;
        border: 2px solid #000840 !important;
        border-radius: 6px;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Render the input box (no label)
question = st.text_input("Ask a question", placeholder="Type your question here...", key="question_input", label_visibility="collapsed")

st.markdown("""
    <style>
    .custom-spinner {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 10px;
    }

    .lds-dual-ring {
        display: inline-block;
        width: 24px;
        height: 24px;
    }

    .lds-dual-ring:after {
        content: " ";
        display: block;
        width: 24px;
        height: 24px;
        margin: 1px;
        border-radius: 50%;
        border: 4px solid #28adfe;
        border-color: #28adfe transparent #28adfe transparent;
        animation: lds-dual-ring 1.2s linear infinite;
    }

    @keyframes lds-dual-ring {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    .thinking-text {
        font-weight: bold;
        color: #000840;
        font-size: 1.05rem;
    }

    .custom-answer {
        background-color: white;
        color: #000840;
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 1rem;
        border-left: 4px solid #28adfe;
    }

    div.stButton > button {
        background-color: #000840 !important;
        color: white !important;
        font-weight: 700;
        border: 3px solid #28adfe !important;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #28adfe !important;
        color: #000840 !important;
        border-color: #000840 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Button handler
if question and st.button("💬 Get Answer"):
    with st.container():
        spinner_html = """<div class="custom-spinner"><div class="lds-dual-ring"></div><div class="thinking-text">Thinking...</div></div>"""
        spinner_slot = st.empty()
        spinner_slot.markdown(spinner_html, unsafe_allow_html=True)

        try:
            img_paths = search_image_by_question(question, co)

            # ✅ Ensure it's a list
            if not isinstance(img_paths, list):
                raise TypeError(f"Expected list of image paths, got: {type(img_paths)} → {img_paths}")

            answer = answer_question_about_images(question, img_paths, client)

            # Store to session
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "images": img_paths
            })
            save_chat_history(st.session_state.chat_history, current_chat_path)
            spinner_slot.empty()

        except Exception as e:
            spinner_slot.empty()
            st.error(f"❌ Error: {str(e)}")


if st.sidebar.button("🆕 Start New Chat"):
    # Save current chat before resetting
    if st.session_state.get("chat_history"):
        old_path = CHAT_DATA_DIR / f"{st.session_state.chat_id}.json"
        save_chat_history(st.session_state.chat_history, old_path)

    # Reset session
    st.session_state.chat_id = generate_session_id()
    st.session_state.chat_history = []

    st.rerun()

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

user_icon = get_image_base64("boy.png")
bot_icon = get_image_base64("assistant.png")

# Build chat HTML dynamically
chat_html = """
<style>
.scroll-box {
    max-height: 500px;
    overflow-y: auto;
    padding: 12px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.7);
    backdrop-filter: blur(4px);
    border: 1px solid #ccc;
    margin-top: 1rem;
}

.chat-message {
    display: flex;
    margin-bottom: 12px;
    align-items: flex-start;
}

.chat-message.user {
    justify-content: flex-end;
}

.chat-message.bot {
    justify-content: flex-start;
}

.avatar {
    width: 32px;
    height: 32px;
    margin: 0 8px;
    border-radius: 50%;
}

.bubble {
    padding: 12px 16px;
    border-radius: 18px;
    max-width: 70%;
    font-size: 0.95rem;
    line-height: 1.4;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.bot .bubble {
    background-color: white;
    color: #000;
    border-top-left-radius: 4px;
}

.user .bubble {
    background-color: #d0e8ff;
    color: #000840;
    border-top-right-radius: 4px;
}
</style>

<div class="scroll-box">
"""

# Build chat HTML
for pair in reversed(st.session_state.chat_history):
    # User message (right-aligned)
    chat_html += f"""
    <div class="chat-message user">
        <div class="bubble">{pair['question']}</div>
        <img class="avatar" src="data:image/png;base64,{user_icon}" />
    </div>
    """

    # Bot message (left-aligned)
    chat_html += f"""
    <div class="chat-message bot">
        <img class="avatar" src="data:image/png;base64,{bot_icon}" />
        <div class="bubble">{pair['answer']}</div>
    </div>
    """

# Render chat using components (this respects scrolling)
components.html(chat_html + "</div>", height=520, scrolling=False)

# 📸 Display images in a separate Streamlit container
st.markdown("""
    <h3 style='font-weight: 700; color: #ffffff;'>
        🖼️ Relevant Images
    </h3>
""", unsafe_allow_html=True)

# Build HTML/CSS/JS content
image_html = """
<style>
.image-scroll-box {
    max-height: 580px;
    overflow-y: auto;
    padding: 12px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.7);
    backdrop-filter: blur(4px);
    border: 1px solid #ccc;
    margin-top: 1rem;
}

.image-block {
    margin-bottom: 24px;
}

.image-title {
    font-weight: 700;
    color: #000840;
    margin-bottom: 8px;
    font-size: 1rem;
}

.image-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.image-card {
    text-align: center;
    width: 200px;
    word-break: break-word;
}

.image-card img {
    height: 240px;
    width: 100%;
    border-radius: 10px;
    object-fit: cover;
    border: 2px solid #28adfe;
    cursor: pointer;
    transition: transform 0.2s;
}

.image-card img:hover {
    transform: scale(1.05);
}

.image-name {
    font-size: 0.95rem;
    margin-top: 8px;
    color: #000840;
    font-weight: 600;
}

/* Modal Styles */
.modal {
    display: none;
    position: fixed;
    z-index: 999;
    padding-top: 40px;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.8);
}

.modal-content {
    margin: auto;
    display: block;
    max-width: 90%;
    max-height: 90%;
    border: 4px solid white;
    border-radius: 12px;
}

.close {
    position: fixed;
    top: 20px;
    right: 40px;
    color: white;
    font-size: 40px;
    font-weight: bold;
    cursor: pointer;
    z-index: 1000;
}
</style>

<!-- Modal Structure -->
<div id="imageModal" class="modal">
    <span class="close" onclick="document.getElementById('imageModal').style.display='none'">&times;</span>
    <img class="modal-content" id="modalImage">
</div>

<div class="image-scroll-box">
"""

# Build image grid HTML
for idx, pair in enumerate(reversed(st.session_state.chat_history)):
    if pair.get("images"):
        image_html += f"""
        <div class="image-block">
            <div class="image-title">🔹 Q{len(st.session_state.chat_history)-idx}: {pair['question']}</div>
            <div class="image-grid">
        """
        for img_path in pair["images"]:
            if os.path.exists(img_path):
                img_name = os.path.basename(img_path)
                with open(img_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                    image_html += f"""
                    <div class="image-card">
                        <img src="data:image/png;base64,{img_base64}" onclick="openModal(this.src)">
                        <div class="image-name">{img_name}</div>
                    </div>
                    """

        image_html += "</div></div>"

image_html += "</div>"

# JavaScript to open modal
image_html += """
<script>
function openModal(src) {
    var modal = document.getElementById("imageModal");
    var modalImg = document.getElementById("modalImage");
    modal.style.display = "block";
    modalImg.src = src;
}
</script>
"""

# Show final HTML with modal
components.html(image_html, height=620, scrolling=False)

