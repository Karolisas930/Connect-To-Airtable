import streamlit as st
import torch
import clip
from PIL import Image
from pyairtable import Table
import pytesseract
import pdf2image
import io
from datetime import datetime

# ---------- Airtable Configuration (from your screenshots) ----------
AIRTABLE_API_KEY = "pat5CxpaVv5dxi1av.4a5d5cf5e911fc12fdab72acddbd64ca02ef53d714fe9d6c0a644b6b7156208"
AIRTABLE_BASE_ID = "appxK32N1zW2Qw6GJ"
AIRTABLE_RESULTS_TABLE = "tblX6BWMPp2PQtpML2" # Job Results
AIRTABLE_CATEGORIES_TABLE = "tbl0vfLoznHT0BjRX" # Job Categories

result_table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_RESULTS_TABLE)
category_table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_CATEGORIES_TABLE)

# ---------- Load CLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- Load Categories from Airtable ----------
st.title("🧠 AI Job Classifier (Image + PDF)")
with st.spinner("🔄 Loading categories from Airtable..."):
    records = category_table.all()
    labels = []
    for r in records:
        cat = r["fields"].get("category")
        subcat = r["fields"].get("subcategory")
        if cat and subcat:
            labels.append(f"{cat}: {subcat}")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📎 Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(uploaded_pdf):
    images = pdf2image.convert_from_bytes(uploaded_pdf.read())
    text = ""
    for img in images:
        text += extract_text_from_image(img) + "\n"
    return text.strip(), images[0] # Return first page image

if uploaded_file:
    filename = uploaded_file.name
    ext = filename.split('.')[-1].lower()

    predicted_label = ""
    preview_img = None
    classification_type = ""

    if ext in ["jpg", "jpeg", "png"]:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_file).convert("RGB")
        preview_img = image

        with st.spinner("🧠 Classifying image..."):
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = clip.tokenize(labels).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                similarity = (image_features @ text_features.T).squeeze(0)
                top_idx = similarity.argmax().item()
                predicted_label = labels[top_idx]
                classification_type = "Image"

    elif ext == "pdf":
        with st.spinner("📖 Extracting text from PDF..."):
            extracted_text, first_page = extract_text_from_pdf(uploaded_file)
            preview_img = first_page
            st.image(first_page, caption="Preview (Page 1)", use_column_width=True)
            st.text_area("📝 Extracted Text", extracted_text, height=200)

            with st.spinner("🔍 Matching text to category..."):
                text_inputs = clip.tokenize(labels).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_inputs)
                    query_text = clip.tokenize([extracted_text[:300]]).to(device)
                    query_features = model.encode_text(query_text)
                    similarity = (query_features @ text_features.T).squeeze(0)
                    top_idx = similarity.argmax().item()
                    predicted_label = labels[top_idx]
                    classification_type = "PDF/Text"

    # ---------- Manual Adjustment ----------
    st.subheader("✅ Suggested Category")
    selected_label = st.selectbox("Confirm or adjust the prediction:", options=labels, index=labels.index(predicted_label))

    # ---------- Save to Airtable ----------
    if st.button("📤 Save to Airtable"):
        data = {
            "Filename": filename,
            "Predicted Label": selected_label,
            "Source": classification_type,
            "Timestamp": datetime.now().isoformat()
        }
        result_table.create(data)
        st.success("Saved to Airtable ✅")
