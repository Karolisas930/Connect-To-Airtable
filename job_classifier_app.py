import streamlit as st
import torch
import clip
from PIL import Image
from pyairtable import Table
import pytesseract
import pdf2image
from datetime import datetime
import pandas as pd

# Airtable Setup
AIRTABLE_API_KEY = "patvOXAPPzVeUU1tK.b0b961530230189b5a9bf5cd82a846b830dcefe285b93f86c402c0106b4c02cd"
AIRTABLE_BASE_ID = "appxK32N1zW2Qw6Cj"
AIRTABLE_RESULTS_TABLE = "Section"
AIRTABLE_CATEGORIES_TABLE = "Catogarie"
AIRTABLE_SUBCATEGORIES_TABLE = "Subcatogarie"

result_table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_RESULTS_TABLE)
category_table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_CATEGORIES_TABLE)
subcategory_table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_SUBCATEGORIES_TABLE)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def truncate_label(label, max_length=512):
    return label[:max_length]

# Load categories and subcategories from Airtable
@st.cache_data
def load_labels():
    categories = []
    for r in category_table.all():
        name = r.get("fields", {}).get("Category_Name_EN", "")
        if name:
            categories.append(truncate_label(name))

    subcategories = []
    for r in subcategory_table.all():
        name = r.get("fields", {}).get("Subcategory_Name_EN", "")
        if name:
            subcategories.append(truncate_label(name))

    return categories, subcategories

categories, subcategories = load_labels()
category_tokens = clip.tokenize(categories).to(device)
subcategory_tokens = clip.tokenize(subcategories).to(device)

# UI
st.title("🧠 AI Job Classifier (Multi-file Image/PDF Support)")

uploaded_files = st.file_uploader("Upload job description files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

results = []

def extract_first_image(file):
    if file.type == "application/pdf":
        return pdf2image.convert_from_bytes(file.read())[0]
    return Image.open(file)

def classify_content(image):
    text = pytesseract.image_to_string(image)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        text_embed = model.encode_text(clip.tokenize([text]).to(device))
        image_embed = model.encode_image(image_input)

        combined_embed = (text_embed + image_embed) / 2

        cat_probs = (combined_embed @ model.encode_text(category_tokens).T).softmax(dim=-1)
        subcat_probs = (combined_embed @ model.encode_text(subcategory_tokens).T).softmax(dim=-1)

        top_cat = categories[cat_probs.argmax().item()]
        top_subcat = subcategories[subcat_probs.argmax().item()]

    return text, top_cat, top_subcat

if uploaded_files:
    for file in uploaded_files:
        image = extract_first_image(file)
        st.image(image, caption=f"Preview: {file.name}", use_column_width=True)

        with st.spinner("Classifying..."):
            text, cat, subcat = classify_content(image)

        st.markdown(f"**Predicted Category:** `{cat}`")
        st.markdown(f"**Predicted Subcategory:** `{subcat}`")

        results.append({
            "Filename": file.name,
            "Text": text[:300],
            "Category": cat,
            "Subcategory": subcat,
            "Timestamp": datetime.now().isoformat()
        })

        result_table.create({
            "Text": text,
            "Predicted_Category": cat,
            "Predicted_Subcategory": subcat,
            "Timestamp": datetime.now().isoformat()
        })

    df = pd.DataFrame(results)
    st.success("✅ All files processed and saved to Airtable!")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", data=csv, file_name="job_results.csv", mime="text/csv")
