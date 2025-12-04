# streamlit_app.py
import streamlit as st
import pickle
import os
from datetime import datetime

st.set_page_config(page_title="Wheat Disease Predictor", layout="centered")

st.title("🌾 Wheat Disease Predictor")

BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "wheat_model.pkl")

if not os.path.exists(MODEL_FILE):
    st.error(
        "Model file wheat_model.pkl not found. "
        "Please add wheat_model.pkl to the repository root (next to streamlit_app.py)."
    )
    st.stop()

# Load model safely
try:
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
except Exception as e:
    st.error("Failed to load model file (pickle). See exception details below.")
    st.exception(e)
    st.stop()

# Expecting dict with keys: 'model', 'stage_encoder', 'disease_encoder'
try:
    model = data["model"]
    stage_encoder = data["stage_encoder"]
    disease_encoder = data["disease_encoder"]
except Exception as e:
    st.error("Model file does not contain expected keys ('model','stage_encoder','disease_encoder').")
    st.exception(e)
    st.stop()

REFERENCE_DATE = datetime(2025, 12, 2)

# UI
st.markdown("Enter crop stage and sowing date. The app will compute days-since-sowing and predict disease.")

stage_options = list(stage_encoder.classes_)
crop_stage = st.selectbox("Crop Stage", stage_options)

sowing_date = st.date_input("Sowing Date", value=REFERENCE_DATE)

if st.button("Predict"):
    try:
        days_since = (REFERENCE_DATE - datetime.strptime(str(sowing_date), "%Y-%m-%d")).days
        stage_enc = stage_encoder.transform([crop_stage])[0]
        pred_enc = model.predict([[stage_enc, days_since]])[0]
        disease = disease_encoder.inverse_transform([pred_enc])[0]
        st.success(f"### Predicted Disease: **{disease}**")
    except Exception as e:
        st.error("Prediction failed — see details:")
        st.exception(e)
