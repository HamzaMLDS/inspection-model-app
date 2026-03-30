import streamlit as st
import pandas as pd
import joblib
import os
import re
import json
from openai import OpenAI

# =========================
# LOAD MODEL (SAFE)
# =========================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "inspection_model.pkl")
    return joblib.load(model_path)

model = load_model()

# =========================
# OPENAI CLIENT (SAFE)
# =========================
api_key = st.secrets.get("OPENAI_API_KEY", None)

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None

# =========================
# DOMAIN WORDS (OIL RIG)
# =========================
DOMAIN_WORDS = [
    "crosby", "shackle", "derrick", "swl", "kn", "bolt", "nut",
    "rig", "hook", "chain", "sling", "drill", "crown", "block",
    "lifting", "inspection", "load", "pin", "bow", "anchor"
]

# =========================
# SPELL CHECK FUNCTION
# =========================
def check_spelling(text):
    if client is None:
        return text, True, False  # corrected, is_correct, is_gibberish

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a strict spell checker for oil rig inspection data.

Rules:
1. Preserve original text exactly except spelling fixes.
2. Do NOT rephrase.
3. Do NOT change formatting.
4. PRIORITIZE oil rig and lifting terms like:
{", ".join(DOMAIN_WORDS)}

5. If text contains random/gibberish words with no meaning,
mark it as gibberish.

Return ONLY JSON:
{{
"corrected": "corrected text",
"is_correct": true/false,
"is_gibberish": true/false
}}
"""
                },
                {"role": "user", "content": text}
            ]
        )

        result = response.choices[0].message.content
        parsed = json.loads(result)

        corrected = parsed.get("corrected", text)
        is_correct = parsed.get("is_correct", False)
        is_gibberish = parsed.get("is_gibberish", False)

        # Safety check
        if corrected.strip() != text.strip():
            is_correct = False

        return corrected, is_correct, is_gibberish

    except:
        return text, False, False


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Inspection Predictor", layout="centered")

st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🔍 Inspection Status Predictor</h1>
<p style='text-align: center;'>Enter item details to classify inspection status</p>
""", unsafe_allow_html=True)

st.divider()

# =========================
# INPUT SECTION
# =========================
st.subheader("📋 Input Details")

use_ai = st.checkbox("🧠 Enable AI Spell Check", value=True)

col1, col2 = st.columns(2)

with col1:
    swl_input = st.text_input("⚖️ SWL", placeholder="e.g. 3.25 t, 30 kN")
    location = st.text_input("📍 Location", placeholder="e.g. Derrick Under Crown")

with col2:
    manufacture = st.text_input("🏭 Manufacturer", placeholder="e.g. Crosby")

description = st.text_area("📝 Description", placeholder="Enter full item description...")

st.divider()

# =========================
# SWL PROCESSING
# =========================
def process_swl(swl):
    try:
        swl = str(swl).lower().strip()
        match = re.search(r"\d+\.?\d*", swl)

        if match:
            value = float(match.group())
        else:
            return 0.0

        if "kn" in swl:
            value = value / 9.81

        return float(value)

    except:
        return 0.0


# =========================
# VALIDATION FUNCTION
# =========================
def validate_field(label, text):
    if not use_ai or text.strip() == "":
        return text, True

    corrected, is_correct, is_gibberish = check_spelling(text)

    # 🚨 GIBBERISH
    if is_gibberish:
        st.error(f"🚫 {label} contains invalid/gibberish text. No valid words detected.")
        return text, False

    # 🚨 SPELLING ISSUE
    if not is_correct:
        st.markdown(f"🔴 **{label} has spelling issues**")
        st.markdown(f"👉 Suggested: `{corrected}`")

    return corrected, is_correct


# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Status", use_container_width=True):

    if description.strip() == "":
        st.warning("⚠️ Description is required")

    else:
        # SPELL CHECK
        location_corr, loc_ok = validate_field("Location", location)
        manufacture_corr, man_ok = validate_field("Manufacturer", manufacture)
        description_corr, desc_ok = validate_field("Description", description)

        all_correct = loc_ok and man_ok and desc_ok

        # ❌ STOP IF ISSUES
        if not all_correct:
            st.error("🚫 Please fix spelling issues before prediction.")
            st.stop()

        try:
            # PROCESS SWL
            swl_value = process_swl(swl_input)
            st.write(f"🔢 Processed SWL value: {round(swl_value, 3)}")

            # MODEL INPUT
            input_data = pd.DataFrame({
                "SWL": [swl_value],
                "Location": [location_corr.lower().strip()],
                "Description": [description_corr.lower().replace("\n", " ")],
                "Manufacture": [manufacture_corr.lower().strip()]
            })

            prediction = model.predict(input_data)[0]

            try:
                probs = model.predict_proba(input_data)[0]
                confidence = round(max(probs) * 100, 2)
            except:
                confidence = None

            st.divider()
            st.subheader("📊 Prediction Result")

            if prediction.lower() == "accepted":
                st.success(f"✅ Status: {prediction}")
            elif prediction.lower() in ["quarantinne", "quarantine"]:
                st.warning(f"⚠️ Status: {prediction}")
            else:
                st.error(f"❌ Status: {prediction}")

            if confidence:
                st.info(f"🔢 Confidence: {confidence}%")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with AI + ML 🚀</p>", unsafe_allow_html=True)
