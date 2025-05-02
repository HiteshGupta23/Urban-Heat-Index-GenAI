import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import numpy as np
import cloudpickle
import joblib
from google.generativeai import GenerativeModel
import plotly.graph_objects as go
import requests
import plotly.express as px
from streamlit_lottie import st_lottie

# -------------------------------
# 1. Page Configuration & Lottie
# -------------------------------
st.set_page_config(page_title="Urban Heat Island Predictor", layout="wide")
st.title("🌆 Urban Heat Island (UHI) Predictor")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_uhi = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_0fhlytwe.json")
if lottie_uhi:
    st_lottie(lottie_uhi, height=250, key="uhi_anim")

# ----------------------
# 2. Introduction
# ----------------------
with st.markdown("📘 What is the Urban Heat Island Effect?"):
    st.markdown("""
    The **Urban Heat Island (UHI)** effect refers to the observed phenomenon where urban or metropolitan areas experience significantly warmer temperatures than nearby rural areas.

    ### 🌍 Why It Matters:
    - Impacts public health, especially during heatwaves
    - Increases energy consumption and greenhouse gas emissions
    - Reduces urban livability

    ### 🚧 Causes:
    - Excessive use of concrete and asphalt
    - Loss of vegetation (low NDVI)
    - High population density

    ### 🎯 Goal of this App:
    Use machine learning and AI to **predict UHI severity** and provide **sustainable planning recommendations**.
    """)

# ----------------------
# 3. Load ML Model & Scaler
# ----------------------
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------
# 4. Input Section
# ----------------------
st.sidebar.header("🔢 Feature Inputs")

NDVI = st.sidebar.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.4,
                         help="Higher NDVI = more vegetation. Helps reduce UHI.")
Albedo = st.sidebar.slider("Albedo (Reflectivity)", 0.0, 1.0, 0.2,
                           help="Higher albedo = reflects more sunlight, reduces heat.")
CO = st.sidebar.number_input("Carbon Monoxide (mol/m²)", 0.0, 1.0, 0.2, step=0.01)
Ozone = st.sidebar.number_input("Ozone (mol/m²)", 0.0, 1.0, 0.1, step=0.01)
NO2 = st.sidebar.number_input("NO₂ (mol/m²)", 0.0, 1.0, 0.1, step=0.01)
PopDensity = st.sidebar.number_input("Population Density (people/km²)", 0, 50000, 1000, step=100)

with st.sidebar.expander("📚 Feature Definitions"):
    st.markdown("""
    - **NDVI**: Normalized Difference Vegetation Index. Measures vegetation cover.
    - **Albedo**: Surface reflectivity. High values reflect sunlight.
    - **CO, NO₂, Ozone**: Air quality indicators affecting urban temperature.
    - **Population Density**: Number of people per square km.
    """)

# ----------------------
# 5. Model Prediction
# ----------------------
st.header("🧠 UHI Prediction & Risk Analysis")

user_input = pd.DataFrame([[NDVI, Albedo, NO2, Ozone, CO, PopDensity]],
                          columns=['NDVI', 'Albedo', 'NO2', 'Ozone', 'CO', 'Population_Density'])

input_scaled = scaler.transform(user_input)
pred_LST = rf_model.predict(input_scaled)[0]

st.metric("🌡️ Predicted Land Surface Temp (LST)", f"{pred_LST:.2f} °C")

if pred_LST > 35:
    st.error("🚨 High UHI Risk! Consider urgent mitigation measures.")
elif pred_LST > 30:
    st.warning("⚠️ Moderate UHI Risk. Consider improvements.")
else:
    st.success("✅ Low UHI Risk. Conditions are relatively stable.")

# Optional animated chart
fig = px.bar(user_input.T, labels={"index": "Feature", "value": "Value"}, title="Environmental Feature Overview")
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# 6. Gemini API Integration
# ----------------------
st.header("🤖 Ask AI About UHI")

user_q = st.text_input("Ask a question about urban heat islands, sustainability, or your prediction")

if user_q:
    valid_topics = ["uhi", "heat", "climate", "sustain", "vegetation", "temperature", "urban"]
    if any(keyword in user_q.lower() for keyword in valid_topics):
        with st.spinner("AI is generating a response..."):
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": user_q}]}],
                "generationConfig": {"temperature": 0.7}
            }
            response = requests.post(f"{endpoint}?key={GEMINI_API_KEY}", json=data, headers=headers)
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.info(answer)
            else:
                st.error("Failed to fetch AI response.")
    else:
        st.info("❗ This assistant is focused on the Urban Heat Island project. Please ask topic-related questions.")

# ----------------------
# 7. Summary & Feedback
# ----------------------
st.header("📊 Your Inputs Recap")
st.dataframe(user_input)

st.markdown("### 🔍 Key Indicators")
st.write(f"- NDVI: {'✅ Good' if NDVI > 0.4 else '⚠️ Low vegetation'}")
st.write(f"- Albedo: {'✅ Reflective surface' if Albedo > 0.3 else '🟥 Absorbs heat'}")
st.write(f"- Population Density: {'🔺 Dense Area' if PopDensity > 5000 else '✅ Moderate/Low'}")

# ----------------------
# 8. Sidebar: Tools & About
# ----------------------
st.sidebar.markdown("### 🛠️ Tools Used")
st.sidebar.write("- Streamlit\n- Scikit-learn\n- Gemini API\n- Pandas, NumPy\n- Plotly\n- Lottie")

with st.sidebar.expander("👤 About the Creator"):
    st.markdown("""
    **Hitesh Gupta**  
    Data Science & Sustainability Enthusiast 🌿  
    Built this project to promote awareness and solutions to urban climate issues.  
    [GitHub](https://github.com/yourprofile) | [LinkedIn](https://linkedin.com/in/yourprofile)
    """)

