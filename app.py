import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import numpy as np
from streamlit_folium import folium_static
import google.generativeai as genai

# ========== GEMINI CONFIGURATION ==========
genai.configure(api_key=st.secrets["gemini"]["api_key"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ========== PAGE TITLE ==========
st.set_page_config(page_title="Urban Heat Index Prediction", layout="wide")
st.title('🌆 Urban Heat Index (UHI) Prediction')
st.markdown("""
    This application predicts the **Urban Heat Index** using ML models and displays results on an interactive map.

    **Features used for prediction:**
    - NDVI (Vegetation)
    - Albedo (Surface reflectivity)
    - NDBI (Built-up Index)
    - Interactions: LST_NDBI, NDVI_NDBI
    - Log(Population Density)
""")

# ========== SIDEBAR INPUTS ==========
st.sidebar.header('🔍 Prediction Input')
latitude = st.sidebar.number_input("Latitude", min_value=28.0, max_value=29.5, value=28.6448, step=0.0001)
longitude = st.sidebar.number_input("Longitude", min_value=76.8, max_value=77.5, value=77.216721, step=0.0001)

# ========== DUMMY MODEL (Replace with real model later) ==========
def dummy_predict(lat, lon):
    return round(np.random.uniform(0.3, 0.8), 2)

uhi_prediction = dummy_predict(latitude, longitude)

# ========== PREDICTION DISPLAY ==========
st.subheader("📍 Prediction Result")
st.success(f"Predicted UHI at ({latitude}, {longitude}): **{uhi_prediction}**")

# ========== MAP DISPLAY ==========
st.subheader("🗺️ Location on Map")
m = folium.Map(location=[latitude, longitude], zoom_start=12)
folium.Marker([latitude, longitude], popup=f"UHI: {uhi_prediction}").add_to(m)

# Delhi bounding box
delhi_coordinates = [
    [28.4, 76.8],
    [28.9, 76.8],
    [28.9, 77.5],
    [28.4, 77.5]
]
folium.Polygon(locations=delhi_coordinates, color='blue', fill=True, fill_opacity=0.1).add_to(m)

folium_static(m)

# ========== GEMINI INTEGRATION ==========
st.sidebar.markdown("---")
st.sidebar.header("🤖 Ask Gemini")
user_prompt = st.sidebar.text_area("Ask about UHI, satellite data, climate analysis...")

if st.sidebar.button("Ask Gemini"):
    if user_prompt.strip():
        with st.spinner("Gemini is thinking..."):
            response = gemini_model.generate_content(user_prompt)
            st.subheader("🔎 Gemini's Response")
            st.write(response.text)
    else:
        st.sidebar.warning("Please enter a prompt first.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("Built using Earth Engine, Gemini AI & Streamlit 🌍")
st.markdown("Developed by **[Hitesh]** · [GitHub](https://github.com/hiteshgupta23)")
