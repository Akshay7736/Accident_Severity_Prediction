import streamlit as st
import pandas as pd
import joblib
import os

import base64

def set_bg_gif(gif_path):
    with open(gif_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: none;
        }}
        .stApp::before {{
            content: "";
            background: url(data:image/gif;base64,{encoded}) no-repeat center center fixed;
            background-size: cover;
            filter: blur(2px) brightness(0.9);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# --- Paths ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
MAPPING_PATH = os.path.join(ROOT, "models", "label_mappings.pkl")
PROCESSED_DATA_PATH = os.path.join(ROOT, "data", "processed", "Accident_Severity_no_outliers.csv")

# --- Load model and data ---
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    mappings = joblib.load(MAPPING_PATH)
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    return model, mappings, processed_df

model, mappings, processed_df = load_assets()

# --- Streamlit Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="centered")
set_bg_gif("assets/traffic_bg.jpg")
st.title("🚦 Accident Severity Prediction App")

st.markdown("""
This app predicts the **severity** of a road accident 
(`Slight`, `Serious`, or `Fatal`) based on the given road and environmental conditions.
""")

# --- Helper functions ---
def get_dropdown_options(df, column):
    return sorted(df[column].dropna().unique())

# --- Input Fields ---
st.header("🧭 Enter Accident Details")

col1, col2 = st.columns(2)
with col1:
    day_of_week = st.selectbox("Day of Week", get_dropdown_options(processed_df, "Day_of_Week"))
    time = st.selectbox("Time", get_dropdown_options(processed_df, "Time"))
    road_type = st.selectbox("Road Type", get_dropdown_options(processed_df, "Road_Type"))
    weather = st.selectbox("Weather Conditions", get_dropdown_options(processed_df, "Weather_Conditions"))

with col2:
    light = st.selectbox("Light Conditions", get_dropdown_options(processed_df, "Light_Conditions"))
    area = st.selectbox("Urban or Rural Area", get_dropdown_options(processed_df, "Urban_or_Rural_Area"))
    district = st.selectbox("Local Authority (District)", get_dropdown_options(processed_df, "Local_Authority_(District)"))
    junction = st.selectbox("Junction Detail", get_dropdown_options(processed_df, "Junction_Detail"))

# Numeric inputs
col3, col4 = st.columns(2)
with col3:
    num_vehicles = st.number_input("Number of Vehicles", min_value=1, value=1)
    casualties = st.number_input("Number of Casualties", min_value=1, value=1)
with col4:
    speed_limit = st.number_input("Speed Limit (mph)", min_value=10, max_value=80, step=10, value=30)

vehicle_type = st.selectbox("Vehicle Type", get_dropdown_options(processed_df, "Vehicle_Type"))

# --- Prepare input for model ---
feature_order = [c for c in processed_df.columns if c != "Accident_Severity"]

input_data = {
    "Day_of_Week": day_of_week,
    "Time": time,
    "Road_Type": road_type,
    "Weather_Conditions": weather,
    "Light_Conditions": light,
    "Urban_or_Rural_Area": area,
    "Local_Authority_(District)": district,
    "Junction_Detail": junction,
    "Number_of_Vehicles": num_vehicles,
    "Number_of_Casualties": casualties,
    "Speed_limit": speed_limit,
    "Vehicle_Type": vehicle_type
}

# Fill missing features (if any)
for col in feature_order:
    if col not in input_data:
        input_data[col] = processed_df[col].mode()[0] if col in processed_df.columns else 0

input_df = pd.DataFrame([input_data], columns=feature_order)

# --- Severity label mapping (adjust if different in your dataset) ---
severity_mapping = {0: "Slight", 1: "Serious", 2: "Fatal"}

# --- Prediction Button ---
if st.button("🔮 Predict Accident Severity"):
    try:
        enc_df = input_df.copy()

        # Apply label encodings
        for col, mapping in mappings.items():
            if col in enc_df.columns:
                enc_df[col] = enc_df[col].astype(str).map(lambda x: mapping.get(x, mapping.get("_unknown_", 0)))

        # Predict severity
        pred = model.predict(enc_df)[0]
        predicted_label = severity_mapping.get(pred, pred)

        st.success(f"### 🧾 Predicted Severity: **{predicted_label}**")

        # --- Show probabilities (if supported) ---
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(enc_df)[0]
            prob_df = pd.DataFrame({
                "Severity": [severity_mapping.get(cls, cls) for cls in model.classes_],
                "Probability": [float(f"{p*100:.2f}") for p in probs]
            })

            st.subheader("Prediction Confidence")
            st.table(prob_df.astype(str))  # ✅ Fix for Arrow serialization

            # Horizontal bar chart for confidence
            st.bar_chart(prob_df.set_index("Severity")["Probability"])

        # --- Show input summary ---
        with st.expander("See Input Details"):
            st.table(input_df.T.astype(str).rename(columns={0: "Value"}))  # ✅ Fix for Arrow serialization

    except Exception as e:
        st.error(f"Error during prediction: {e}")
