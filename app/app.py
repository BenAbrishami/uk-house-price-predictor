import streamlit as st
import joblib
import pandas as pd

# Load the model and feature names
model = joblib.load('models/house_price_model.pkl')
features = joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="UK Property Price Predictor", layout="centered")
st.title("🏡 UK Property Price Predictor")
st.markdown("Estimate UK property prices based on location and basic attributes.")

# User inputs
postcode_area = st.text_input("Enter Postcode Area (e.g. E1, NW1, SW3)").upper()
property_type = st.selectbox("Property Type", ["F", "S", "T", "D", "O"])  # Flat, Semi, Terraced, Detached, Other
new_build = st.selectbox("Is it a new build?", ["Yes", "No"])
freehold = st.selectbox("Is it freehold?", ["Yes", "No"])
year = st.slider("Year of Sale", 2015, 2024, 2023)
month = st.slider("Month of Sale", 1, 12, 6)

if st.button("Predict Price"):
    input_dict = {
        "year": year,
        "month": month,
        "new_build": 1 if new_build == "Yes" else 0,
        "freehold": 1 if freehold == "Yes" else 0,
    }

    # One-hot encoding for postcode_area
    for col in features:
        if col.startswith("postcode_area_"):
            input_dict[col] = 1 if col == f"postcode_area_{postcode_area}" else 0

    # One-hot encoding for property_type
    for col in features:
        if col.startswith("property_type_"):
            input_dict[col] = 1 if col == f"property_type_{property_type}" else 0

    # Fill any other feature with 0
    for col in features:
        if col not in input_dict:
            input_dict[col] = 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])[features]


    # Warning for unknown postcode area
    if f"postcode_area_{postcode_area}" not in features:
        st.warning("⚠️ This postcode area was not in the training set. Prediction may be less accurate.")

    # Predict
    price = model.predict(input_df)[0]
    st.success(f"💷 Estimated Price: £{int(price):,}")
