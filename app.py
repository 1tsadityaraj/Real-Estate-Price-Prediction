import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #2E86C1;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #5D6D7E;
        margin-bottom: 30px;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        color: #27AE60;
        text-align: center;
        padding: 20px;
        background-color: #EAFAF1;
        border-radius: 10px;
        border: 2px solid #2ECC71;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching Data & Models ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model/best_model.pkl')
        preprocessor = joblib.load('model/preprocessor.pkl')
        
        # Load unique locations and property types to populate the UI dropdowns
        df = pd.read_csv('data/processed/geocoded_property_data.csv')
        locations = df['Location'].dropna().unique().tolist()
        property_types = df['Property_Type'].dropna().unique().tolist()
        
        # Build a coordinate mapping for locations
        coord_map = df.drop_duplicates(subset=['Location']).set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
        
        return model, preprocessor, locations, property_types, coord_map
    except Exception as e:
        st.error(f"Error loading required files: {e}")
        return None, None, [], [], {}

def format_indian_currency(num):
    # Simple formatter for Crores/Lakhs
    if num >= 1_00_00_000:
        return f"₹ {num / 1_00_00_000:.2f} Crore"
    elif num >= 1_00_000:
        return f"₹ {num / 1_00_000:.2f} Lakhs"
    else:
        return f"₹ {num:,.0f}"

# Load artifacts
model, preprocessor, locations, property_types, coord_map = load_artifacts()

# --- Main App ---
st.markdown('<div class="main-title">REAL ESTATE PRICE PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter property details to estimate the expected real estate price.</div>', unsafe_allow_html=True)

if model is None or preprocessor is None:
    st.warning("Application is down: Missing model artifacts. Please ensure Step 7 is completed.")
    st.stop()

# --- Input Form ---
with st.container():
    st.subheader("Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox("Location", sorted(locations))
        bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        
    with col2:
        property_type = st.selectbox("Property Type", sorted(property_types))
        area = st.number_input("Area (sq.ft)", min_value=100.0, max_value=20000.0, value=1000.0, step=50.0)

# --- Prediction Action ---
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])

with predict_col2:
    predict_btn = st.button("Predict Property Price", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("Calculating estimate..."):
        try:
            # 1. Fetch Coordinates for selected location
            lat = coord_map.get(location, {}).get('Latitude', 0)
            lon = coord_map.get(location, {}).get('Longitude', 0)
            
            # 2. Build input dataframe exactly as expected by the preprocessor
            input_data = pd.DataFrame([{
                'Location': location,
                'Property_Type': property_type,
                'Area_sqft': area,
                'BHK': bhk,
                'Bathrooms': bathrooms,
                'Latitude': lat,
                'Longitude': lon
            }])
            
            # 3. Apply preprocessing
            input_processed = preprocessor.transform(input_data)
            
            # 4. Predict
            prediction = model.predict(input_processed)[0]
            
            # Display Result
            st.markdown("### ESTIMATED PRICE")
            st.markdown(f'<div class="price-display">{format_indian_currency(prediction)}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Expandable Info Sections ---
st.markdown("---")
with st.expander("ℹ️ About the Project"):
    st.write("""
    This application uses Machine Learning to estimate real-estate property prices based on historical property data.
    
    **Algorithms evaluated during development:**
    * Linear Regression
    * Decision Tree Regression
    * Random Forest Regression
    * Polynomial Regression
    """)
    st.write(f"**Model Used:** Linear Regression")
    st.write(f"**R² Score:** -0.237 *(Demonstration dataset metric)*")
    st.write("""
    **Limitations:** This model was trained on a highly randomized subset of proxy real estate data for academic demonstration. The accuracy represents the relationships in that specific random sample.
    """)
