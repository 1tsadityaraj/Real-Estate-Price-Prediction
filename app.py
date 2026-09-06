import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

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
        
        # Load unique locations and property types from the combined geocoded data
        df = pd.read_csv('data/processed/geocoded_mumbai_indore_property_data.csv')
        
        return model, preprocessor, df
    except Exception as e:
        st.error(f"Error loading required files: {e}")
        return None, None, pd.DataFrame()

@st.cache_data
def get_model_info():
    try:
        with open('model_summary.json', 'r') as f:
            data = json.load(f)
            best_model_name = data.get('best_model', 'Unknown')
            # Extract metrics for best model
            metrics = next((item for item in data.get('results', []) if item["Model"] == best_model_name), None)
            return best_model_name, metrics
    except Exception:
        return "Unknown", None

def format_indian_currency(num):
    # Simple formatter for Crores/Lakhs
    if num >= 1_00_00_000:
        return f"₹ {num / 1_00_00_000:.2f} Crore"
    elif num >= 1_00_000:
        return f"₹ {num / 1_00_000:.2f} Lakhs"
    else:
        return f"₹ {num:,.0f}"

# Load artifacts
model, preprocessor, df = load_artifacts()
best_model_name, metrics = get_model_info()

# --- Main App ---
st.markdown('<div class="main-title">REAL ESTATE PRICE PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict property prices for Mumbai and Indore using Machine Learning.</div>', unsafe_allow_html=True)

if model is None or preprocessor is None or df.empty:
    st.warning("Application is down: Missing model artifacts or dataset.")
    st.stop()

# --- Input Form ---
with st.container():
    st.subheader("Property Details")
    
    # Extract unique cities
    cities = sorted(df['City'].dropna().unique().tolist())
    city = st.selectbox("City", cities)
    
    # Dynamically filter locations based on selected city
    city_df = df[df['City'] == city]
    locations = sorted(city_df['Location'].dropna().unique().tolist())
    
    # Coordinate Map for Latitude/Longitude specific to the selected city and location
    coord_map = city_df.drop_duplicates(subset=['Location']).set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
    property_types = sorted(df['Property_Type'].dropna().unique().tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox("Location", locations)
        bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        
    with col2:
        property_type = st.selectbox("Property Type", property_types)
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
                'City': city,
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
            st.markdown("### Prediction Result")
            st.markdown(f'<div class="price-display">{format_indian_currency(prediction)}</div>', unsafe_allow_html=True)
            
            st.write(f"**Based on:** {bhk} BHK {property_type} in {location}, {city} ({area} sq.ft.)")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Expandable Info Sections ---
st.markdown("---")

with st.expander("📊 Market Analysis"):
    st.write(f"### {city} Market Overview")
    st.write("Overview based on the dataset used to train the model.")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Total Properties Analyzed", len(city_df))
    with col_b:
        st.metric("Median Price", format_indian_currency(city_df['Price_INR'].median()))
    with col_c:
        st.metric("Average Area (sq.ft)", f"{city_df['Area_sqft'].mean():,.0f}")
        
with st.expander("⚖️ Mumbai vs Indore Comparison"):
    st.write("### Cross-City Analytics")
    comp_df = df.groupby('City').agg(
        Total_Properties=('Price_INR', 'count'),
        Median_Price=('Price_INR', 'median'),
        Avg_Area_Sqft=('Area_sqft', 'mean'),
    ).reset_index()
    
    comp_df['Median_Price_Fmt'] = comp_df['Median_Price'].apply(format_indian_currency)
    comp_df['Avg_Area_Sqft'] = comp_df['Avg_Area_Sqft'].round(0)
    st.dataframe(comp_df[['City', 'Total_Properties', 'Median_Price_Fmt', 'Avg_Area_Sqft']], use_container_width=True)

with st.expander("ℹ️ About the Model"):
    st.write(f"**Algorithm Used:** {best_model_name}")
    if metrics:
        st.write("### Model Evaluation Metrics (Test Set)")
        st.write(f"- **R² Score:** {metrics.get('R2', 0):.4f}")
        st.write(f"- **MAE:** ₹ {metrics.get('MAE', 0):,.2f}")
        st.write(f"- **RMSE:** ₹ {metrics.get('RMSE', 0):,.2f}")
    else:
        st.write("Evaluation metrics are currently unavailable.")
