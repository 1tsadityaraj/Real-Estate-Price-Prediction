import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(page_title="Real Estate Analytics", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    /* Typography & Core Styling */
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
        padding-top: 1rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 400;
        color: #64748B;
        margin-bottom: 30px;
    }
    
    /* Card Layouts */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    /* Estimate Highlight Card */
    .estimate-card-ml {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 20px 25px;
        border-radius: 6px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .estimate-title {
        font-size: 1rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .estimate-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .estimate-desc {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 10px;
    }
    
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border: 0;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS & CACHING
# ==========================================
def format_indian_currency(num):
    if num >= 1_00_00_000:
        return f"₹ {num / 1_00_00_000:.2f} Lakhs / ₹ {num / 1_00_00_000:.2f} Crore"
    elif num >= 1_00_000:
        return f"₹ {num / 1_00_000:.2f} Lakhs"
    else:
        return f"₹ {num:,.0f}"

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model/best_model.pkl')
        preprocessor = joblib.load('model/preprocessor.pkl')
        return model, preprocessor
    except Exception:
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/processed/geocoded_mumbai_indore_property_data.csv')
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_model_info():
    try:
        with open('model_summary.json', 'r') as f:
            data = json.load(f)
            best_model_name = data.get('best_model', 'Unknown')
            metrics = next((item for item in data.get('results', []) if item["Model"] == best_model_name), None)
            return best_model_name, metrics
    except Exception:
        return "Unknown", None

# Load artifacts safely
model, preprocessor = load_artifacts()
df = load_data()
best_model_name, metrics = get_model_info()

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-title">REAL ESTATE ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mumbai & Indore Property Price Prediction<br>Machine-learning-based property price prediction using historical real-estate data</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if df.empty:
    st.error("⚠️ Dataset not found. Please ensure the data pipeline is complete and historical data is available.")
    st.stop()
if model is None or preprocessor is None:
    st.error("⚠️ Prediction model is unavailable. Please verify the trained model artifacts exist in the `model/` directory.")
    st.stop()

# ==========================================
# NAVIGATION
# ==========================================
tab1, tab2, tab3 = st.tabs(["🏠 Price Prediction", "📊 Market Analysis", "🤖 About Model"])

# ==========================================
# TAB 1: PRICE PREDICTION
# ==========================================
with tab1:
    col_input, col_result = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("### PROPERTY DETAILS")
        
        # City Filter
        cities = sorted(df['City'].dropna().unique().tolist())
        city = st.selectbox("City", cities, key="pred_city")
        
        city_df = df[df['City'] == city]
        locations = sorted(city_df['Location'].dropna().unique().tolist())
        coord_map = city_df.drop_duplicates(subset=['Location']).set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
        property_types = sorted(df['Property_Type'].dropna().unique().tolist())
        
        location = st.selectbox("Location", locations, key="pred_loc")
        
        row2_1, row2_2 = st.columns(2)
        with row2_1:
            property_type = st.selectbox("Property Type", property_types, key="pred_ptype")
            bhk = st.number_input("BHK / Bedrooms", min_value=1, max_value=10, value=2, step=1)
        with row2_2:
            area = st.number_input("Area (sq.ft.)", min_value=100.0, max_value=20000.0, value=1000.0, step=50.0)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
            
        st.write("") # Spacer
        predict_btn = st.button("ESTIMATE PROPERTY PRICE", use_container_width=True, type="primary")

    with col_result:
        
        if not predict_btn:
            st.info("👈 Enter property details to estimate its value.")
        else:
            with st.spinner("Processing historical property data..."):
                try:
                    lat = coord_map.get(location, {}).get('Latitude', float('nan'))
                    lon = coord_map.get(location, {}).get('Longitude', float('nan'))
                    
                    if pd.isna(lat) or pd.isna(lon):
                        st.warning("Please select a location available in the historical dataset.")
                        st.stop()
                        
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
                    
                    # Historical ML Prediction
                    input_processed = preprocessor.transform(input_data)
                    ml_prediction = model.predict(input_processed)[0]
                    
                    # Layout Results
                    st.markdown(f"""
                    <div class="estimate-card-ml">
                        <div class="estimate-title">ESTIMATED PROPERTY PRICE</div>
                        <div class="estimate-value">{format_indian_currency(ml_prediction)}</div>
                        <div class="estimate-desc">
                            <strong>{best_model_name}</strong><br>
                            Based on historical property data
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction Details
                    st.markdown("#### Prediction Details")
                    st.markdown(f"""
                    * **City:** {city}
                    * **Location:** {location}
                    * **Property Type:** {property_type}
                    * **BHK:** {bhk}
                    * **Area:** {area:,.0f} sq.ft.
                    * **Bathrooms:** {bathrooms}
                    """)
                    
                    st.markdown("---")
                    st.markdown("#### HOW THIS PREDICTION IS GENERATED")
                    st.markdown("""
                    The prediction is generated using a machine-learning model trained on historical property data from Mumbai and Indore.
                    
                    The model learns relationships between location, city, property type, BHK, area, bathrooms, and historical property prices.
                    """)
                    st.caption("*This value is an ML-based estimate derived from historical property data. Actual property prices may vary depending on market conditions, property condition, amenities, exact location, negotiation, and other factors.*")
                    
                except Exception as e:
                    st.error(f"Prediction Error: An unexpected issue occurred during processing. Please verify inputs.")

# ==========================================
# TAB 2: MARKET ANALYSIS
# ==========================================
with tab2:
    st.markdown("### HISTORICAL PROPERTY MARKET ANALYSIS")
    
    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        f_city = st.selectbox("City filter", ["All Cities"] + sorted(df['City'].dropna().unique().tolist()))
        
    filtered_df = df.copy()
    if f_city != "All Cities": 
        filtered_df = filtered_df[filtered_df['City'] == f_city]

    if filtered_df.empty:
        st.warning("No properties match the selected criteria.")
    else:
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Number of Properties", len(filtered_df))
        k2.metric("Median Price", f"₹ {filtered_df['Price_INR'].median() / 1_00_000:.2f} Lakhs")
        k3.metric("Average Price", f"₹ {filtered_df['Price_INR'].mean() / 1_00_000:.2f} Lakhs")
        k4.metric("Average Area", f"{filtered_df['Area_sqft'].mean():,.0f} sq.ft")
        
        st.markdown("---")
        
        # Charts - Wrapped in Try-Except for safety
        try:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Price Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(filtered_df['Price_INR'], kde=True, ax=ax, color='#3B82F6')
                ax.set_xscale('log')
                ax.set_xlabel('Price (INR) - Log Scale')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                
                st.markdown("**Property Type vs Average Price**")
                ptype_df = filtered_df.groupby('Property_Type')['Price_INR'].mean().reset_index()
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=ptype_df, y='Property_Type', x='Price_INR', hue='Property_Type', palette='viridis', legend=False, ax=ax3)
                ax3.set_xlabel('Average Price (INR)')
                ax3.set_ylabel('')
                st.pyplot(fig3)
                
            with c2:
                st.markdown("**Area vs Price Relationship**")
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=filtered_df, x='Area_sqft', y='Price_INR', hue='BHK', palette='coolwarm', alpha=0.7, ax=ax4)
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.set_xlabel('Area (sq.ft)')
                ax4.set_ylabel('Price (INR)')
                st.pyplot(fig4)
                
                st.markdown("**BHK vs Average Price**")
                bhk_df = filtered_df.groupby('BHK')['Price_INR'].mean().reset_index()
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=bhk_df, x='BHK', y='Price_INR', hue='BHK', palette='magma', legend=False, ax=ax2)
                ax2.set_ylabel('Average Price (INR)')
                st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"Failed to render charts: {str(e)}")

# ==========================================
# TAB 3: ABOUT MODEL
# ==========================================
with tab3:
    st.markdown("### ABOUT THE MODEL")
    
    st.markdown("""
    #### Project Objective
    To estimate residential property prices in Mumbai and Indore using machine-learning models trained on historical real-estate data.
    
    #### Models Evaluated
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor
    * Polynomial Regression
    """)
    
    st.markdown("---")
    st.markdown("#### Best Model")
    st.info(f"**{best_model_name}** was automatically selected as the best performing model based on test-set metrics.")
    
    if metrics:
        st.markdown("#### Model Performance (Test Metrics)")
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        met_col1.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
        met_col2.metric("MAE", f"₹ {metrics.get('MAE', 0) / 1_00_00_000:.2f} Cr")
        met_col3.metric("RMSE", f"₹ {metrics.get('RMSE', 0) / 1_00_00_000:.2f} Cr")
        met_col4.metric("MSE", f"{metrics.get('MSE', 0):.2e}")
        
    st.markdown("""
    ---
    #### Features Used
    The model processes the following features to generate predictions:
    - `City` (Categorical)
    - `Location` (Categorical)
    - `Property_Type` (Categorical)
    - `BHK` (Numerical)
    - `Area_sqft` (Numerical)
    - `Bathrooms` (Numerical)
    - `Latitude` (Numerical - Geocoded spatial parameter)
    - `Longitude` (Numerical - Geocoded spatial parameter)
    
    *Note: Leakage features such as `Price_per_sqft` are strictly excluded from the final prediction pipeline.*
    
    #### Training Data
    The machine learning models are trained purely on a processed historical dataset containing **440** real-estate records sourced across Mumbai and Indore.
    """)
