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
from src.live_data.provider import ConfiguredAPIDataProvider, ComparableAnalyzer

# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(page_title="Real Estate Analytics & Prediction", page_icon="🏢", layout="wide")

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
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-card-sub {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 8px;
    }

    /* Estimate Highlight Cards */
    .estimate-card-ml {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 15px 20px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .estimate-card-live {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 15px 20px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .estimate-card-fallback {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 15px 20px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    .estimate-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 5px;
    }
    .estimate-value {
        font-size: 1.8rem;
        font-weight: 800;
    }
    .estimate-desc {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 5px;
    }
    
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_market_data(city, location, property_type, bhk):
    try:
        provider = ConfiguredAPIDataProvider(api_key=None)
        return provider.fetch_listings(city, location, property_type, bhk)
    except Exception as e:
        return pd.DataFrame()

def format_indian_currency(num):
    if num >= 1_00_00_000:
        return f"₹ {num / 1_00_00_000:.2f} Crore"
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
st.markdown('<div class="sub-title">Mumbai & Indore Property Market | AI-powered price prediction and market estimation</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if df.empty:
    st.error("⚠️ Dataset not found. Please ensure the data pipeline is complete.")
    st.stop()
if model is None or preprocessor is None:
    st.error("⚠️ Prediction model could not be loaded. Please verify the trained model artifacts exist in the `model/` directory.")
    st.stop()

# ==========================================
# NAVIGATION
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎯 Price Prediction", "📊 Market Analysis", "ℹ️ About Model"])

# ==========================================
# TAB 1: PRICE PREDICTION
# ==========================================
with tab1:
    col_input, col_result = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("### Property Information")
        
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
            bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1)
        with row2_2:
            area = st.number_input("Area (sq.ft)", min_value=100.0, max_value=20000.0, value=1000.0, step=50.0)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
            
        st.write("") # Spacer
        predict_btn = st.button("Estimate Property Price", use_container_width=True, type="primary")

    with col_result:
        st.markdown("### Prediction Summary")
        
        if not predict_btn:
            st.info("👈 Enter property details and click **Estimate Property Price** to view the valuation.")
        else:
            with st.spinner("Analyzing historical and current market data..."):
                try:
                    lat = coord_map.get(location, {}).get('Latitude', float('nan'))
                    lon = coord_map.get(location, {}).get('Longitude', float('nan'))
                    
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
                    
                    # 1. Historical ML Prediction
                    input_processed = preprocessor.transform(input_data)
                    ml_prediction = model.predict(input_processed)[0]
                    
                    # 2. Live Market Data
                    live_listings_df = fetch_live_market_data(city, location, property_type, bhk)
                    analyzer = ComparableAnalyzer(live_listings_df)
                    comp_result = analyzer.find_comparables(city, location, property_type, bhk, area)
                    
                    # Layout Results
                    st.markdown(f"""
                    <div class="estimate-card-ml">
                        <div class="estimate-title">HISTORICAL ML ESTIMATE</div>
                        <div class="estimate-value" style="color: #2563EB;">{format_indian_currency(ml_prediction)}</div>
                        <div class="estimate-desc">Based on trained historical property patterns</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if comp_result['status'] == 'SUCCESS':
                        comp_estimate = comp_result['estimate']
                        final_estimate = (ml_prediction + comp_estimate) / 2
                        
                        st.markdown(f"""
                        <div class="estimate-card-live">
                            <div class="estimate-title">CURRENT MARKET ESTIMATE</div>
                            <div class="estimate-value" style="color: #059669;">{format_indian_currency(comp_estimate)}</div>
                            <div class="estimate-desc">Based on recent comparable listings in {location}</div>
                        </div>
                        <div style="text-align: center; margin: 15px 0;">
                            <strong>FINAL ESTIMATE:</strong> <span style="font-size: 1.5rem; color: #0F172A; font-weight: 800;">{format_indian_currency(final_estimate)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif comp_result['status'] == 'LIVE DATA PROVIDER NOT CONFIGURED':
                        st.markdown(f"""
                        <div class="estimate-card-fallback">
                            <div class="estimate-title">CURRENT MARKET DATA: UNAVAILABLE</div>
                            <div class="estimate-value" style="color: #DC2626;">Not Configured</div>
                            <div class="estimate-desc">Reason: Live market provider API key is not configured.</div>
                        </div>
                        <div style="margin-top: 10px;">
                            <em>Final estimate falls back to the historical ML estimate because current comparable market data is unavailable.</em>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="estimate-card-fallback">
                            <div class="estimate-title">CURRENT MARKET DATA: INSUFFICIENT</div>
                            <div class="estimate-value" style="color: #D97706;">No Match</div>
                            <div class="estimate-desc">Reason: No current comparable listings were found for this specific profile.</div>
                        </div>
                        <div style="margin-top: 10px;">
                            <em>Final estimate falls back to the historical ML estimate because current comparable market data is unavailable.</em>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Property Summary Box
                    st.markdown("---")
                    st.markdown(f"**Property Summary:** {bhk} BHK {property_type} in {location}, {city} ({area:,.0f} sq.ft, {bathrooms} baths)")
                    
                except Exception as e:
                    st.error(f"Prediction Error: An unexpected issue occurred during processing. Details: {str(e)}")

    st.markdown("---")
    st.markdown("### Current Market Snapshot")
    
    if predict_btn and 'comp_result' in locals():
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Comparable Listings", comp_result.get('count', 0))
        with col_s2:
            med_price = f"₹{comp_result.get('median_sqft_price', 0):,.0f}" if comp_result.get('median_sqft_price') else "N/A"
            st.metric("Median Price (₹/sq.ft)", med_price)
        with col_s3:
            st.metric("Data Updated", datetime.now().strftime('%H:%M'))
            
        st.caption("Data Source: ConfiguredAPIDataProvider (Development Mock)")
        
        if st.button("🔄 Refresh Market Data", type="secondary"):
            fetch_live_market_data.clear()
            st.rerun()

# ==========================================
# TAB 2: MARKET ANALYSIS
# ==========================================
with tab2:
    st.markdown("### Market Overview")
    
    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        f_city = st.selectbox("Analysis City", ["All"] + sorted(df['City'].dropna().unique().tolist()))
    with f_col2:
        if f_city == "All":
            f_loc = st.selectbox("Analysis Location", ["All"] + sorted(df['Location'].dropna().unique().tolist()))
        else:
            f_loc = st.selectbox("Analysis Location", ["All"] + sorted(df[df['City'] == f_city]['Location'].dropna().unique().tolist()))
    with f_col3:
        f_ptype = st.selectbox("Analysis Property Type", ["All"] + sorted(df['Property_Type'].dropna().unique().tolist()))

    filtered_df = df.copy()
    if f_city != "All": filtered_df = filtered_df[filtered_df['City'] == f_city]
    if f_loc != "All": filtered_df = filtered_df[filtered_df['Location'] == f_loc]
    if f_ptype != "All": filtered_df = filtered_df[filtered_df['Property_Type'] == f_ptype]

    if filtered_df.empty:
        st.warning("No properties match the selected criteria.")
    else:
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Properties", len(filtered_df))
        k2.metric("Median Price", format_indian_currency(filtered_df['Price_INR'].median()))
        k3.metric("Average Price", format_indian_currency(filtered_df['Price_INR'].mean()))
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
                
                st.markdown("**Property Type Comparison**")
                ptype_df = filtered_df.groupby('Property_Type')['Price_INR'].median().reset_index()
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=ptype_df, y='Property_Type', x='Price_INR', hue='Property_Type', palette='viridis', legend=False, ax=ax3)
                ax3.set_xlabel('Median Price (INR)')
                ax3.set_ylabel('')
                st.pyplot(fig3)
                
            with c2:
                if f_city == "All":
                    st.markdown("**Mumbai vs Indore (Median Price)**")
                    comp_df = filtered_df.groupby('City')['Price_INR'].median().reset_index()
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=comp_df, x='City', y='Price_INR', hue='City', palette='Set2', legend=False, ax=ax2)
                    ax2.set_yscale('log')
                    ax2.set_ylabel('Median Price (INR) - Log Scale')
                    st.pyplot(fig2)
                else:
                    st.markdown("**BHK vs Price**")
                    bhk_df = filtered_df.groupby('BHK')['Price_INR'].median().reset_index()
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=bhk_df, x='BHK', y='Price_INR', hue='BHK', palette='magma', legend=False, ax=ax2)
                    ax2.set_ylabel('Median Price (INR)')
                    st.pyplot(fig2)

                st.markdown("**Area vs Price Relationship**")
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=filtered_df, x='Area_sqft', y='Price_INR', hue='BHK', palette='coolwarm', alpha=0.7, ax=ax4)
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.set_xlabel('Area (sq.ft)')
                ax4.set_ylabel('Price (INR)')
                st.pyplot(fig4)
                
        except Exception as e:
            st.error(f"Failed to render charts: {str(e)}")

# ==========================================
# TAB 3: ABOUT MODEL
# ==========================================
with tab3:
    st.markdown("### System Architecture & Methodology")
    
    st.markdown("""
    #### Overview
    This platform estimates real estate valuations using a dual-architecture approach, successfully isolating the extreme economic disparities between Tier-1 (Mumbai) and Tier-2 (Indore) cities.
    
    #### 1. Historical ML Prediction
    A robust predictive model trained on a curated historical dataset of 440 properties. 
    It evaluates spatial parameters (Latitude/Longitude), physical dimensions, and configuration.
    
    **Models Evaluated:**
    * Linear Regression (Selected Best Model)
    * Random Forest Regressor
    * Decision Tree Regressor
    * Polynomial Regression
    """)
    
    if metrics:
        st.markdown("#### Selected Model Evaluation (Test Data)")
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        met_col1.metric("Algorithm", best_model_name)
        met_col2.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
        met_col3.metric("MAE", f"₹{metrics.get('MAE', 0) / 1_00_00_000:.2f} Cr")
        met_col4.metric("RMSE", f"₹{metrics.get('RMSE', 0) / 1_00_00_000:.2f} Cr")
        
    st.markdown("""
    ---
    #### 2. Current Market Estimate
    To adjust for recent market volatility and inflation, the application queries an API provider abstraction (`ConfiguredAPIDataProvider`) in real-time. 
    
    It executes a **3-Tier Hierarchical Match** to find the most relevant current comparable listings:
    1. Exact Match (City, Location, Type, BHK, Area)
    2. Relaxed Area Match
    3. Broad Location Match
    
    If the API is unavailable or unconfigured, the system safely triggers a **100% Fallback Protocol** to the Historical ML Prediction to ensure uninterrupted user service.
    """)
