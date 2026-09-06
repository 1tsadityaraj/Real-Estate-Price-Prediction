import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.live_data.provider import ConfiguredAPIDataProvider, ComparableAnalyzer

@st.cache_data(ttl=3600)
def fetch_live_market_data(city, location, property_type, bhk):
    # Retrieve current listings from the provider abstraction
    # By default this will return an empty DataFrame as no commercial API key is configured
    provider = ConfiguredAPIDataProvider(api_key=None)
    return provider.fetch_listings(city, location, property_type, bhk)

def format_indian_currency(num):
    if num >= 1_00_00_000:
        return f"₹ {num / 1_00_00_000:.2f} Crore"
    elif num >= 1_00_000:
        return f"₹ {num / 1_00_000:.2f} Lakhs"
    else:
        return f"₹ {num:,.0f}"

# Load artifacts
model, preprocessor = load_artifacts()
df = load_data()
best_model_name, metrics = get_model_info()

# --- Main App ---
st.markdown('<div class="main-title">REAL ESTATE ANALYTICS & PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mumbai and Indore Property Markets</div>', unsafe_allow_html=True)

if df.empty:
    st.error("Dataset not found. Please ensure the data pipeline is complete.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🏠 Property Price Prediction", "📊 Market Analysis Dashboard", "ℹ️ About Model"])

# ==========================================
# TAB 1: PREDICTION
# ==========================================
with tab1:
    st.subheader("Estimate Property Price")
    if model is None or preprocessor is None:
        st.warning("Model artifacts are missing. Prediction is unavailable.")
    else:
        with st.container():
            # Extract unique cities
            cities = sorted(df['City'].dropna().unique().tolist())
            col_city, col_empty = st.columns(2)
            with col_city:
                city = st.selectbox("Select City", cities, key="pred_city")
            
            # Dynamically filter locations based on selected city
            city_df = df[df['City'] == city]
            locations = sorted(city_df['Location'].dropna().unique().tolist())
            
            # Coordinate Map
            coord_map = city_df.drop_duplicates(subset=['Location']).set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
            property_types = sorted(df['Property_Type'].dropna().unique().tolist())
            
            col1, col2 = st.columns(2)
            with col1:
                location = st.selectbox("Location", locations, key="pred_loc")
                bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1, key="pred_bhk")
                bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1, key="pred_bath")
                
            with col2:
                property_type = st.selectbox("Property Type", property_types, key="pred_ptype")
                area = st.number_input("Area (sq.ft)", min_value=100.0, max_value=20000.0, value=1000.0, step=50.0, key="pred_area")

        # Prediction Action
        st.markdown("---")
        predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
        with predict_col2:
            predict_btn = st.button("Predict Property Price", use_container_width=True, type="primary")

        if predict_btn:
            with st.spinner("Calculating estimate..."):
                try:
                    lat = coord_map.get(location, {}).get('Latitude', float('nan'))
                    lon = coord_map.get(location, {}).get('Longitude', float('nan'))
                    

                    if not city or not location or not property_type:
                        st.warning("Please enter valid property details before generating a prediction.")
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
                    
                    input_processed = preprocessor.transform(input_data)
                    prediction = model.predict(input_processed)[0]
                    
                    st.markdown("### Current Market Estimate")
                    
                    # Fetch live market data
                    live_listings_df = fetch_live_market_data(city, location, property_type, bhk)
                    
                    # Analyze comparables
                    analyzer = ComparableAnalyzer(live_listings_df)
                    comp_result = analyzer.find_comparables(city, location, property_type, bhk, area)
                    
                    st.markdown("#### Comparison Summary")
                    
                    if comp_result['status'] == 'SUCCESS':
                        comp_estimate = comp_result['estimate']
                        final_estimate = (prediction + comp_estimate) / 2 # 50/50 weighting for now
                        comp_display = format_indian_currency(comp_estimate)
                        final_display = format_indian_currency(final_estimate)
                        status_msg = f"Current comparable listings: {comp_result['count']} | Median ₹/sq.ft.: ₹{comp_result['median_sqft_price']:,.2f}"
                    elif comp_result['status'] == 'LIVE DATA PROVIDER NOT CONFIGURED':
                        comp_estimate = None
                        final_estimate = prediction
                        comp_display = "Not available (API Not Configured)"
                        final_display = format_indian_currency(final_estimate)
                        status_msg = "Current comparable listings: 0"
                    else:
                        comp_estimate = None
                        final_estimate = prediction
                        comp_display = "Not available (Insufficient listings)"
                        final_display = format_indian_currency(final_estimate)
                        status_msg = f"Current comparable listings: {comp_result['count']}"
                        
                    # Display the estimate table
                    st.markdown(f"""
                    | Estimate Type | Value |
                    | --- | --- |
                    | Historical ML Estimate | **{format_indian_currency(prediction)}** |
                    | Current Comparable Estimate | **{comp_display}** |
                    | **Final Market Estimate** | **{final_display}** |
                    """)
                    
                    st.info(f"{status_msg} | Data retrieved: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
                    
                    if st.button("🔄 Refresh Current Market Data"):
                        fetch_live_market_data.clear()
                        st.rerun()

                    st.markdown("#### Prediction Context")
                    st.markdown(f"""
                    * **Model Used:** `{best_model_name}`
                    * **City:** {city}
                    * **Location:** {location}
                    * **Property Type:** {property_type}
                    * **BHK:** {bhk}
                    * **Area:** {area} sq.ft.
                    """)
                    
                    st.markdown("#### Current Data Source")
                    st.markdown(f"""
                    * **Provider:** ConfiguredAPIDataProvider (Mock)
                    * **Retrieval Time:** {datetime.now().strftime('%d-%m-%Y %H:%M')}
                    * **Search Location:** {city}, {location}
                    * **Listings Retrieved:** {comp_result['count']}
                    """)
                    
                    st.warning("**Important Note**: This is an estimated market value based on current property listings and historical machine-learning data. Listing prices are asking prices and may differ from actual negotiated or registered transaction prices. This estimate is not a legal property valuation or guaranteed sale price.")


                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# ==========================================
# TAB 2: MARKET ANALYSIS DASHBOARD
# ==========================================
with tab2:
    st.subheader("Market Analysis")
    
    # --- Filters ---
    st.markdown("##### Filter Data")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        f_city = st.selectbox("City", ["All"] + sorted(df['City'].dropna().unique().tolist()))
    with f_col2:
        f_ptype = st.selectbox("Property Type", ["All"] + sorted(df['Property_Type'].dropna().unique().tolist()))
    with f_col3:
        bhk_options = ["All"] + sorted(df['BHK'].dropna().unique().astype(str).tolist())
        f_bhk = st.selectbox("BHK", bhk_options)
    with f_col4:
        # Dynamic location based on City filter
        if f_city == "All":
            loc_options = ["All"] + sorted(df['Location'].dropna().unique().tolist())
        else:
            loc_options = ["All"] + sorted(df[df['City'] == f_city]['Location'].dropna().unique().tolist())
        f_loc = st.selectbox("Location", loc_options)

    # Apply Filters
    filtered_df = df.copy()
    if f_city != "All":
        filtered_df = filtered_df[filtered_df['City'] == f_city]
    if f_ptype != "All":
        filtered_df = filtered_df[filtered_df['Property_Type'] == f_ptype]
    if f_bhk != "All":
        filtered_df = filtered_df[filtered_df['BHK'] == int(f_bhk)]
    if f_loc != "All":
        filtered_df = filtered_df[filtered_df['Location'] == f_loc]
        
    if filtered_df.empty:
        st.warning("No properties match the selected filters. Please adjust your criteria.")
    else:
        # --- Market Overview Metrics ---
        st.markdown("---")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Properties", len(filtered_df))
        m_col2.metric("Median Price", format_indian_currency(filtered_df['Price_INR'].median()))
        m_col3.metric("Average Price", format_indian_currency(filtered_df['Price_INR'].mean()))
        m_col4.metric("Average Area", f"{filtered_df['Area_sqft'].mean():,.0f} sq.ft")

        # --- Visualizations ---
        st.markdown("---")
        
        # 1. Price Distribution & Mumbai vs Indore (if All cities selected)
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("##### 📈 Price Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_df['Price_INR'], kde=True, ax=ax, color='teal')
            ax.set_xscale('log')
            ax.set_xlabel('Price (INR) - Log Scale')
            st.pyplot(fig)
            
        with row1_col2:
            if f_city == "All":
                st.markdown("##### 🏙️ Mumbai vs Indore (Median Price)")
                comp_df = filtered_df.groupby('City')['Price_INR'].median().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=comp_df, x='City', y='Price_INR', hue='City', palette='viridis', legend=False, ax=ax)
                ax.set_yscale('log')
                ax.set_ylabel('Median Price (INR) - Log Scale')
                st.pyplot(fig)
            else:
                st.markdown(f"##### 🏙️ Median Prices by Property Type in {f_city}")
                ptype_df = filtered_df.groupby('Property_Type')['Price_INR'].median().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=ptype_df, y='Property_Type', x='Price_INR', hue='Property_Type', palette='Set2', legend=False, ax=ax)
                ax.set_xlabel('Median Price (INR)')
                st.pyplot(fig)

        # 2. Area vs Price Scatter
        st.markdown("---")
        st.markdown("##### 📐 Area vs Price Relationship")
        fig, ax = plt.subplots(figsize=(10, 5))
        if f_city == "All":
            sns.scatterplot(data=filtered_df, x='Area_sqft', y='Price_INR', hue='City', palette='Set1', alpha=0.7, ax=ax)
        else:
            sns.scatterplot(data=filtered_df, x='Area_sqft', y='Price_INR', hue='BHK', palette='coolwarm', alpha=0.7, ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Area (sq.ft) - Log Scale')
        ax.set_ylabel('Price (INR) - Log Scale')
        st.pyplot(fig)
        
        # 3. Location Analysis
        st.markdown("---")
        st.markdown("##### 📍 Location Analysis (Top 10 by Volume)")
        top_locs = filtered_df['Location'].value_counts().nlargest(10).index
        loc_df = filtered_df[filtered_df['Location'].isin(top_locs)]
        
        if not loc_df.empty:
            loc_price = loc_df.groupby('Location')['Price_INR'].median().sort_values(ascending=False).reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=loc_price, y='Location', x='Price_INR', hue='Location', palette='Reds_d', legend=False, ax=ax)
            ax.set_xlabel('Median Price (INR)')
            ax.set_title('Median Price of Most Common Locations')
            st.pyplot(fig)


# ==========================================
# TAB 3: MODEL INFO
# ==========================================
with tab3:
    st.subheader("Model Information")
    st.write(f"**Algorithm Used:** {best_model_name}")
    if metrics:
        st.write("### Model Evaluation Metrics (Test Set)")
        st.write(f"- **R² Score:** {metrics.get('R2', 0):.4f}")
        st.write(f"- **MAE:** ₹ {metrics.get('MAE', 0):,.2f}")
        st.write(f"- **RMSE:** ₹ {metrics.get('RMSE', 0):,.2f}")
    else:
        st.write("Evaluation metrics are currently unavailable.")
    
    st.write("### Data Source")
    st.write("The models are trained strictly on the extracted and geocoded dataset containing properties from Mumbai and Indore.")
