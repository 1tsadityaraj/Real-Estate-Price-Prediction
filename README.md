# Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning

## 1. Project Overview
This project is an end-to-end Machine Learning pipeline and robust Live-Data architecture that collects, cleans, and analyzes real estate property data to estimate market prices for **Mumbai** and **Indore**. It aims to provide users, researchers, and stakeholders with data-driven insights into the property market by estimating prices based on geographical coordinates, property type, area, and spatial configuration. 

The system culminates in an interactive Streamlit web dashboard that transparently combines a **Historical ML Estimate** with a **Recently Retrieved Listing-Based Market Estimate** to generate a final objective valuation.

## 2. Features
- **Mumbai and Indore Property Analysis:** Detailed EDA comparing the two distinct property markets.
- **Data Preprocessing Pipeline:** Automated missing value handling, Outlier detection, and One-Hot Encoding.
- **Geographic Analysis:** Integration with Nominatim APIs (Geopy) to map property coordinates spatially.
- **Multi-City Price Prediction:** Unified predictive model managing disparate urban markets simultaneously.
- **Current Market Data Architecture:** Integrates a Live Provider Abstraction that searches current property listings and computes a dynamic 3-Tier Comparable Median ₹/sq.ft.
- **Graceful Fallback Protocols:** If live APIs are unavailable or unconfigured, the system safely triggers a 100% fallback to the Historical ML Estimate without crashing.
- **Streamlit Interface:** A dynamic dashboard to forecast prices, visualize market analytics, and trigger fresh API data retrievals.

## 3. Project Workflow
```text
Data Collection (Historical proxy data)
      ↓
Data Cleaning & EDA
      ↓
Geocoding (Extracting Latitude & Longitude)
      ↓
Feature Engineering (One-Hot Encoding, scaling)
      ↓
Model Training & Evaluation
      ↓
Best Model Selection (Linear Regression)
      ↓
Current Listing Retrieval (Live API Abstraction)
      ↓
Comparable Analysis (3-Tier Match for Current ₹/sq.ft)
      ↓
Streamlit Deployment (Combining ML + Current Market)
```

## 4. Dataset
- **Cities Covered:** Mumbai, Indore
- **Number of Records:** 440 unique properties
- **Important Columns:** `City`, `Location`, `Property_Type`, `Price_INR`, `Area_sqft`, `BHK`, `Bathrooms`, `Latitude`, `Longitude`
- **Target Variable:** `Price_INR` (Property Price in Indian Rupees)
- **Limitations:** The training dataset is a synthetic proxy representation generated to emulate real-world distributions for academic purposes. 

## 5. Technologies Used
- **Python** (Core language)
- **Pandas / NumPy** (Data manipulation)
- **Scikit-learn** (Machine learning pipelines)
- **BeautifulSoup / Requests** (Web scraping simulation)
- **Geopy** (Location geocoding)
- **Folium / Seaborn** (Mapping and statistical visualizations)
- **Streamlit** (Web application frontend and state caching)
- **Joblib** (Model serialization)

## 6. Machine Learning Models
The project trained and evaluated the following regression models strictly on the historical training data:
1. **Linear Regression:** Found to be highly effective at handling the heavily linearly correlated `Area_sqft` and encoded `City` properties without overfitting.
2. **Decision Tree Regressor:** Prone to overfitting on the specific high-value luxury outliers present in the Mumbai dataset.
3. **Random Forest Regressor:** Provided moderate generalization but suffered from memorizing sparse anomalies on this specific dataset size.
4. **Polynomial Regression (Degree 2):** Resulted in extreme overfitting on numerical bounds.

## 7. Model Evaluation
The performance of the models on the test split:

| Model                 |         MAE |                 MSE |        RMSE |     R² |
| --------------------- | ----------: | ------------------: | ----------: | -----: |
| **Linear Regression** | 13,623,956 | 388,416,513,203,432 | 19,708,285 | 0.5135 |
| Random Forest         | 14,803,633 | 482,099,143,830,985 | 21,956,756 | 0.3961 |
| Polynomial Regression | 17,535,481 | 655,906,326,619,255 | 25,610,668 | 0.1784 |
| Decision Tree         | 17,908,450 | 701,707,183,098,591 | 26,489,756 | 0.1211 |

**Best Performing Model:** `Linear Regression` (Highest R² and lowest errors).

*(Note: The Comparable-Only mathematical model was tested offline independently and achieved an MAE of ₹16.20M, proving its viability).*

## 8. Current Market Data & Streamlit Application
The web interface enables a fully dual-architecture analysis:
- **Historical ML Estimate:** Instantly processed via `best_model.pkl`.
- **Current Comparable Estimate:** The system queries an abstract `ConfiguredAPIDataProvider` for live listings. It calculates a median ₹/sq.ft. using matching comparable hierarchies (Exact Match → Relaxed Area → Relaxed Type).
- **Refresh Mechanism:** Users can clear the `@st.cache_data` memory to execute a fresh API retrieval timestamped on the UI.
- **Fallback Safe-Mode:** If the provider lacks commercial API keys, the system detects `LIVE DATA PROVIDER NOT CONFIGURED` and smoothly reverts the Final Estimate back to the Historical ML Prediction, preventing UI crashes.

## 9. Installation

### Clone repository
```bash
git clone https://github.com/1tsadityaraj/Real-Estate-Price-Prediction.git
cd Real-Estate-Price-Prediction
```

### Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run application
```bash
streamlit run app.py
```

## 10. Project Structure
```text
Real-Estate-Price-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/ (Data Cleaning, EDA, Model Building)
│
├── model/ (Pickled LR pipeline)
│
├── src/live_data/
│   └── provider.py (Live API Provider Abstraction)
│
├── docs/ (Final Reports, Figures, Testing Logs)
│
├── evaluate_comparables.py (Offline validation logic)
├── app.py (Streamlit UI)
├── requirements.txt
└── README.md
```

## 11. Results
- **Mumbai vs Indore Differences:** Mumbai property median prices heavily overshadow Indore's, with Mumbai regularly crossing ₹3 Crores compared to Indore's sub-₹1 Crore medians. 
- **Area-Price Dynamic:** The model accurately isolated the `City` variable to realize that identical areas (sq.ft) cost exponentially more in Mumbai than in Indore.
- **Combined Estimates:** The architecture successfully merges static historical statistical models with dynamic real-time querying logic.

## 12. Limitations
- **Dataset Size:** The ML model is trained on a 440-row proxy dataset. Commercial scaling requires scraping hundreds of thousands of records.
- **Asking vs. Transaction Price:** Live estimates rely on recently retrieved asking prices from listing APIs, which are typically inflated compared to the actual negotiated legal transaction value.
- **API Availability:** The live comparable module currently operates in a mock/fallback state requiring injection of a valid commercial Real-Estate Aggregator API key to function fully.

## 13. Future Scope
- Injection of a commercial B2B API to activate the Live Data Provider.
- Expansion to include 10+ major Indian metropolitan cities (e.g., Delhi, Bangalore, Pune).
- Inclusion of micro-economic features (proximity to transit, hospital density, interest rates).
- Exploring advanced ensemble architectures (e.g., XGBoost, Gradient Boosting) on scaled datasets.

## 14. License
This project is for academic and demonstration purposes. Datasets are simulated/mocked for educational use and do not represent proprietary third-party property listings. Estimates do not constitute legal property valuations.
