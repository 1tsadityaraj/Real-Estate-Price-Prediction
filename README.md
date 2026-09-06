# Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning

## 1. Project Overview
This project is an end-to-end Machine Learning pipeline that collects, cleans, and analyzes real estate property data to estimate market prices for **Mumbai** and **Indore**. It aims to provide users, researchers, and stakeholders with data-driven insights into the property market by estimating prices based on historical geographical coordinates, property type, area, and spatial configuration. 

The system culminates in an interactive Streamlit web dashboard that leverages historical statistical models to generate an objective property valuation, entirely offline.

## 2. Features
- **Mumbai and Indore Property Analysis:** Detailed Exploratory Data Analysis (EDA) comparing two distinct property markets.
- **Data Preprocessing Pipeline:** Automated missing value handling, Outlier detection, Geocoding, and One-Hot Encoding.
- **Multi-City Price Prediction:** Unified predictive model managing disparate urban markets simultaneously.

## 3. Project Workflow
```text
Historical Property Dataset
      ↓
Data Cleaning & Normalization
      ↓
Exploratory Data Analysis
      ↓
Geocoding (Extracting Latitude & Longitude)
      ↓
Feature Engineering (One-Hot Encoding, Median Imputation)
      ↓
Model Training & Evaluation
      ↓
Best Model Selection (Linear Regression)
      ↓
Saved Model + Preprocessor Artifacts
      ↓
Streamlit Prediction Application
      ↓
Predicted Property Price
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
- **Geopy** (Location geocoding)
- **Folium / Seaborn / Matplotlib** (Mapping and statistical visualizations)
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
| **Linear Regression** | 15,111,250 | 377,002,476,537,380 | 19,416,551 | 0.4885 |
| Random Forest         | 14,862,030 | 382,943,767,825,972 | 19,568,949 | 0.4803 |
| Polynomial Regression | 17,519,460 | 542,217,358,016,944 | 23,285,561 | 0.2642 |
| Decision Tree         | 19,188,640 | 707,562,357,954,545 | 26,599,997 | 0.0398 |

**Best Performing Model:** `Linear Regression` (Consistently highest predictive stability across both cities).

*The final system predicts property prices using historical real-estate data. It does not provide real-time transaction prices.*

## 8. Installation

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

## 9. Project Structure
```text
Real-Estate-Price-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/ (Data Cleaning, EDA, Model Building, Geocoding)
│
├── model/ (Pickled ML pipeline and preprocessor)
│
├── docs/ (Final Reports, Presentation Materials)
│
├── app.py (Streamlit UI)
├── requirements.txt
└── README.md
```

## 10. Results
- **Mumbai vs Indore Differences:** Mumbai property median prices heavily overshadow Indore's, with Mumbai regularly crossing ₹3 Crores compared to Indore's sub-₹1 Crore medians. 
- **Area-Price Dynamic:** The model accurately isolated the `City` variable to realize that identical areas (sq.ft) cost exponentially more in Mumbai than in Indore.

## 11. Limitations
- **Dataset Size:** The ML model is trained on a 440-row dataset. Commercial scaling requires scraping hundreds of thousands of records.
- **Outlier Extremes:** Because it is a linear model operating on historical data, attempting to predict an extreme outlier (e.g., a 15 BHK apartment with 20,000 sq.ft) may yield exponentially extrapolated prices outside the training distribution.

## 12. Future Scope
- Expansion to include 10+ major Indian metropolitan cities (e.g., Delhi, Bangalore, Pune).
- Inclusion of micro-economic features (proximity to transit, hospital density, interest rates).
- Exploring advanced ensemble architectures (e.g., XGBoost, LightGBM) on scaled datasets.

## 13. License
This project is for academic and demonstration purposes. Datasets are simulated/mocked for educational use and do not represent proprietary third-party property listings. Estimates do not constitute legal property valuations.
