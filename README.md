# Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning

## 1. Project Overview
This project is an end-to-end Machine Learning pipeline that collects, cleans, and analyzes real estate property data to predict market prices for **Mumbai** and **Indore**. It aims to provide users, researchers, and stakeholders with data-driven insights into the property market by predicting prices based on geographical coordinates, property type, area, and spatial configuration. The system culminates in an interactive Streamlit web dashboard.

## 2. Features
- **Mumbai and Indore Property Analysis:** Detailed EDA comparing the two distinct property markets.
- **Data Preprocessing Pipeline:** Automated missing value handling and outlier detection.
- **Geographic Analysis:** Integration with Nominatim APIs (Geopy) to map property coordinates.
- **Multi-City Price Prediction:** Unified predictive model managing disparate urban markets.
- **Multiple Regression Models:** Evaluates and compares four distinct ML algorithms.
- **Streamlit Interface:** A dynamic dashboard to forecast prices and visualize market analytics.

## 3. Project Workflow
```text
Data Collection (Simulated Web Scraping)
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis (EDA)
      ↓
Geocoding (Extracting Latitude & Longitude)
      ↓
Feature Engineering (One-Hot Encoding, scaling)
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection (Linear Regression)
      ↓
Streamlit Deployment
```

## 4. Dataset
- **Cities Covered:** Mumbai, Indore
- **Number of Records:** 440 unique properties
- **Important Columns:** `City`, `Location`, `Property_Type`, `Price_INR`, `Area_sqft`, `BHK`, `Bathrooms`, `Latitude`, `Longitude`
- **Target Variable:** `Price_INR` (Property Price in Indian Rupees)
- **Limitations:** The dataset is a synthetic proxy representation generated to emulate real-world distributions for academic purposes. Due to limitations in public scraping, coordinate lookup success rates vary slightly.

## 5. Technologies Used
- **Python** (Core language)
- **Jupyter Notebook** (Pipeline development)
- **Pandas / NumPy** (Data manipulation)
- **Matplotlib / Seaborn** (Data visualization)
- **Scikit-learn** (Machine learning)
- **BeautifulSoup / Requests** (Web scraping simulation)
- **Geopy** (Location geocoding)
- **Folium** (Interactive mapping)
- **Streamlit** (Web application frontend)
- **Git / GitHub** (Version control)

## 6. Machine Learning Models
The project trained and evaluated the following regression models:
1. **Linear Regression:** Found to be highly effective at handling the heavily linearly correlated `Area_sqft` and encoded `City` properties.
2. **Decision Tree Regressor:** Prone to overfitting on the specific high-value outliers present in the Mumbai dataset.
3. **Random Forest Regressor:** Provided moderate generalization but was computationally heavier than standard regression for this specific feature shape.
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

## 8. Streamlit Application
The web interface allows users to input custom property metrics:
- **City & Location Selection:** Dynamically updates available locations based on the chosen city (Mumbai/Indore).
- **Property Inputs:** BHK, Area, Bathrooms, and Property Type.
- **Price Prediction:** Instantly outputs the estimated property value formatted in Indian Rupees using the loaded `best_model.pkl`.
- **Market Analysis:** Shows live data distribution metrics, median prices, and cross-city comparisons fetched directly from the clean dataset.

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
│   │   ├── mumbai_property_data.csv
│   │   ├── indore_property_data.csv
│   │   └── mumbai_indore_property_data.csv
│   │
│   └── processed/
│       ├── cleaned_mumbai_indore_property_data.csv
│       ├── geocoded_mumbai_indore_property_data.csv
│       └── ml_ready_mumbai_indore_data.csv
│
├── notebooks/
│   ├── web_scraping.ipynb
│   ├── data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── geocoders_maps.ipynb
│   ├── feature_engineering.ipynb
│   └── model_building.ipynb
│
├── model/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_summary.json
│
├── docs/
│   ├── figures/
│   └── testing_report.md
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Results
- **Mumbai vs Indore Differences:** Mumbai property median prices heavily overshadow Indore's, with Mumbai regularly crossing ₹3 Crores compared to Indore's sub-₹1 Crore medians. 
- **Area-Price Dynamic:** The model accurately isolated the `City` variable to realize that identical areas (sq.ft) cost exponentially more in Mumbai than in Indore.
- **Best Model:** Linear Regression yielded a 0.51 R² score, providing reliable baseline market estimations.

## 12. Limitations
- **Dataset Size:** The model is trained on a 440-row proxy dataset. Real-world property prediction models require hundreds of thousands of rows.
- **Data Freshness:** Real estate is highly sensitive to time (interest rates, development projects); this dataset represents a static snapshot.
- **Geocoding Constraints:** Free geocoding APIs (Nominatim) occasionally fail to match specific hyper-local building names.

## 13. Future Scope
- Integration with live real-estate APIs for real-time data streaming.
- Expansion to include 10+ major Indian metropolitan cities (e.g., Delhi, Bangalore, Pune).
- Inclusion of micro-economic features (proximity to transit, hospital density, crime rates).
- Exploring advanced ensemble architectures (e.g., XGBoost, Gradient Boosting) on larger datasets.

## 14. License
This project is for academic and demonstration purposes. Datasets are simulated/mocked for educational use and do not represent proprietary third-party property listings.
