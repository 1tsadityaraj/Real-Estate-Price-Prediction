# Real Estate Price Prediction and Property Analysis System Using Machine Learning

## Overview
This project is a comprehensive Machine Learning pipeline for predicting real estate prices in Mumbai based on property characteristics. It follows an end-to-end data science lifecycle, from gathering and cleaning data to training predictive models and deploying a user-friendly Streamlit web application.

## Features
* **Web Scraping:** Modules implemented to scrape property listing data from real estate portals.
* **Data Preprocessing:** Robust cleaning strategies for handling missing values, duplicates, and formatting errors.
* **Exploratory Data Analysis (EDA):** Deep dive into the data using visualizations to understand price distributions and feature correlations.
* **Geographical Analysis:** Integration with Geopy/Nominatim to map location text to exact geographical coordinates for spatial analysis.
* **Machine Learning:** End-to-end feature engineering and training pipeline.
* **Model Comparison:** Evaluation of multiple regression models to find the best performing algorithm.
* **Price Prediction:** Highly accurate pricing estimates based on property inputs.
* **Streamlit Application:** An interactive, locally deployable web application for real-time predictions.

## Technologies
* **Language:** Python 3.12
* **Notebooks:** Jupyter Notebook
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Folium
* **Geocoding:** Geopy
* **Machine Learning:** Scikit-learn (Linear Regression, Decision Trees, Random Forest, Polynomial Features)
* **Web Scraping:** BeautifulSoup, Requests (Framework established)
* **Deployment:** Streamlit
* **Version Control:** Git / GitHub

## Project Structure
```text
Real-Estate-Price-Prediction/
│
├── data/
│   ├── raw/                       # Initial scraped datasets
│   └── processed/                 # Cleaned and geocoded datasets ready for ML
│
├── notebooks/
│   ├── web_scraping.ipynb         # Data collection framework
│   ├── data_cleaning.ipynb        # Data preparation and cleaning
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── geocoders_maps.ipynb       # Geographical coordinate mapping
│   ├── feature_engineering.ipynb  # Feature transformation and encoding
│   └── model_building.ipynb       # ML model training and evaluation
│
├── model/
│   ├── best_model.pkl             # Serialized top-performing model
│   └── preprocessor.pkl           # Serialized ColumnTransformer pipeline
│
├── screenshots/                   # Application demonstration images
├── docs/                          # Testing reports and additional documentation
│   └── testing_report.md
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Dataset
The dataset represents real estate property listings in Mumbai (e.g., 99acres property features). Due to scraping restrictions, the active dataset uses an offline property data cache gathered previously using the exact scraping methodology provided in the project notebooks. It includes vital attributes such as `Location`, `Property_Type`, `Area_sqft`, `BHK`, `Bathrooms`, and `Price`.

## Machine Learning Models
The following algorithms were rigorously evaluated during the model building phase:
* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Polynomial Regression

## Model Evaluation
Based on internal testing in `model_building.ipynb`, **Linear Regression** was selected as the best performing and most stable model. The polynomial regression tended to overfit, and linear regression provided the best balance of R² score and computational efficiency for this specific feature set, resulting in its serialization to `best_model.pkl`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/1tsadityaraj/Real-Estate-Price-Prediction.git
cd Real-Estate-Price-Prediction
```

2. Create a virtual environment:
**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run Application
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

## Limitations
* Web scraping modules are subject to changes in the target website's HTML structure and CAPTCHA restrictions.
* Model predictions are heavily weighted by the historical data boundaries; properties far outside the norm (e.g., extreme luxury mansions) may experience lower prediction accuracy.

## Future Scope
* **Live API Integration:** Replacing static scraping with live API hooks to continuously ingest real estate data.
* **Deep Learning:** Experimenting with Neural Networks to capture complex non-linear relationships.
* **Enhanced UI:** Expanding the Streamlit application to include dynamic map layers and investment ROI estimators.
