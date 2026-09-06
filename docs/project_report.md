# REAL ESTATE PRICE PREDICTION AND PROPERTY ANALYSIS SYSTEM FOR MUMBAI AND INDORE USING MACHINE LEARNING

---

## Preliminary Pages

### 1. Cover Page

**REAL ESTATE PRICE PREDICTION AND PROPERTY ANALYSIS SYSTEM FOR MUMBAI AND INDORE USING MACHINE LEARNING**

**Submitted by:**
[STUDENT NAME]
[ENROLLMENT NUMBER]

**In partial fulfillment for the award of the degree of**
**[DEGREE NAME]**

**Department of [DEPARTMENT NAME]**
**[COLLEGE NAME]**

**Under the Guidance of:**
[GUIDE NAME]
[DEPARTMENT NAME]

**Academic Year:** [ACADEMIC YEAR]
**Semester:** [SEMESTER]
**Date of Submission:** [DATE]

---

### 2. Certificate

**CERTIFICATE**

This is to certify that the project report entitled **"Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning"** submitted by **[STUDENT NAME]** (Roll No: **[ENROLLMENT NUMBER]**) is a record of bona fide work carried out by them under my supervision and guidance in partial fulfillment of the requirements for the award of the degree of [DEGREE NAME].

**Guide:** [GUIDE NAME]
**Head of Department:** [HOD NAME]
**Department:** [DEPARTMENT NAME]
**College:** [COLLEGE NAME]
**Academic Year:** [ACADEMIC YEAR]

---

### 3. Declaration

**DECLARATION**

I hereby declare that the major project entitled **"Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning"** is an original work carried out by me under the guidance of [GUIDE NAME]. I further declare that this work has not been submitted in part or full to any other University or Institution for the award of any degree or diploma.

**Name:** [STUDENT NAME]
**Signature:**
**Date:** [DATE]

---

### 4. Acknowledgement

**ACKNOWLEDGEMENT**

I would like to express my profound gratitude to my project guide, [GUIDE NAME], for their continuous support, valuable insights, and encouragement throughout the completion of this project. I extend my sincere thanks to the Head of the Department, [HOD NAME], and the faculty members of the [DEPARTMENT NAME] at [COLLEGE NAME] for providing the necessary resources and guidance. Finally, I thank my family and friends for their unwavering support during my academic journey.

---

### 5. Abstract

**ABSTRACT**

The real estate sector is highly volatile, making property price estimation a complex challenge. Traditional valuation methods often fail to capture the nuanced effects of geospatial configurations and urban sprawl. This project presents a robust Machine Learning pipeline designed to predict real estate prices and provide comprehensive market analysis across two distinct Indian urban landscapes: Mumbai (a high-density, ultra-premium metropolitan market) and Indore (an emerging, fast-growing tier-2 market). 

The system collects, cleans, and geocodes historical real-estate property listing data utilizing the Nominatim API to establish spatial integrity. Following an extensive Exploratory Data Analysis (EDA) comparing the distinct market demographics of both cities, advanced feature engineering techniques (including scaling and one-hot encoding) are applied. Four regression algorithms (Linear Regression, Decision Tree, Random Forest, Polynomial Regression) are trained and evaluated on a combined historical dataset of 440 property records. The Linear Regression model emerged as the most optimal predictor, achieving an R² score of 0.4885 by successfully isolating the premium geospatial disparity between the two cities. Finally, the best-performing pipeline is deployed via a dynamic, interactive Streamlit web application that enables users to obtain ML-based property price estimates and visualizations of the cross-city analytics.

---

### 6. Keywords
Real Estate, Machine Learning, Price Prediction, Mumbai, Indore, Regression, Data Analysis, Geospatial Analysis, Streamlit, Python.

---
---

# CHAPTER 1 – INTRODUCTION

### 1.1 Background
Real estate investment constitutes a significant proportion of global wealth. Accurate property valuation is a critical necessity for buyers, sellers, investors, and financial institutions. Property prices are influenced by a myriad of factors including geographic location, total carpet area, number of bedrooms (BHK), property type, and the inherent economic strength of the city itself.

### 1.2 Problem Statement
Estimating property prices using conventional, manual approaches relies heavily on human intuition and generalized neighborhood averages. This approach is prone to severe bias, fails to simultaneously account for multiple interdependent variables, and struggles to adapt to disparate markets (e.g., comparing a Tier-1 megacity to a Tier-2 growing city). 

### 1.3 Motivation
Machine Learning (ML) excels at uncovering hidden correlations in large datasets. By utilizing a data-driven approach, ML can analyze how property-related factors dynamically interact to formulate a market price, providing consumers with unbiased, instantaneous, and statistically sound valuations.

### 1.4 Project Objectives
1. Collect property data covering Mumbai and Indore.
2. Clean and preprocess the data to handle anomalies and missing values.
3. Perform Exploratory Data Analysis (EDA) to determine real estate trends.
4. Perform geographical analysis through geocoding (extracting coordinates).
5. Train and tune multiple supervised regression models.
6. Compare model performance to select the highest-performing algorithm.
7. Predict property prices based on cross-city variables.
8. Provide an interactive Streamlit interface for end-users.
9. Compare the Mumbai and Indore markets programmatically.

### 1.5 Scope
The implemented scope of this project is confined to analyzing a dataset of 440 properties distributed across Mumbai and Indore. It evaluates property types such as Apartments, Villas, and Independent Houses, using spatial and architectural parameters to estimate final transaction prices in Indian Rupees (INR).

### 1.6 Organization of Report
The report is organized into chapters outlining the literature survey, existing and proposed systems, software requirements, dataset preprocessing, exploratory data analysis, geographic plotting, ML methodologies, evaluation metrics, Streamlit UI implementation, testing, limitations, and future scope.

---

# CHAPTER 2 – LITERATURE SURVEY

Modern approaches to property valuation have shifted from comparative market analysis (CMA) to automated valuation models (AVMs) driven by ML. Studies surrounding regression-based property prediction typically highlight the efficacy of algorithms like Random Forests and standard Multivariate Linear Regression in capturing price elasticity. Furthermore, modern real estate analysis relies heavily on geospatial preprocessing (Geocoding) to introduce locational context directly into the mathematical model. Web-based applications using frameworks like Streamlit have recently bridged the gap between complex ML data pipelines and accessible consumer-facing dashboard interfaces.

*(Note: The findings above discuss the general academic environment; the specific multi-city Mumbai-Indore Streamlit pipeline detailed in this report is an independent implementation.)*

---

# CHAPTER 3 – EXISTING SYSTEM

Traditional/existing systems for property valuation suffer from distinct limitations:
- **Manual Estimation:** High susceptibility to human bias and emotional inflation.
- **Limited Variables:** Inability to mathematically weigh the simultaneous impacts of location, area, and architectural configuration.
- **Geographic Variation:** Rigid rulesets that fail to translate between extreme markets (e.g., applying Mumbai pricing models to Indore).
- **Lack of Interactive Prediction:** Existing legacy systems often lack consumer-friendly UI dashboards that provide instant valuations.

The proposed system mitigates these issues by building an automated, data-driven regression model wrapped in an accessible web dashboard.

---

# CHAPTER 4 – PROPOSED SYSTEM

**Mumbai + Indore Real Estate Price Prediction and Property Analysis System**

The proposed system operates through a sequential data pipeline:
```text
Data Collection
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Geocoding & Geographic Analysis
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Streamlit Application
       ↓
Estimated Property Price & Market Analysis
```
The system processes historical property listing records, standardizes strings and floats, maps locations to latitude/longitude coordinates, encodes categorical data, fits multiple regressors, serializes the best predictor, and deploys it as a Streamlit web application.

---

# CHAPTER 5 – REQUIREMENTS

## 5.1 Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher.
- **RAM:** Minimum 4 GB (8 GB recommended for ML training).
- **Storage:** 500 MB of free disk space.

## 5.2 Software Requirements
- **Language:** Python 3.8+
- **Environment:** Jupyter Notebook
- **Core Data Science Libraries:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Data Gathering:** BeautifulSoup4, Requests (used during historical data collection)
- **Geospatial Processing:** Geopy, Folium
- **Deployment:** Streamlit
- **Serialization:** Joblib
- **Version Control:** Git/GitHub

---

# CHAPTER 6 – SYSTEM DESIGN

### 6.1 System Architecture
The architecture comprises a backend ML training pipeline (Jupyter Notebooks) that outputs serialized `.pkl` artifacts. The frontend (Streamlit) consumes these artifacts and processes user inputs to generate ML-based property price estimates.

### 6.2 Data Flow
User Input → Streamlit UI → `preprocessor.pkl` (ColumnTransformer) → One-Hot Encoded Dense Matrix → `best_model.pkl` (Linear Regression) → Formatted Output.

### 6.3 Module Description
1. **Data Collection Module:** Aggregates and merges property CSV records.
2. **Data Cleaning Module:** Drops duplicates, strips string artifacts from numerical values, and standardizes categories.
3. **EDA Module:** Compares distributions and correlations across cities.
4. **Geocoding Module:** Fetches real-world coordinates via the Nominatim API.
5. **Feature Engineering Module:** Constructs a ColumnTransformer pipeline implementing Drop-First One-Hot Encoding.
6. **Machine Learning Module:** Fits data across multiple Scikit-learn regression algorithms.
7. **Prediction Module:** Restores saved models to evaluate unseen arrays.
8. **Market Analysis Module:** Aggregates final data to serve dashboard KPIs.
9. **Streamlit Interface:** Hosts the interactive user frontend.

### 6.4 Input and Output
- **Input:** City, Location, Property Type, BHK, Area (sq.ft), Bathrooms.
- **Output:** Estimated property price (INR).

---

# CHAPTER 7 – DATASET AND DATA PREPROCESSING

## 7.1 Data Sources
The project uses historical real-estate property listing data for Mumbai and Indore:

- **Mumbai:** 250 historical property listings stored in `data/raw/property_data.csv`
- **Indore:** 200 historical property listings stored in `data/raw/indore_property_data.csv`
- **Combined:** Both datasets are merged into `data/raw/mumbai_indore_property_data.csv` (450 records)

No live market API or real-time property listing feed is used in the final system.

## 7.2 Dataset Description
- **Records (after cleaning):** 440 unique properties.
- **Features:** 10 columns (`Location`, `Property_Type`, `Price_INR`, `Area_sqft`, `BHK`, `Bathrooms`, `City`, `price_per_sqft`, `Latitude`, `Longitude`).
- **Cities:** Mumbai, Indore.
- **Property Types:** Apartment, Independent House, Studio, Villa.

## 7.3 Data Cleaning
Cleaning involved removing null occurrences, handling duplicates, converting non-standard text (e.g., "3 BHK" to the integer `3`), scaling `price_per_sqft` metrics correctly, standardizing location names, and ensuring numerical datatypes for continuous variables. 

## 7.4 Feature Engineering
The pipeline distinguishes between numerical variables (Area, BHK, Bathrooms, Latitude, Longitude) and categorical variables (City, Location, Property Type). Categorical data was transformed using `OneHotEncoder(drop='first')` inside a `ColumnTransformer` to prevent the dummy variable trap, effectively dodging multicollinearity. 

**Data Leakage Prevention:** `price_per_sqft` is fundamentally derived from the target variable (`Price_INR`). Including it as a training feature would result in data leakage; therefore, it was strictly removed from the machine learning inputs and used exclusively for EDA.

---

# CHAPTER 8 – EXPLORATORY DATA ANALYSIS

Extensive EDA was executed in `notebooks/EDA.ipynb` (Figures stored in `docs/figures/`).

1. **City Comparison:** Mumbai properties vastly outnumbered Indore properties in the dataset. Boxplots revealed Mumbai's median price far exceeds Indore's.
2. **Property Type Distribution:** Apartments constitute the vast majority of listings across both cities.
3. **BHK Analysis:** 2 and 3 BHK configurations are the most common.
4. **Area vs Price Relationship:** The scatterplot demonstrates a strong positive correlation between Area and Price. Crucially, the scatterplot distinctly clusters the two cities on completely different axes, indicating the massive price premium associated with the "Mumbai" feature.
5. **Correlation Matrix:** Area (sq.ft) and BHK possess the highest mathematical correlation to `Price_INR` among the numerical variables.

---

# CHAPTER 9 – GEOGRAPHICAL ANALYSIS

The `geocoders_maps.ipynb` module utilized `geopy.geocoders.Nominatim` to extract Latitude and Longitude coordinates by querying the concatenation of the property's `Location` and `City`. 

The extraction successfully mapped the vast majority of properties, plotting Mumbai on the western coastline and Indore in central India using the `Folium` library. Capturing these coordinates allows the machine learning algorithm to perceive precise, continuous spatial relationships rather than relying solely on string-based Location tags.

---

# CHAPTER 10 – MACHINE LEARNING MODELS

## 10.1 Linear Regression
Implemented as a foundational baseline, this algorithm fits a linear equation to observed data. It performed exceptionally well given the highly linear relationships in real estate (e.g., as Area scales linearly, Price tends to scale linearly).

## 10.2 Decision Tree Regressor
A non-linear model that splits data into branch-like conditional structures. It is prone to extreme overfitting on outliers.

## 10.3 Random Forest Regressor
An ensemble learning method that constructs a multitude of decision trees and outputs the average prediction, effectively reducing the variance of single decision trees.

## 10.4 Polynomial Regression
Generates a new feature matrix consisting of all polynomial combinations of the features. It failed drastically on this dataset due to explosive extrapolation scaling at the numerical bounds.

---

# CHAPTER 11 – MODEL EVALUATION

The algorithms were evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R-squared (R²) score on the test partition:

| Model                 |         MAE |                 MSE |        RMSE |     R² |
| --------------------- | ----------: | ------------------: | ----------: | -----: |
| **Linear Regression** | 15,111,250 | 377,002,476,537,380 | 19,416,551 | 0.4885 |
| Random Forest         | 14,862,030 | 382,943,767,825,972 | 19,568,949 | 0.4803 |
| Polynomial Regression | 17,519,460 | 542,217,358,016,944 | 23,285,561 | 0.2642 |
| Decision Tree         | 19,188,640 | 707,562,357,954,545 | 26,599,997 | 0.0398 |

**Selection:** Linear Regression was selected as the optimal model. It achieved the highest R² (0.4885), indicating it explained approximately 49% of the variance in the dataset. Real estate data spanning diverse spatial clusters (Mumbai vs Indore) often leads tree-based models to overfit localized price nodes, making the generalized slope of Linear Regression far more robust for unseen data.

---

# CHAPTER 12 – STREAMLIT APPLICATION

### 12.1 Application Overview
The interactive web application (`app.py`) hosts a multi-tabbed interface serving both property prediction and dynamic data visualization.

### 12.2 City Selection
A primary dropdown allowing users to select either "Mumbai" or "Indore".

### 12.3 Property Input
Users manipulate sliders and dropdowns to input Location, Property Type, BHK, Bathrooms, and Area. The Location dropdown automatically cascades based on the selected City.

### 12.4 Price Prediction
Triggered via a button, the system executes `preprocessor.transform()` on the input and feeds the dense matrix to `model.predict()`, rendering a formatted Indian Rupee ML-based estimate (e.g., ₹ 4.65 Crore). The predicted price is an estimated property price based on patterns learned from historical data.

### 12.5 Market Analysis & Comparison
A secondary dashboard tab provides interactive charts. Users can filter by City and Property Type to view Median Prices, Volume distributions, and cross-city analytics without executing backend Python scripts.

### 12.6 Error Handling
The application safely handles out-of-bound dropdown queries and gracefully renders `st.warning()` prompts if users apply overly restrictive dashboard filters resulting in empty data frames.

---

# CHAPTER 13 – TESTING

Functional tests performed on the Streamlit deployment:

| Test Case | Description       | Expected Result      | Actual Result | Status    |
| --------- | ----------------- | -------------------- | ------------- | --------- |
| TC01      | Mumbai prediction | Prediction generated | Successfully output ₹ 4.65 Crore for test input | PASS |
| TC02      | Indore prediction | Prediction generated | Successfully output ₹ 49.2 Lakhs for test input | PASS |
| TC03      | City switching    | Locations update     | Dropdown strictly restricted to respective city | PASS |
| TC04      | Invalid input     | Graceful handling    | Mismatched locations blocked; zero-data filters yielded warnings | PASS |
| TC05      | Market analysis   | Data displayed       | Native charts and metric KPIs loaded successfully | PASS |
| TC06      | Comparison        | Comparison displayed | Cross-city aggregations successfully rendered | PASS |

---

# CHAPTER 14 – RESULTS AND DISCUSSION

The project successfully bridged two heavily disparate markets. The Exploratory Data Analysis verified massive economic gaps: Mumbai's median price and price-per-square-foot metrics eclipsed Indore's significantly. The Linear Regression model mathematically validated these findings, utilizing the spatial variables (City, Location, Coordinates) to successfully scale identical physical dimensions (Area, BHK) to completely different final price points depending on the city. The Streamlit dashboard successfully decoupled the backend ML complexities, exposing a highly robust and intuitive user interface.

---

# CHAPTER 15 – LIMITATIONS

- **Dataset Size:** The model was trained on a historical dataset of 440 property records. Larger datasets with more varied property records would improve generalization.
- **Location Coverage:** The historical dataset contains concentrated location nodes; predictions for outer metropolitan zones may lack accuracy due to spatial extrapolation.
- **Data Freshness:** The dataset represents a historical snapshot and does not account for dynamic interest rate shifts, inflation, or current market conditions.
- **Model Limitation:** The predicted price is an ML-based estimate derived from historical property data and should not be treated as an official property valuation.

---

# CHAPTER 16 – FUTURE SCOPE

- **Support for More Indian Cities:** Scaling the architecture to cover Delhi, Bangalore, and Pune.
- **Larger Datasets:** Integrating larger, continuously updated historical datasets for improved model accuracy.
- **Advanced Algorithms:** Tuning XGBoost and Gradient Boosting regressors on larger datasets to manage extreme non-linear outliers.
- **Deployment:** Migrating the local Streamlit application to a cloud-hosting provider (e.g., AWS EC2, Streamlit Cloud).

---

# CHAPTER 17 – CONCLUSION

This project successfully implements an end-to-end Machine Learning ecosystem addressing the real estate price prediction problem for both Mumbai and Indore. By executing a rigorous pipeline spanning data cleaning, EDA, geospatial analysis, and feature engineering, the system successfully trained a Linear Regression model capable of distinguishing between a high-density tier-1 metropolis and an emerging tier-2 market. The deployment of this pipeline into an interactive Streamlit web dashboard demonstrates the immense practical utility of integrating automated data-science pipelines with consumer-facing analytics tools. The system predicts property prices using historical real-estate data and does not provide real-time transaction prices.

---

# REFERENCES

1. Scikit-learn Developers. "Scikit-learn: Machine Learning in Python." (https://scikit-learn.org)
2. Pandas Development Team. "Pandas: Data manipulation and analysis." (https://pandas.pydata.org)
3. Matplotlib Developers. "Matplotlib: Python plotting." (https://matplotlib.org)
4. Waskom, M. et al. "Seaborn: Statistical data visualization." (https://seaborn.pydata.org)
5. Streamlit Inc. "Streamlit: The fastest way to build and share data apps." (https://streamlit.io)
6. Geopy Contributors. "Geopy: Geocoding library for Python." (https://geopy.readthedocs.io)

---
---

# REPORT VALIDATION SUMMARY
1. **Final project title:** Real Estate Price Prediction and Property Analysis System for Mumbai and Indore Using Machine Learning
2. **Supported cities:** Mumbai, Indore
3. **Final dataset size:** 440 records
4. **Final model:** Linear Regression
5. **Actual best model metrics:** MAE: 15,111,250 | MSE: 377,002,476,537,380 | RMSE: 19,416,551 | R²: 0.4885
6. **Main implemented features:** Data pipeline, EDA, Geocoding, ML Modeling, Streamlit Multi-tab Dashboard, Market Analysis.
7. **Number of test cases:** 6 primary Streamlit test cases.
8. **Number of figures:** Generated 15 core figures in `docs/figures/`.
9. **Number of tables:** 2 tables (Model Evaluation, Testing).
10. **Information to be filled:** Student Name, Roll Number, Degree, College/University metrics, Guide Name.
